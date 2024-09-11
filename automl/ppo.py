import numpy as np
import torch
from torch.optim import Adam, SGD
import gym
import time
import dic.automl.core as core

import spring.linklink as link
import importlib
import torch


def _build_func(import_path: str):
    module_spl = import_path.split('.')
    module, func_name = '.'.join(module_spl[:-1]), module_spl[-1]
    try:
        func = importlib.import_module(module).__dict__[func_name]
    except Exception:
        print(module)
        print(func_name)
        importlib.import_module(module)
        raise ValueError('unexpected func: ' + import_path)

    return func


def _build_cls(import_path: str):
    module_spl = import_path.split('.')
    module, cls_name = '.'.join(module_spl[:-1]), module_spl[-1]
    try:
        cls = importlib.import_module(module).__dict__[cls_name]
    except Exception:
        print(module)
        print(cls_name)
        importlib.import_module(module)
        raise ValueError('unexpected cls: ' + import_path)

    return cls


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # assert self.ptr < self.max_size  # buffer has to have room so you can store
        if self.ptr == self.max_size:
            self.obs_buf[:-1] = self.obs_buf[1:]
            self.act_buf[:-1] = self.act_buf[1:]
            self.rew_buf[:-1] = self.rew_buf[1:]
            self.val_buf[:-1] = self.val_buf[1:]
            self.logp_buf[:-1] = self.logp_buf[1:]
            self.adv_buf[:-1] = self.adv_buf[1:]
            self.ret_buf[:-1] = self.ret_buf[1:]
            self.ptr = self.ptr - 1
            self.path_start_idx = self.path_start_idx - 1

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size  # buffer has to be full before you can get
        # self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_buf = self.adv_buf[:ptr]
        adv_mean = adv_buf.mean()
        adv_std = adv_buf.std()
        adv_buf = (adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class LFSBaseAgent(object):
    def __init__(self, group=None, num_groups=None, **kwargs):
        self.group = group
        self.num_groups = num_groups

        # Instantiate environment
        obs_dim = kwargs['obs_dim']
        act_dim = kwargs['act_dim']
        gamma = kwargs['gamma']
        lam = kwargs['lam']
        self.clip_ratio = kwargs['clip_ratio']

        # Set up experience buffer
        buf_len = num_groups * kwargs['samples_per_group_in_buf']
        self.buf = PPOBuffer(obs_dim, act_dim, buf_len, gamma, lam)

        # Create actor-critic module
        self.ac = _build_cls(kwargs['ac'])(obs_dim, act_dim, **kwargs['ac_kwargs'])

        # Sync params across processes
        for name, p in sorted(self.ac.state_dict().items(), key=lambda x: x[0]):
            link.broadcast(p, rank)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

    def sample_subfunction(self, env):
        # Main loop: collect experience in env and update/log each epoch
        # Update obs (critical!)
        self.old_obs = self.build_obs(env)
        self.old_a, self.old_v, self.old_logp = self.ac.step(
            torch.as_tensor(self.old_obs, dtype=torch.float32)
        )

        data = {
            'o': self.old_obs,
            'a': self.old_a,
            'v': self.old_v,
            'logp': self.old_logp,
        }

        new_data = {}
        for k, v in data.items():
            v = torch.as_tensor(v).contiguous().cuda()
            link.broadcast(v, 0, group_idx=self.group)
            new_data[k] = v.numpy()

        self.old_obs, self.old_a, self.old_v, self.old_logp = [
            new_data[i] for i in ['o', 'a', 'v', 'logp']
        ]

        return self.old_a

    def build_obs(self, env):
        # mean: [eval/Y:6.4939361, eval/bpp:1.1662843,
        # eval/entropy_total:1.1625074, eval/grad:3.0616972,
        # eval/grad_simple:17.5260391, eval/grad_simple_v2:6.0096078,
        # eval/ms(db):20.7881870, eval/mse:11.0425453,
        # eval/msssim:0.9912772, eval/psnr:37.8700447,
        # eval/rmse:3.2906773, eval/smoothL1:2.0102987,
        # eval/total:1.9650711, eval/tv:0.0000144,
        # eval/y_bpp:1.1448499, eval/y_entropy_loss:1.1414715,
        # eval/yuv:5.1962466, eval/z_bpp:0.0214344, eval/z_entropy_loss:0.0210359]
        return np.asarray([
            float(env['eval/' + i]) for i in
            ['Y', 'bpp', 'entropy_total', 'grad', 'grad_simple',
             'grad_simple_v2', 'ms(db)', 'mse', 'msssim', 'psnr',
             'rmse', 'smoothL1', 'total', 'tv', 'y_bpp', 'y_entropy_loss',
             'yuv', 'z_bpp', 'z_entropy_loss']
        ])

    def store_helper(self, env, r):
        next_o = self.build_obs(env)

        data = {
            'o': self.old_obs,
            'next_o': next_o,
            'a': self.old_a,
            'r': r,
            'v': self.old_v,
            'logp': self.old_logp,
        }

        all_data = {}
        for k, v in data.items():
            v = torch.as_tensor(v).contiguous().cuda()
            allv = torch.zeros(link.get_world_size(), *v.shape).contiguous().cuda()
            link.allgather(allv, v)
            ans = []
            for idx, per_v in enumerate(allv.cpu()):
                if idx % (link.get_world_size() // self.num_groups) != 0:
                    continue
                ans.append(per_v.numpy())
            all_data[k] = ans

        for o, a, r, v, logp, next_o in zip(
                *[all_data[i] for i in ['o', 'a', 'r', 'v', 'logp', 'next_o']]):
            self.buf.store(o, a, r, v, logp)
            _, next_v, _ = self.ac.step(torch.as_tensor(next_o, dtype=torch.float32))
            self.buf.finish_path(next_v)

    def step(self, env, reward):
        info = {}
        start_time = time.time()

        # save and log
        self.store_helper(env, reward)
        info.update({'VVals': self.v})

        # Perform PPO update!
        info_ = self.update()
        info.update(info_)
        info.update({'Time': time.time() - start_time})
        # Log info about epoch

        return info

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        clip_ratio = self.clip_ratio

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self):
        info = {}

        data = self.buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            # kl = mpi_avg(pi_info['kl'])
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)  # average grads across MPI processes
            self.pi_optimizer.step()

        info.update(dict(StopIter=i))

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        info.update(dict(LossPi=pi_l_old, LossV=v_l_old,
                         KL=kl, Entropy=ent, ClipFrac=cf,
                         DeltaLossPi=(loss_pi.item() - pi_l_old),
                         DeltaLossV=(loss_v.item() - v_l_old)))
        return info


class LossFuncSearch(object):
    def __init__(self, config):
        """
        model: net
        total_epoch: to broadcast
        sample_type: choose detail of sample
        dist_type: gaussian or cauchy
        lr:
        scale:
        model_group_size:
        """
        self.model = None
        self.model_opt = None

        # self.reward_func = _build_func(config.reward_func)

        self.loss_func = _build_func(config.loss_func)
        self.sample_step = config.sample_step

        # !!! same freq !!!
        assert config.update_freq == self.sample_step
        config.update_freq = self.sample_step
        self.update_freq = config.update_freq

        self.start_sample_iter = config.start_sample_iter

        self.global_world_size = link.get_world_size()
        self.global_rank = link.get_rank()

        self.group = config.group
        self.num_groups = config.num_groups
        self.lambda_dis = config.get('lambda_dis', 1.)
        self.loss_kwargs = config.get('loss_kwargs', {})
        self.agent_kwargs = config.get('agent_kwargs', {})

        self.config = config
        self.best_acc = 0

        self.agent = _build_cls(self.sample_type)(
            **self.agent_kwargs,
            group=self.group, num_groups=self.num_groups, )
        self.a = self.config.agent_kwargs.loc

    def set_model(self, model):
        self.model = model

    def set_model_opt(self, model_opt):
        self.model_opt = model_opt

    def get_loss(self, outputs, targets):
        loss = self.loss_func(outputs, targets, a=self.a, lambda_dis=self.lambda_dis, **self.loss_kwargs)
        return loss

    def set_loss_parameters(self, env):
        self.a = self.agent.sample_subfunction(env)

    # todo: to utils
    def _broadcast_parameters(self, rank):
        """
        broadcast model parameters
        """
        for name, p in sorted(self.model.state_dict().items(), key=lambda x: x[0]):
            link.broadcast(p, rank)

        for k, v in self.model_opt.state.items():
            for name in ['exp_avg', 'exp_avg_sq', 'momentum_buffer']:
                if name in v:
                    if link.get_rank() == 0:
                        print(f'boardcast {name}', flush=True)

                    link.broadcast(v[name], rank)

    def update_lfs(self, iter, env, reward):
        rank = self.global_rank
        world_size = self.global_world_size

        test_acc_tensor = torch.zeros(world_size)

        # reward = self.reward_func(env)
        temp_acc = reward
        test_acc_tensor[rank] = temp_acc
        link.allreduce(test_acc_tensor)
        best_test_acc_rank = torch.argmax(test_acc_tensor)
        current_best_acc = test_acc_tensor[best_test_acc_rank].item()
        is_best = False
        if current_best_acc > self.best_acc:
            self.best_acc = current_best_acc
            is_best = True
        if self.global_rank == 0:
            print('broadcast {}'.format(best_test_acc_rank), flush=True)
        self._broadcast_parameters(rank=best_test_acc_rank)

        if iter + 1 > self.update_freq:
            info = self.agent.step(env, reward)


"""
        Proximal Policy Optimization (by clipping),

        with early stopping based on approximate KL

        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with a
                ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
                module. The ``step`` method should accept a batch of observations
                and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``v``        (batch,)          | Numpy array of value estimates
                                               | for the provided observations.
                ``logp_a``   (batch,)          | Numpy array of log probs for the
                                               | actions in ``a``.
                ===========  ================  ======================================

                The ``act`` method behaves the same as ``step`` but only returns ``a``.

                The ``pi`` module's forward call should accept a batch of
                observations and optionally a batch of actions, and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``pi``       N/A               | Torch Distribution object, containing
                                               | a batch of distributions describing
                                               | the policy for the provided observations.
                ``logp_a``   (batch,)          | Optional (only returned if batch of
                                               | actions is given). Tensor containing
                                               | the log probability, according to
                                               | the policy, of the provided actions.
                                               | If actions not given, will contain
                                               | ``None``.
                ===========  ================  ======================================

                The ``v`` module's forward call should accept a batch of observations
                and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``v``        (batch,)          | Tensor containing the value estimates
                                               | for the provided observations. (Critical:
                                               | make sure to flatten this!)
                ===========  ================  ======================================


            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to PPO.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.

            gamma (float): Discount factor. (Always between 0 and 1.)

            clip_ratio (float): Hyperparameter for clipping in the policy objective.
                Roughly: how far can the new policy go from the old policy while
                still profiting (improving the objective function)? The new policy
                can still go farther than the clip_ratio says, but it doesn't help
                on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
                denoted by :math:`\epsilon`.

            pi_lr (float): Learning rate for policy optimizer.

            vf_lr (float): Learning rate for value function optimizer.

            train_pi_iters (int): Maximum number of gradient descent steps to take
                on policy loss per epoch. (Early stopping may cause optimizer
                to take fewer than this.)

            train_v_iters (int): Number of gradient descent steps to take on
                value function per epoch.

            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            target_kl (float): Roughly what KL divergence we think is appropriate
                between new and old policies after an update. This will get used
                for early stopping. (Usually small, 0.01 or 0.05.)

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """
# ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
#         steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
#         vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
#         target_kl=0.01, logger_kwargs=dict(), save_freq=10):
# parser.add_argument('--env', type=str, default='HalfCheetah-v2')
# parser.add_argument('--hid', type=int, default=64)
# parser.add_argument('--l', type=int, default=2)
#
# ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
#     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
#     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
#     logger_kwargs=logger_kwargs)
