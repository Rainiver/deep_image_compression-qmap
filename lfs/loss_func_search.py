from lfs.lfs_agent import *
import spring.linklink as link
import importlib
import torch


def _build_loss_func(import_path: str):
    module_spl = import_path.split('.')
    module, func_name = '.'.join(module_spl[:-1]), module_spl[-1]
    try:
        func = importlib.import_module(module).__dict__[func_name]
    except Exception:
        print(module)
        print(func_name)
        importlib.import_module(module)
        raise ValueError('unexpected loss func: ' + import_path)

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


class LossFuncSearch(object):
    def __init__(self, config, args=None):
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
        self.sample_type = config.get('sample_type', '')
        self.dist_type = config.get('dist_type', '')
        self.loss_func = _build_loss_func(config.loss_func)
        self.lr = config.get('lfs_lr', 0.)
        self.sample_step = config.sample_step
        self.update_freq = config.update_freq
        self.scale = config.get('lfs_scale', 0.)
        self.loc = config.get('lfs_loc', 0.)
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

        self.__init()

    def __init(self):
        sample_type, dist_type, lr, scale, loc = self.sample_type, self.dist_type, self.lr, self.scale, self.loc
        if sample_type == 'default':
            self.agent = LFSDefaultAgent(lr, scale, loc, dist_type,
                                         group=self.group, num_groups=self.num_groups, )
            self.a = loc
        elif sample_type == 'pos':
            self.agent = LFSDefaultAgent_pos(lr, scale, loc, dist_type,
                                             group=self.group, num_groups=self.num_groups, )
            self.a = loc
        elif sample_type == 'pos_1':
            self.agent = LFSDefaultAgent_pos1(lr, scale, loc, dist_type,
                                              group=self.group, num_groups=self.num_groups, )
            self.a = loc
        elif sample_type == 'pos_adapt':
            self.agent = pos_mu_adapt_s(lr, scale, loc, dist_type,
                                        group=self.group, num_groups=self.num_groups, )
            self.a = loc
        else:
            self.agent = _build_cls(sample_type)(
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

    def set_loss_parameters(self, iter):
        # todo
        if iter >= self.start_sample_iter and iter % self.sample_step:
            self.a = self.agent.sample_subfunction()

    # todo: to utils
    def _broadcast_parameters(self, rank):
        """
        broadcast model parameters
        """
        for name, p in sorted(self.model.state_dict().items(), key=lambda x: x[0]):
            link.broadcast(p, rank)

        for k, v in self.model_opt.state.items():
            for name in ['exp_avg', 'exp_avg_sq']:
                link.broadcast(v[name], rank)

    def update_lfs(self, iter, reward):
        rank = self.global_rank
        world_size = self.global_world_size

        #  print(rank, self.agent.last_a, "last a must not same", flush=True)
        #  print(rank, reward, "reward must not same", flush=True)

        test_acc_tensor = torch.zeros(world_size)

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

        # for k, v in self.model_opt.state.items():
        # kk = k
        # vv = v['exp_avg']
        # while kk.dim() != 1:
        #     kk = kk[0]
        # while vv.dim() != 1:
        #     vv = vv[0][0]
        # print(rank, k.sum(), "param must same", flush=True)
        # print(rank, v['exp_avg'].sum(), "opt state must same", flush=True)
        # break

        if iter + 1 > self.update_freq:
            reward = (test_acc_tensor - torch.mean(test_acc_tensor)) / \
                     ((torch.max(test_acc_tensor) - torch.min(test_acc_tensor)) + 1e-6) * 2
            self.agent.step(reward=reward[rank].item())

        # self.a = self.agent.sample_subfunction()
