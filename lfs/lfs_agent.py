import numpy as np
import math
import random

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import spring.linklink as link
import time
import importlib


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


class LFSBaseAgent(object):
    r"""
    this class sample loss's params, update distribution's params using REINFORCE
    """

    def __init__(self, lr=1e-4, scale=0.1, loc=None, dist_type='gaussian', group=None, num_groups=None, **kwargs):
        self.log_prob = []
        self.actions = []
        self.last_a = None

        self.group = group
        self.num_groups = num_groups

        self.param_number = len(loc)
        self.param_loc = torch.nn.Parameter(torch.Tensor(loc))
        if isinstance(scale, float):
            self.scale = torch.Tensor([scale, ] * self.param_number)
        elif isinstance(scale, list):
            self.scale = torch.Tensor(scale)
        else:
            raise NotImplementedError

        self.optimizer = _build_cls(kwargs.get('opt', 'torch.optim.Adam'))(
            [self.param_loc],
            **kwargs.get('opt_kwargs', dict(lr=lr, betas=(0.5, 0.999), weight_decay=0.0))
        )
        if link.get_rank() == 0:
            print(self.optimizer, flush=True)

        self.dist_type = dist_type
        if self.dist_type == 'gaussian':
            self.dist = torch.distributions.normal.Normal
        elif self.dist_type == 'cauchy':
            self.dist = torch.distributions.cauchy.Cauchy
        else:
            raise NotImplementedError

    def sample_subfunction(self):
        raise NotImplementedError

    def step(self, reward=0.0):
        self.optimizer.zero_grad()
        self.add_multi_log_prob(self.actions)
        loss = -torch.sum(torch.stack(self.log_prob, dim=-1)) * reward / (
            link.get_world_size() / self.num_groups
        )
        loss.backward()

        # reduce gradients
        for param in [self.param_loc, ]:
            if param.requires_grad:
                link.allreduce(param.grad.data)

        self.optimizer.step()
        if link.get_rank() == 0:
            print('\n\nafter update\n', flush=True)
            print(self.param_loc, flush=True)

        # broadcast gradients
        for param in [self.param_loc, ]:
            link.broadcast(param, 0)

        # reset
        del self.actions[:]
        del self.log_prob[:]

    def add_multi_log_prob(self, actions):
        # IMPORTANT: the main function for the agent in the main process compute log probablility of act1 ans act2
        self.loc_param_cuda = self.param_loc.cuda()
        self.scale_cuda = self.scale.cuda().detach()

        actions_ = torch.stack(actions, dim=1).to('cuda')

        m = []
        for i in range(self.param_number):
            m.append(self.dist(self.loc_param_cuda[i], self.scale_cuda[i]))

        for i in range(self.param_number):
            self.log_prob.append(torch.sum(m[i].log_prob(actions_[i])))

    def scale_step(self, epoch, tot_epoch=1000, start_scale=0.1, final_scale=0.01):
        temp_scale = start_scale + \
                     (final_scale - start_scale) * (epoch / tot_epoch)
        self.scale = torch.Tensor([temp_scale, ] * self.param_number)


class LFSDefaultAgent(LFSBaseAgent):
    def sample_subfunction(self):
        a = []
        m = []

        for i in range(self.param_number):
            m.append(self.dist(self.param_loc[i], self.scale[i]))

        for i in range(self.param_number):
            # sample = torch.tensor(m[i].sample())
            sample = m[i].sample()

            # rank = link.get_rank()
            # world_size = link.get_world_size()
            #
            # group_size = world_size // self.num_groups
            # assert group_size == 8 and world_size == 64, "default config"
            #
            # root_rank = rank // group_size * group_size
            # print(rank, ' ', root_rank, ' ', self.group)

            link.broadcast(sample, 0, group_idx=self.group)

            # rank = link.get_rank()
            # print(rank, ' ', sample, flush=True)
            a.append(sample.item())

        self.actions.append(torch.tensor(a))
        self.last_a = a
        return a


class LFSExpReparaAgent(LFSBaseAgent):
    def sample_subfunction(self):
        a = []
        m = []

        for i in range(self.param_number):
            m.append(self.dist(self.param_loc[i], self.scale[i]))

        for i in range(self.param_number):
            sample = m[i].sample()
            link.broadcast(sample, 0, group_idx=self.group)
            a.append(sample.item())

        self.actions.append(torch.tensor(a))
        self.last_a = a
        return a


class LFSDefaultAgent_pos(LFSBaseAgent):
    def sample_subfunction(self):
        a = []
        m = []

        for i in range(self.param_number):
            m.append(self.dist(self.param_loc[i], self.scale[i]))

        for i in range(self.param_number):
            # sample = torch.tensor(m[i].sample())
            sample = m[i].sample()

            # rank = link.get_rank()
            # world_size = link.get_world_size()
            #
            # group_size = world_size // self.num_groups
            # assert group_size == 8 and world_size == 64, "default config"
            #
            # root_rank = rank // group_size * group_size
            # print(rank, ' ', root_rank, ' ', self.group)

            link.broadcast(sample, 0, group_idx=self.group)

            # rank = link.get_rank()
            # print(rank, ' ', sample, flush=True)
            x = sample.item()
            a.append(x if x > 0. else 0.)

        self.actions.append(torch.tensor(a))
        self.last_a = a
        return a


class pos_mu_adapt_s(LFSBaseAgent):
    BIAS = 0.1

    def sample_subfunction(self):
        a = []
        m = []

        for i in range(self.param_number):
            m.append(self.dist(self.param_loc[i],
                               self.param_loc[i] * self.scale[i] + self.BIAS))

        for i in range(self.param_number):
            sample = m[i].sample()
            link.broadcast(sample, 0, group_idx=self.group)

            x = sample.item()
            a.append(x if x > 0. else 0.)

        self.actions.append(torch.tensor(a))
        self.last_a = a
        return a

    def add_multi_log_prob(self, actions):
        # IMPORTANT: the main function for the agent in the main process compute log probablility of act1 ans act2
        self.loc_param_cuda = self.param_loc.cuda()
        self.scale_cuda = self.scale.cuda().detach()

        actions_ = torch.stack(actions, dim=1).to('cuda')

        m = []
        for i in range(self.param_number):
            m.append(self.dist(self.loc_param_cuda[i],
                               self.loc_param_cuda[i].detach() * self.scale_cuda[i] + self.BIAS))

        for i in range(self.param_number):
            self.log_prob.append(torch.sum(m[i].log_prob(actions_[i])))


class pos_mu_adapt_s_auto_lr(LFSBaseAgent):
    BIAS = 0.1

    def __init__(self, *args, **kwargs):
        #  lr=1e-4, scale=0.1, loc=None, dist_type='gaussian', group=None, num_groups=None)
        super(pos_mu_adapt_s_auto_lr, self).__init__(*args, **kwargs)
        self.param_number = len(kwargs['loc']) - 1

    def sample_subfunction(self):
        a = []
        m = []

        k = self.param_loc[-1]
        for i in range(self.param_number):
            m.append(self.dist(self.param_loc[i] * k,
                               self.param_loc[i] * k * self.scale[i] + self.BIAS))

        for i in range(self.param_number):
            sample = m[i].sample()
            link.broadcast(sample, 0, group_idx=self.group)

            x = sample.item()
            a.append(x if x > 0. else 0.)

        self.actions.append(torch.tensor(a))
        self.last_a = a
        return a

    def add_multi_log_prob(self, actions):
        self.loc_param_cuda = self.param_loc.cuda()
        self.scale_cuda = self.scale.cuda().detach()
        k_cuda = self.loc_param_cuda[-1]

        actions_ = torch.stack(actions, dim=1).to('cuda')

        m = []
        for i in range(self.param_number):
            m.append(self.dist(self.loc_param_cuda[i] * k_cuda,
                               self.loc_param_cuda[i].detach() * k_cuda * self.scale_cuda[i] + self.BIAS))

        for i in range(self.param_number):
            self.log_prob.append(torch.sum(m[i].log_prob(actions_[i])))


class LFSDefaultAgent_pos1(LFSBaseAgent):
    def sample_subfunction(self):
        a = []
        m = []

        for i in range(self.param_number):
            m.append(self.dist(self.param_loc[i], self.scale[i]))

        for i in range(self.param_number):
            sample = m[i].sample()
            # print("sample started!", flush=True)
            link.broadcast(sample, 0, group_idx=self.group)
            # print("sample succeed!", flush=True)
            x = sample.item()
            x = x if x > 0. else 0.
            x = x if x < 1. else 1.
            a.append(x)

        self.actions.append(torch.tensor(a))
        self.last_a = a
        return a
