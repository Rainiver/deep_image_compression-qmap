import os
import time
import warnings

import torch

try:
    import spring.linklink as link
except:
    try:
        # for deprecated spring1
        import spring.linklink as link
    except:
        warnings.warn('cannot import spring.linklink. disable distributing training')
        link = None

allow_dead_parameter = True


class DistModule(torch.nn.Module):
    def __init__(self, module, sync=False):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)

        if not sync:
            self._grad_accs = []
            self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)

    def _register_hooks(self):
        for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(self, name, p, i):
        def hook(*ignore):
            link.allreduce_async(name, p.grad.data)

        return hook


def reduce_gradients(model, sync=False, group=None):
    """ sum up gradients """
    if sync:
        for name, param in model.named_parameters():
            try:
                if param.requires_grad:
                    if group is None:
                        link.allreduce(param.grad.data)
                    else:
                        link.allreduce(param.grad.data, group_idx=group)
            except Exception as e:
                if not allow_dead_parameter:
                    print("error caused when processing grad of '{}'".format(name))
                    raise e
    else:
        link.synchronize()


def broadcast_params(model):
    """ broadcast model parameters """
    for name, p in model.state_dict().items():
        link.broadcast(p, 0)


def dist_init():
    world_size = get_world_size()
    rank = get_rank()
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    if link:
        link.initialize()

    return rank, world_size


def get_rank():
    """Replace linklink.get_rank"""
    return int(os.environ.get('SLURM_PROCID', 0))


def get_world_size():
    """Replace linklink.get_world_size"""
    return int(os.environ.get('SLURM_NTASKS', 1))


def barrier():
    """Replace linklink.barrier"""
    if get_world_size() > 1:
        link.barrier()


def finalize():
    """Relpace linklink.finalize"""
    if link:
        link.finalize()
