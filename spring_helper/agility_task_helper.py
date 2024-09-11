import abc
import copy
import os
import random
import time
from random import sample
import cv2
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn
import importlib

from dataset.data_builder import build_dataloader
try:
    from integer2.scheduler import QuantEpochScheduler
except:
    print("load integer2 failed", flush=True)
    from integer.scheduler import QuantEpochScheduler
from nets.model_builder import model_builder
from dataset.prefetcher import DataPrefetcher
from pipelines.models import BaseCodec, CodecStageEnum
from pipelines.models_helper import get_codec
from utils import test_time
from tools.config import _convert_config_args
import math
from collections import defaultdict
from lfs.reward import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    from SpringCommonInterface import SpringCommonInterface as Interface

    Interface = object
except ImportError as e:
    Interface = object

try:
    import spring.linklink as link
    from utils.distributed_utils import dist_init, reduce_gradients, DistModule
except ImportError as e:
    link = None

__all__ = ['AgilityTaskHelper']


# args.lr_milestion change it's mean: update every iter
# args.show_bpp_interval change it's mean: update every show_interval

class AgilityTaskHelper(Interface):
    def __init__(self, config, metric_dict=None, work_dir=None, ckpt_dict=None):
        if Interface is not object:
            super(AgilityTaskHelper, self).__init__(config, metric_dict, work_dir, ckpt_dict)
        self.work_dir = work_dir
        config_copy = copy.deepcopy(config)  # mdzz wsl
        self.args = self._upgrade_config(config_copy)
        if "lfs" in self.args:
            del self.args["lfs"]

        if "lfs" in config:
            self.args.lfs = config.lfs
        # print(id(config.lfs), "init", link.get_rank())

        self._setup_env()
        self._build()
        self._resume(ckpt_dict)
        self._temporaries = defaultdict(float)

    def to_log(self, content, flush=True, link_inited=True):
        if self.rank == 0:
            print(content, file=self.log, flush=True)
            print(content, flush=True)

    def _resume(self, ckpt_dict):
        def _remove_dist_module_prefix(k: str):
            return k[7:] if k.startswith('module.') else k

        if ckpt_dict is not None:
            self.get_model().load_state_dict(
                {_remove_dist_module_prefix(k): v for k, v in ckpt_dict['model'].items()}
            )
            self.get_optimizer().load_state_dict(ckpt_dict['optimizer'])
            self.get_scheduler().load_state_dict(ckpt_dict['lr_scheduler'])

    def _upgrade_config(self, config):
        # work_dir lose it's origin mean
        args = _convert_config_args(self.work_dir, config)

        args.log_dir_ckpt = args.log_dir
        args.log_dir = args.log_dir
        return args

    def _setup_env(self):
        global link
        self.output_act = False
        self.tensorboard_summary_level = 'full'
        self.quant_cfg_file = None
        self.quant_cfg = None

        self.rank, self.world_size = link.get_rank(), link.get_world_size() if link else (0, 1)
        self.args.rank = self.rank

        log_dir = self.args.log_dir
        if self.rank == 0 and not os.path.exists(log_dir):
            os.mkdir(log_dir)
        while self.rank != 0 and not os.path.exists(log_dir):
            time.sleep(0.5)
        log_fname = 'log.txt'
        self.log = open(os.path.join(log_dir, log_fname), 'w')

        device = torch.device('cuda:' + str(torch.cuda.current_device()))
        print('on device:', device)
        self.args.device = device

    def _build(self):
        global link

        summary_dir = os.path.join(self.args.log_dir, 'events')
        if self.rank == 0:
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.writer = SummaryWriter(log_dir=summary_dir)

        self.to_log(self.args, link_inited=False)

        self.models = model_builder(self.args)
        self.codec = self.build_model()

        for name, p in self.codec.state_dict().items():
            link.broadcast(p, 0)

        if self.rank == 0:
            for stage in CodecStageEnum:
                self.codec.print_processes(stage=stage)

        self.opt = torch.optim.Adam([{'params': self.codec.parameters()}], lr=self.args.lr)

        for name, model in self.models.items():
            if model:
                self.to_log(name + ':', link_inited=False)
                self.to_log(model, link_inited=False)

        self.sch = torch.optim.lr_scheduler.MultiStepLR(self.opt,
                                                        self.args.lr_milestion,
                                                        gamma=self.args.lr_scheduler_gamma)

        lfs_config = self.args.get('lfs_config', None)
        if lfs_config is not None:
            # if link.get_rank() == 0:
            # print(f"lfs_config in task helper's init, "
            #       f"num_groups: {lfs_config.num_groups}")
            self.args.dataset.num_groups = lfs_config.num_groups
            self.num_groups = lfs_config.num_groups

            def _build_reward(import_path: str):
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

            self.reward_func = lfs_config.get('reward_func', None)
            self.reward_kwargs = lfs_config.get('reward_kwargs', {})
            self.reward_func = reward_base_bpp if self.reward_func is None else _build_reward(self.reward_func)

        train_loader = build_dataloader(self.args.dataset, 'train')
        test_loader = build_dataloader(self.args.dataset, 'test')
        fast_test_loader = build_dataloader(self.args.dataset, 'fast_test')
        train_loader.get_epoch_size = lambda: len(train_loader.batch_sampler)
        test_loader.get_epoch_size = lambda: len(test_loader.batch_sampler)
        fast_test_loader.get_epoch_size = lambda: len(fast_test_loader.batch_sampler)

        self.data_loaders = {
            'train': train_loader,
            'test': test_loader,
            'fast_test': fast_test_loader,
        }

        train_data_size = self.data_loaders['train'].get_epoch_size()
        max_epoch = self.args.epoch
        self.max_iter = int(max_epoch * train_data_size)

        self.gsch = self.sch

    def get_model(self):
        return self.codec

    def get_optimizer(self):
        return self.opt

    def get_scheduler(self):
        return self.gsch

    def get_dummy_input(self):
        if torch.cuda.is_available():
            return torch.randn(2, 3, 600, 600).cuda()
        else:
            # TODO: only use in debug
            # return torch.randn(2, 3, 60, 60)
            return EnvironmentError('need cuda')

    @property
    def cur_iter(self):
        # TODO:epoch change to iter !!
        return self.get_scheduler().last_epoch

    def cur_epoch(self, mode='round'):
        epoch = self.cur_iter / self.data_loaders['train'].get_epoch_size()
        if mode == 'round':
            return round(epoch)
        elif mode == 'floor':
            return math.floor(epoch)
        elif mode == 'ceil':
            return math.ceil(epoch)
        else:
            raise ValueError

    def get_dump_dict(self):
        dump_dict = {
            'epoch': self.cur_epoch('round'),
            'iter': self.cur_iter,
            'optimizer': self.get_optimizer().state_dict(),
            'model': self.get_model().state_dict(),
            'lr_scheduler': self.get_scheduler().state_dict()
        }
        return dump_dict

    def load_ckpt(self, ckpt_dict):
        self.get_optimizer().load_state_dict(ckpt_dict.get('optimizer', {}))
        self.get_scheduler().load_state_dict(ckpt_dict.get('lr_scheduler', {}))
        self.get_model().load_state_dict(ckpt_dict.get('model', {}))

        print(f"rank{self.rank} load ckpt succeed!!! epoch{self.cur_epoch('round')} iter{self.cur_iter}", flush=True)
        # print(f"rank{self.rank} opt{self.get_optimizer().state_dict()} sch{self.get_scheduler().state_dict()}",
        #       flush=True)

    def cur_eval_iter(self, step=None):
        if not hasattr(self, '_eval_iter'):
            self._eval_iter = 0
        if step is not None:
            self._eval_iter += step
        return self._eval_iter

    def cur_eval_epoch(self, step=None, loader=None):
        if loader is None:
            loader = self.data_loaders['test']
        return (self.cur_eval_iter(step) - 1) // loader.get_epoch_size()

    def local_eval_iter(self, step=None, loader=None):
        if loader is None:
            loader = self.data_loaders['test']
        return self.cur_eval_iter(step) % loader.get_epoch_size()

    def get_batch(self, batch_type='train', step=True):
        # TODO: step control is more and more ugly now
        if batch_type != 'train' and step:
            _ = self.cur_eval_iter(step=1)

        if not hasattr(self, 'data_iterators'):
            self.data_iterators = {}

        if batch_type not in self.data_iterators:
            iterator = self.data_iterators[batch_type] = iter(
                DataPrefetcher(self.data_loaders[batch_type], self.args.device)
            )
        else:
            iterator = self.data_iterators[batch_type]

        try:
            batch = next(iterator)
        except StopIteration as e:
            iterator = self.data_iterators[batch_type] = iter(
                DataPrefetcher(self.data_loaders[batch_type], self.args.device)
            )
            batch = next(iterator)

        return batch

    def get_total_iter(self):
        return self.max_iter

    @staticmethod
    def load_weights(model, ckpt_dict):
        return model.load_state_dict(ckpt_dict.get('model', {}))

    def forward(self, batch):
        codec, args = self.get_model(), self.args
        outputs = codec(batch, stage=CodecStageEnum.TRAIN, args=args)
        self._temporaries['last_output'] = outputs
        loss = outputs['loss/total']
        return loss

    def backward(self, loss):
        self.get_optimizer().zero_grad()
        loss.backward()
        if self.args.linklink:
            if "lfs" in self.args and "group" in self.args:
                # print("use group reduce gradient")
                reduce_gradients(self.get_model(), True, group=self.args.group)
            else:
                reduce_gradients(self.get_model(), True)
        return loss

    def update(self):
        self.get_optimizer().step()
        self.get_scheduler().step()

        # TODO: i think it's deprecate
        # if self.get_total_iter() == self.cur_iter:
        #     self.log.close()
        #     if self.rank == 0:
        #         self.writer.close()

    def train(self):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, as_val=True, func_mode="fast", lfs=False, loc=[],
                 automl=False, ret_rew=False, **kwargs):
        r"""
        verbose: will print every image's test result in order and time of each stage and a summary
        fast: will use multi process to speed up and only print a summary

        NOTE: however we only use verbose in test and fast in nas
        """
        # we hope don't consider rank outside this function, that is every processes must jumped in
        # as_val will record everything into tensorboard, otherwise will save images

        # TODO: i don't find different in TEST and VALID stage
        # print(f"{self.rank}eval start", flush=True)

        assert func_mode in ["verbose", "fast"]  # affect log and multi processes

        ans = torch.Tensor([0.])
        rank = self.rank
        args = self.args
        summary_level = self.tensorboard_summary_level
        nas_state = 'nas' in self.__dict__ and self.nas
        nas_prefix = 'nas_' if nas_state else ''
        if as_val:
            if nas_state:
                stage = CodecStageEnum.SEARCH
            else:
                stage = CodecStageEnum.VALID
        else:
            stage = CodecStageEnum.TEST
        codec = self.get_model()
        log_dir = args.log_dir
        real_bpp = True

        compute = func_mode == "fast" or (func_mode == "verbose" and rank == 0)

        # that is the x axis in tensor board
        val_iter = self.nas_cnt if nas_state else self.cur_iter

        if compute:
            result_dir = os.path.join(log_dir, f'results{rank}')  # TODO: argument instead
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            codec.eval()
            eval_sums = {}
            test_time.clear()

            if func_mode == "verbose":
                load_mode = 'test'
            elif func_mode == "fast":
                load_mode = 'fast_test'
            else:
                raise NotImplementedError

            # the behavior may be difficult to understand
            loader = self.data_loaders[load_mode]
            for iter in range(loader.get_epoch_size()):
                test_time.mark(name="", new_pipeline=True)
                args['test_real_bpp'] = real_bpp
                args['y_save_path'] = os.path.join(result_dir, '%d.txt' % iter)
                args['z_save_path'] = os.path.join(result_dir, '%d_side.txt' % iter)

                x = self.get_batch(load_mode, False if lfs else True)
                outputs = codec(x, stage=stage, args=args)

                eval_fmt = []
                for name, value in sorted(outputs.items()):
                    if name.startswith('eval/'):
                        if name not in eval_sums:
                            eval_sums[name] = torch.Tensor([0.])
                        eval_sums[name] += value
                        eval_fmt.append('{}:{:.7f}'.format(name, eval_sums[name].item() / (iter + 1)))
                    elif name.startswith('image/') and not lfs:
                        # convert NCHW tensor to HWC numpy array
                        if isinstance(value, np.ndarray):
                            img = value
                        else:
                            img = torch.round(value * 255).int().clamp(0, 255)
                            img = img.detach().cpu().numpy()
                            img = np.array(img, dtype=np.uint8).squeeze(0)
                            img = np.transpose(img, [1, 2, 0])

                        if as_val and summary_level == 'full':
                            eval_epoch = self.nas_cnt if nas_state else self.cur_eval_epoch(loader=loader)
                            #  print(rank, iter, eval_epoch, name)
                            if eval_epoch % self.args.show_img_interval == 0:
                                if func_mode == "verbose":
                                    self.writer.add_image(f'{nas_prefix}{iter}/' + name[len('image/'):],
                                                          img_tensor=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                                          global_step=val_iter,
                                                          dataformats='HWC')
                                elif func_mode == "fast":
                                    # please choose the max size of the images in eval
                                    max_w, max_h = self.args.get('eval_max_w', 768), self.args.get('eval_max_h', 768)
                                    x = torch.from_numpy(img).cuda()
                                    x = x.permute([2, 0, 1]).unsqueeze(dim=0)
                                    # l, r, t, b
                                    x = nn.ZeroPad2d((0, max_w - img.shape[1], 0, max_h - img.shape[0]))(x)
                                    x = x.squeeze().permute([1, 2, 0]).int().contiguous()
                                    y = torch.zeros(link.get_world_size(), *x.shape).int().cuda()
                                    # TODO: add assert or warning eg: all image must have the same size
                                    link.gather(y, x, 0)
                                    y = y.cpu().numpy().astype(np.uint8)
                                    if self.rank == 0:
                                        for idx, img_y in enumerate(y):
                                            self.writer.add_image(f'{nas_prefix}{idx}_{iter}/' + name[len('image/'):],
                                                                  img_tensor=cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB),
                                                                  global_step=val_iter,
                                                                  dataformats='HWC')
                        else:
                            # if summary_level is set to 'slim',
                            # do not save images to tsbd event but write em to disk directly
                            name = name[len('image/'):]
                            cv2.imwrite(result_dir + '/%d_%s.png' % (iter, name), img)

                eval_fmt = '[' + ', '.join(eval_fmt) + ']'

                # TODO: add options
                # TODO images
                if func_mode == "verbose":
                    self.to_log(f"[rank {rank:04d}]-[img {iter:04d}]: " + eval_fmt, link_inited=as_val)

            eval_fmt = []

            if func_mode == "fast":
                if "group" in self.args:
                    # print("evaluate use group", flush=True)
                    for name, value in eval_sums.items():
                        link.allreduce(value, self.args.group)
                else:
                    for name, value in eval_sums.items():
                        link.allreduce(value)

            for name, value in eval_sums.items():
                eval_sums[name] = value / loader.dataset.__len__()
                eval_fmt.append('{}:{:.7f}'.format(name, eval_sums[name].item()))
            eval_fmt = 'mean: [' + ', '.join(eval_fmt) + ']'

            if func_mode == "verbose":
                test_time.output()
                self.to_log(
                    f"csv:[{eval_sums['eval/msssim'].item()},{eval_sums['eval/psnr'].item()},{torch.tensor(eval_sums['eval/bpp']).item()}]",
                    link_inited=as_val
                )

            self.to_log(eval_fmt, link_inited=as_val)

            if lfs or automl or ret_rew:
                # loss func search
                eval_msssim = eval_sums['eval/' + 'msssim']
                eval_psnr = eval_sums['eval/' + 'psnr']
                eval_bpp = torch.Tensor([eval_sums['eval/' + 'bpp']])  # torch.Tensor
                if 'req_more' in self.reward_kwargs and self.reward_kwargs['req_more']:
                    ans_more = self.reward_func(eval_bpp, eval_msssim, eval_psnr, eval_sums, **self.reward_kwargs)
                    ans = ans_more['ans']
                    # print("req_more succeed!!!", flush=True)
                else:
                    ans = self.reward_func(eval_bpp, eval_msssim, eval_psnr, eval_sums, **self.reward_kwargs)
            else:
                # normal
                for iter, (name, lam) in enumerate(self.args.get('metrics', {'msssim': 1.}).items()):
                    ans += lam * eval_sums['eval/' + name]

            if as_val:
                if lfs or automl:
                    if rank == 0:
                        self.writer.add_scalar('reward', ans.item(), val_iter)

                    if 'req_more' in self.reward_kwargs and self.reward_kwargs['req_more']:
                        for k, v in sorted(ans_more.items()):
                            # print(self.rank, k, v, "from ans_more", flush=True)
                            eval_sums['reward/' + k] = v
                    else:
                        eval_sums['reward'] = ans

                    for loc_i, loc_v in enumerate(loc):
                        if rank == 0:
                            self.writer.add_scalar(f'loss_param/{loc_i}',
                                                   loc_v.item(), val_iter)

                    # print(self.rank, id(self.args.lfs.agent.param_loc), self.args.lfs.agent.param_loc, flush=True)
                    # print(self.rank, id(loc), loc, flush=True)

                    # print(self.rank, id(self.args.lfs.a), self.args.lfs.a, 'args a', flush=True)
                    # print(self.rank, id(kwargs['LFS'].a), kwargs['LFS'].a, 'LFS a', flush=True)
                    # print(self.rank, id(self.args.lfs), 'args lfs', flush=True)
                    # print(self.rank, id(kwargs['LFS']), 'LFS', flush=True)

                    loc_x = torch.tensor(self.args.lfs.a).contiguous().cuda()
                    loc_tot = torch.zeros(link.get_world_size(), *loc_x.shape).contiguous().cuda()
                    link.gather(loc_tot, loc_x, 0)
                    if rank == 0:
                        for idx_t, loc_t in enumerate(loc_tot.cpu()):
                            if idx_t % (link.get_world_size() // self.num_groups) != 0:
                                continue
                            for loc_i, loc_v in enumerate(loc_t):
                                self.writer.add_scalar(
                                    f'group{idx_t // (link.get_world_size() // self.num_groups)}/loss_param_{loc_i}',
                                    loc_v.item(), val_iter)

                    # print(f"{self.rank}start perpar eval sum", flush=True)
                    for name, value in sorted(eval_sums.items()):
                        # print(self.rank, name, value, "in for...", flush=True)
                        x = value.clone().contiguous().cuda().float()
                        y = torch.zeros(link.get_world_size(), *x.shape).contiguous().cuda().float()
                        link.gather(y, x, 0)
                        if rank == 0:
                            for idx_t, y_e in enumerate(y.cpu()):
                                if idx_t % (link.get_world_size() // self.num_groups) != 0:
                                    continue
                                self.writer.add_scalar(
                                    f"group{idx_t // (link.get_world_size() // self.num_groups)}/" +
                                    "lfs_" + nas_prefix + name,
                                    y_e.item(), val_iter)
                    # print(f"{self.rank}finish", flush=True)
                else:
                    for name, value in eval_sums.items():
                        if rank == 0:
                            self.writer.add_scalar(
                                nas_prefix + name, value.item(), val_iter)

        # print(f"{self.rank}compute finish ", flush=True)
        if "group" in self.args:
            # link.broadcast(ans, 0)
            # group => lfs stage => must fast test
            # print("not broadcast ans", flush=True)
            pass
        else:
            link.broadcast(ans, 0)

        # print(f"{self.rank}eval finish", flush=True)
        if automl:
            return eval_sums
        return ans

    def get_env_info(ds, loc):
        if ds == 'fast_test':
            fm = 'fast'
        # elif ds == 'test':
        #     fm = 'verbose'
        else:
            raise NotImplementedError

        return self.evaluate(as_val=True, func_mode=fm,
                             automl=True, lfs=False, loc=loc)

    @staticmethod
    def build_model_helper(config_dict=None):
        nas = config_dict.get('nas', False)
        return get_codec(config_dict.codec, model_builder(config_dict), nas, codec_cfg=self.args).cuda()

    def build_model(self):
        nas = self.args.get('nas', False)
        if torch.cuda.is_available():
            return get_codec(self.args.codec, self.models, nas, codec_cfg=self.args).cuda()
        else:
            # TODO:only for debug
            # return get_codec(self.args.codec, self.models, nas)
            raise EnvironmentError('need cuda')

    def show_log(self):
        if not hasattr(self, 'show_log_cnt'):
            self.show_log_cnt = 0
        self.show_log_cnt += 1

        if self.rank == 0:
            outputs = self._temporaries.pop('last_output')
            losses = []
            for name, value in outputs.items():
                if name.startswith('loss/'):
                    losses.append((name, value.item()))
                    self.writer.add_scalar(name, value.item(), self.cur_iter)
                elif '/' in name:
                    # eval, args, images and other values
                    continue
                else:
                    # record data distribution for debugging or exploration
                    if self.tensorboard_summary_level != 'full':
                        continue  # skip this when level is 'slim'
                    self.writer.add_scalar(name + '/max', value.max().item(), self.cur_iter)
                    self.writer.add_scalar(name + '/min', value.min().item(), self.cur_iter)
                    self.writer.add_histogram(name, value.cpu().detach().numpy(), self.cur_iter)
            self.writer.add_scalar('lr/lr', self.get_scheduler().get_lr()[0], self.cur_iter)

            losses.sort(key=lambda x: x[0])
            loss_fmt = ', '.join(['{}:{:.7f}'.format(k, v) for k, v in losses])

            # TODO log bpp
            self.to_log(
                ('[epoch:{}, batch:{}]\t[' + loss_fmt + ']\t[lr:{:.7f}]').format(self.cur_epoch(),
                                                                                 self.cur_iter,
                                                                                 self.get_scheduler().get_lr()[0]))

        if self.args.show_bpp_interval > 0:
            if self.show_log_cnt % self.args.show_bpp_interval == 0:
                self.evaluate(
                    self.args.get('eval_in_train', {'as_val': True,
                                                    'func_mode': "fast"})
                )
                self.get_model().train()
        link.barrier()

    def add_external_model(cls, name, callable_object):
        raise NotImplementedError

    def convert_model(self, type="skme"):
        raise NotImplementedError

    def get_kestrel_parameter(self):
        raise NotImplementedError


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    import os
    import sys
    from utils.color_print import cprint

    link.initialize()
    torch.cuda.set_device(link.get_local_rank())

    # with open(os.path.join('dic', 'spring_helper', 'train_val_config_v2', 'gg18.yaml')) as f:
    with open(os.path.join('dic', 'spring_helper', 'nas_val_config', 'gg18.yaml')) as f:
        cfg = yaml.load(f)
        cfg = EasyDict(cfg)
        cfg = cfg.config

    helper = AgilityTaskHelper(cfg, work_dir='.')

    cprint('check eval 24 photo && evaluate success')
    helper.evaluate()

    cprint('check save image')
    helper.evaluate(as_val=False)

    cprint('check fast')
    helper.evaluate(func_mode="fast")

    cprint('check fast save image')
    helper.evaluate(as_val=False, func_mode="fast")

    # cprint('check 7999 train-set len')
    # helper.get_batch()
    # print(len(helper.data_iterators['train']))
    #
    # cprint('check 23 eval-set len')
    # helper.get_batch('test')
    # print(len(helper.data_iterators['test']))
    #
    # cprint('check eval 24 photo && evaluate success')
    # helper.evaluate()
    #
    # cprint('check save image')
    # helper.evaluate(as_val=False)
    #
    # cprint('train can be exec BUT not sure it\'s precision')
    # helper.get_model().train()
    # x = helper.get_batch()
    # y = helper.forward(x)
    # helper.backward(y)
    # helper.update()
    #
    # cprint('check log')
    # helper.show_log()
    #
    # cprint('this time should eval')
    # helper.get_model().train()
    # x = helper.get_batch()
    # y = helper.forward(x)
    # helper.backward(y)
    # helper.update()
    #
    # helper.args.show_bpp_interval = 1
    # helper.show_log()
    #
    # cprint('dump')
    # d = helper.get_dump_dict()
    # torch.save(d, 'hello')
    #
    # cprint('load')
    # e = torch.load('hello')
    # helper._resume(e)
    #
    # cprint('load weight')
    # helper.load_weights(helper.get_model(), e)
    #
    # cprint('check iter')
    # print(helper.cur_iter)
    # print(helper.cur_epoch())
    # print(helper.cur_eval_iter())
    # print(helper.local_eval_iter())
    #
    # cprint('check dataloader')
    # print(helper.data_loaders['train'].get_epoch_size())
    # print(helper.data_loaders['test'].get_epoch_size())

    link.finalize()
