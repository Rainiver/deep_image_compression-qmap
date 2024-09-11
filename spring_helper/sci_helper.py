r"""

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

from dataset.data_builder import build_dataloader
try:
    from integer2.scheduler import QuantEpochScheduler
except:
    print("load integer2 failed", flush=True)
    from integer.scheduler import QuantEpochScheduler
from kestrel.act_helper import extract_act
from losses.PSNR_Loss import Loss as PSNR
from losses.SSIM_Loss import msssim
from nets.model_builder import model_builder
from pipelines.models import get_codec, BaseCodec, CodecStageEnum
from tools.config import load_yaml
from tools.train_val_helper import load, convert_caffe, convert_nart
from utils import test_time
from tools.config import _convert_config_args
import math
from collections import defaultdict

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    from SpringCommonInterface import SpringCommonInterface as Interface
except ImportError as e:
    Interface = object

rank, world_size = None, None
try:
    import linklink as link
    from utils.distributed_utils import dist_init, reduce_gradients, DistModule
except:
    link = None
plt, sns = None, None

log = None


def to_log(content, flush=True, link_inited=True):
    if rank == 0:
        print(content, file=log, flush=True)
        print(content, flush=True)


def attr_controller(codec, args, stage, test_lambda4=None, test_b=None):
    # to_log("variable mode with lambda4s {}  and b_range {}".format(args.lambda4s, args.b_range))

    if stage == CodecStageEnum.TRAIN:
        if len(args.lambda4s) > 1:
            sampled_lambda = sample(args.lambda4s[1:], 1)[0]

            for name, model in codec.module.sub_models.items():
                if hasattr(model, 'sampled_lambda'):
                    # to_log('set {} {}'.format(name, sampled_lambda))
                    model.sampled_lambda = sampled_lambda

            args['lambda4'] = sampled_lambda

        if len(args.b_range) > 1:
            assert len(args.b_range) == 3
            sampled_b = random.uniform(args.b_range[1], args.b_range[2])
            sampled_delta = 2 ** sampled_b

            for name, model in codec.module.sub_models.items():
                if ('quant' in name or 'entropy' in name) and model is not None:
                    assert hasattr(model, 'delta')
                    model.delta = sampled_delta

    else:
        if stage == CodecStageEnum.VALID:
            if link is not None:
                codec_module = codec.module  # get wrapped module from DistModule
            else:
                codec_module = codec
        elif stage == CodecStageEnum.TEST:
            codec_module = codec

        if len(args.lambda4s) > 1:
            sampled_lambda = args.lambda4s[0] if test_lambda4 is None else test_lambda4

            for name, model in codec_module.sub_models.items():
                if hasattr(model, 'sampled_lambda'):
                    to_log('set {} {}'.format(name, sampled_lambda))
                    model.sampled_lambda = sampled_lambda

            args['lambda4'] = sampled_lambda

        if len(args.b_range) > 1:
            assert len(args.b_range) == 3
            sampled_b = args.b_range[0] if test_b is None else test_b
            sampled_delta = 2 ** sampled_b

            for name, model in codec_module.sub_models.items():
                if ('quant' in name or 'entropy' in name) and model is not None:
                    assert hasattr(model, 'delta')
                    model.delta = sampled_delta

            scale_factor = 1. / sampled_delta
            args['scale_factor_from_delta'] = scale_factor
    return codec, args


# args.lr_milestion change it's mean: update every iter
# args.show_bpp_interval change it's mean: update every show_interval

class SCIHelper(Interface):
    def __init__(self, config, metric_dict=None, work_dir=None, ckpt_dict=None):
        if Interface is not object:
            super(SCIHelper, self).__init__(config, metric_dict, work_dir, ckpt_dict)
        self.work_dir = work_dir
        config = copy.deepcopy(config)
        self.args = self._upgrade_config(config)

        self._setup_env()
        self._build()
        self._resume(ckpt_dict)
        self._temporaries = defaultdict(float)

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

        args.zero_pad = False
        args.out_dir = None
        args.kept = None
        args.log_dir_ckpt = args.log_dir
        log_dir = args.log_dir if args.out_dir is None else os.path.join(args.out_dir, 'logs')
        args.log_dir = log_dir
        return args

    def _setup_env(self):
        global link, log

        self.test_lambda4 = self.args.lambda4s[0]
        self.test_real_bpp = False
        self.test_quality_entropy = False
        self.is_test_mode = self.test_real_bpp or self.test_quality_entropy
        self.test_b = 0.
        self.test_time = False
        self.to_caffe = False
        self.to_nart = False
        self.flops_without_to_caffe = False
        self.output_act = False
        self.tensorboard_summary_level = 'full'
        self.quant_cfg_file = None
        self.quant_cfg = None

        self.rank, self.world_size = dist_init() if link else (0, 1)
        self.args.rank = rank

        log_dir = self.args.log_dir
        if rank == 0 and not os.path.exists(log_dir):
            os.mkdir(log_dir)
        while rank != 0 and not os.path.exists(log_dir):
            time.sleep(0.5)
        log_fname = 'log.txt'
        log = open(os.path.join(log_dir, log_fname), 'w')

    def _build(self):
        global link

        summary_dir = os.path.join(self.args.log_dir, 'events')
        if rank == 0:
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.writer = SummaryWriter(log_dir=summary_dir)

        to_log(self.args, link_inited=False)

        self.models = model_builder(self.args)
        self.codec = self.build_model()
        if self.rank == 0:
            for stage in CodecStageEnum:
                self.codec.print_processes(stage=stage)

        self.opt = torch.optim.Adam([{'params': self.codec.parameters()}], lr=self.args.lr)

        for name, model in self.models.items():
            if model:
                to_log(name + ':', link_inited=False)
                to_log(model, link_inited=False)

        self.sch = torch.optim.lr_scheduler.MultiStepLR(self.opt,
                                                        self.args.lr_milestion,
                                                        gamma=self.args.lr_scheduler_gamma)

        train_loader = build_dataloader(self.args.dataset, True)
        test_loader = build_dataloader(self.args.dataset, False)
        train_loader.get_epoch_size = lambda s: len(s.batch_sampler)
        test_loader.get_epoch_size = lambda s: len(s.batch_sampler)

        self.data_loaders = {
            'train': train_loader,
            'test': test_loader
        }

        if self.quant_cfg is not None:
            self.gsch = QuantEpochScheduler(self.codec, self.sch, self.quant_cfg)
            self.args.epoch = self.gsch.total_epoch
            to_log("total_epoch: {}".format(self.args.epoch), flush=True)
        else:
            self.gsch = self.sch

    def get_model(self):
        return self.codec

    def get_optimizer(self):
        return self.opt

    def get_scheduler(self):
        return self.gsch

    def get_dummy_input(self):
        return torch.randn(2, 3, 600, 600).cuda()

    @property
    def cur_iter(self):
        return self.lr_scheduler.last_iter

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
            'model': self.get_model().state_dict(model_cfg=True),
            'lr_scheduler': self.get_scheduler().state_dict()
        }
        return dump_dict

    def cur_eval_iter(self, step=None):
        if not hasattr(self, '_eval_iter'):
            self._eval_iter = 0
        if step is not None:
            self._eval_iter += step
        return self._eval_iter

    def local_eval_iter(self, step=None):
        return self.cur_eval_iter(step) % self.data_loaders['test'].get_epoch_size()

    def get_batch(self, batch_type='train'):
        if batch_type != 'train':
            _ = self.local_eval_iter(step=1)

        if not hasattr(self, 'data_iterators'):
            self.data_iterators = {}

        if batch_type not in self.data_iterators:
            iterator = self.data_iterators[batch_type] = iter(self.data_loaders[batch_type])
        else:
            iterator = self.data_iterators[batch_type]

        try:
            batch = next(iterator)
        except StopIteration as e:
            iterator = self.data_iterators[batch_type] = iter(self.data_loaders[batch_type])
            batch = next(iterator)

        return batch

    def get_total_iter(self):
        raise NotImplementedError

    @staticmethod
    def load_weights(model, ckpt_dict):
        return model.load(ckpt_dict.get('model', {}))

    def forward(self, batch):
        codec, args = attr_controller(self.get_model(), self.args, CodecStageEnum.TRAIN)
        outputs = codec(batch, stage=CodecStageEnum.TRAIN, args=args)
        self._temporaries['last_output'] = outputs
        loss = outputs['loss/total']
        return loss

    def backward(self, loss):
        self.get_optimizer().zero_grad()
        loss.backward()
        if self.args.linklink:
            reduce_gradients(self.get_model(), True)
        return loss

    def update(self):
        self.get_optimizer().step()
        self.get_scheduler().step()

    def train(self):
        # train given codec model

        rank = self.rank
        codec = self.get_model()
        train_loader = self.data_loaders['train']
        val_loader = self.data_loaders['test']
        opt = self.get_optimizer()
        sch = self.get_scheduler()
        args = self.args
        summary_level = self.tensorboard_summary_level

        if args.linklink and torch.cuda.is_available():
            codec = DistModule(codec, True)

        self.finished_iter = sch.last_epoch * len(train_loader)

        codec.train()
        for epoch in range(sch.last_epoch, args.epoch):
            sch.step()
            for iteration, data in enumerate(train_loader):

                # total_iter: [1, num_total_iter]
                self.finished_iter += 1
                if torch.cuda.is_available():
                    data = data.cuda()

                # wrapped before every iter
                codec, args = attr_controller(codec, args, CodecStageEnum.TRAIN)

                outputs = codec(data, stage=CodecStageEnum.TRAIN, args=args)

                loss = outputs['loss/total']
                loss /= world_size
                opt.zero_grad()
                loss.backward()
                if args.linklink:
                    reduce_gradients(codec, True)
                opt.step()

                # print log and write summary for each `show_interval` iterations
                if self.finished_iter % args.show_interval == 0 and rank == 0:
                    losses = []
                    for name, value in outputs.items():
                        if name.startswith('loss/'):
                            losses.append((name, value.item()))
                            self.writer.add_scalar(name, value.item(), self.finished_iter)
                        elif '/' in name:
                            # eval, args, images and other values
                            continue
                        else:
                            # record data distribution for debugging or exploration
                            if summary_level != 'full':
                                continue  # skip this when level is 'slim'
                            self.writer.add_scalar(name + '/max', value.max().item(), self.finished_iter)
                            self.writer.add_scalar(name + '/min', value.min().item(), self.finished_iter)
                            self.writer.add_histogram(name, value.cpu().detach().numpy(), self.finished_iter)
                    self.writer.add_scalar('lr/lr', sch.get_lr()[0], self.finished_iter)

                    losses.sort(key=lambda x: x[0])
                    loss_fmt = ', '.join(['{}:{:.7f}'.format(k, v) for k, v in losses])

                    # TODO log bpp
                    to_log(
                        ('[epoch:{}, batch:{}]\t[' + loss_fmt + ']\t[lr:{:.7f}]').format(epoch, iteration,
                                                                                         sch.get_lr()[0]))

                # validate for each `show_bpp_interval` iterations
                # if `show_bpp_interval` is less or equal to zero,
                # will never do that
                if args.show_bpp_interval > 0 and rank == 0:
                    if self.finished_iter % args.show_bpp_interval == 0:
                        self.evaluate()

                        # test will set codec to eval model
                        codec.train()

                if args.linklink and torch.cuda.is_available():
                    link.synchronize()

            # save checkpoint for each `args.snapshot_interval`
            # always save at the end of the last epoch
            if (epoch == args.epoch - 1 or epoch % args.snapshot_interval == 0) and rank == 0:
                torch.save(self.get_dump_dict(), os.path.join(args.log_dir, 'codec_epoch-{}.pth'.format(epoch)))

        # after the last epoch, test the finally saved model
        if rank == 0:
            self.evaluate()

    def evaluate(self):
        as_val = False
        val_iter = self.finished_iter if hasattr(self, 'finished_iter') else -1
        test_lambda4 = None
        test_b = None

        stage = CodecStageEnum.VALID if as_val else CodecStageEnum.TEST
        rank = self.rank
        codec = self.get_model()
        loader = self.data_loaders['test']
        args = self.args
        real_bpp = self.test_real_bpp
        quality_entropy = self.test_quality_entropy
        summary_level = self.tensorboard_summary_level
        assert not (as_val and val_iter < 0)

        log_dir = args.log_dir
        result_dir = os.path.join(log_dir, 'results')  # TODO: argument instead

        draw_grad = False
        testwriter = None
        if test_lambda4 is not None:
            result_dir = os.path.join(log_dir, 'results_{}_{}'.format(test_lambda4, test_b))

            draw_grad = args.get('draw_grad', False)
            event_dir = result_dir + '/event'
            if not os.path.exists(event_dir):
                os.makedirs(event_dir)
            testwriter = SummaryWriter(log_dir=event_dir)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with torch.set_grad_enabled(draw_grad):
            codec.eval()

            # wrapped before testing
            codec, args = attr_controller(codec, args, stage, test_lambda4=test_lambda4, test_b=test_b)

            eval_sums = {}

            test_time.clear()

            for i, x in enumerate(tqdm(loader)):

                test_time.mark(name="", new_pipeline=True)

                if torch.cuda.is_available():
                    x = x.cuda()

                if draw_grad:
                    x = x.clone().detach().requires_grad_(True)

                args['test_real_bpp'] = real_bpp
                args['y_save_path'] = os.path.join(result_dir, '%d.txt' % i)
                args['z_save_path'] = os.path.join(result_dir, '%d_side.txt' % i)
                if as_val or args.get('blocks', None) != True:
                    outputs = codec(x, stage=stage, args=args)
                else:
                    output = []
                    blocks = 0
                    l, r = 0, 64
                    lr = []
                    base = 8
                    while True:
                        blocks += 1
                        if r > x.shape[2]:
                            r = x.shape[2]
                        lr.append([l, r])
                        xx = x[:, :, l:r, :]
                        output.append({k: v for k, v in codec(xx, stage=stage, args=args).items()})
                        if r == x.shape[2]:
                            break
                        l += 64 - base
                        r = l + 64
                        if x.shape[2] - (r + 1) < 64:
                            r = x.shape[2]

                    x_hat = torch.zeros(x.shape)
                    if torch.cuda.is_available():
                        x_hat = x_hat.cuda()
                    for j in range(blocks):
                        l, r = lr[j]
                        if j > 0:
                            for k in range(l, l + base):
                                output[j]['image/reconstruction'][:, :, k - l, :] *= (k - l) / base
                        if j < blocks - 1:
                            for k in range(r - base, r):
                                output[j]['image/reconstruction'][:, :, k - l, :] *= (r - k) / base
                        x_hat[:, :, l:r, :] += output[j]['image/reconstruction']

                    outputs = output[0]
                    for name, value in output[0].items():
                        if name.startswith('eval/'):
                            temp = 0.
                            for j in range(0, blocks):
                                temp += output[j][name]
                            temp /= blocks
                            outputs[name] = temp
                    outputs['image/original'] = x
                    outputs['image/reconstruction'] = x_hat
                    outputs['eval/msssim'] = msssim(x, x_hat)
                    psnr = PSNR()
                    outputs['eval/psnr'] = psnr(x, x_hat)

                if draw_grad:
                    for loss_type in ['total', 'msssim', 'mse', 'rmse', 'yuv', 'entropy_total', 'tv']:
                        codec.zero_grad()
                        x.grad = None
                        outputs['loss/' + loss_type].backward(retain_graph=True)
                        grad = x.grad
                        # print('%d/' % i + loss_type + '_grad/min', grad.min())
                        # print('%d/' % i + loss_type + '_grad/max', grad.max())
                        grad_1 = grad - grad.min()
                        grad_1 = grad_1 / grad_1.max()
                        # print('%d/' % i + loss_type + '_grad_norm/min', grad.min())
                        # print('%d/' % i + loss_type + '_grad_norm/max', grad.max())
                        grad_2 = grad - grad.mean(dim=[2, 3], keepdim=True)
                        grad_2 = grad_2 / grad_2.std(dim=[2, 3], keepdim=True)
                        outputs['image/' + loss_type + '_norm'] = grad_1
                        outputs['image/' + loss_type + '_gauss'] = grad_2

                eval_fmt = []
                for name, value in outputs.items():
                    if name.startswith('eval/'):
                        if name not in eval_sums:
                            eval_sums[name] = 0.
                        eval_sums[name] += value
                        eval_fmt.append('{}:{:.7f}'.format(name, eval_sums[name] / (i + 1)))
                    elif name.startswith('image/') and rank == 0:
                        # convert NCHW tensor to HWC numpy array
                        img = torch.round(value * 255).int().clamp(0, 255)
                        img = img.detach().cpu().numpy()
                        img = np.array(img, dtype=np.uint8).squeeze(0)
                        img = np.transpose(img, [1, 2, 0])

                        if as_val and summary_level == 'full':
                            self.writer.add_image('%d/' % i + name[len('image/'):],
                                                  img_tensor=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                                  global_step=val_iter,
                                                  dataformats='HWC')
                        else:
                            # if summary_level is set to 'slim',
                            # do not save images to tsbd event but write em to disk directly
                            name = name[len('image/'):]
                            cv2.imwrite(result_dir + '/%d_%s.png' % (i, name), img)

                        if draw_grad:
                            testwriter.add_image('%d/' % i + name,
                                                 img_tensor=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                                 global_step=val_iter,
                                                 dataformats='HWC')

                eval_fmt = '[' + ', '.join(eval_fmt) + ']'

                # TODO: add options
                # TODO images
                to_log(eval_fmt, link_inited=as_val)

            test_time.output()

            eval_fmt = []
            for name, value in eval_sums.items():
                eval_sums[name] = value / len(loader)
                eval_fmt.append('{}:{:.7f}'.format(name, eval_sums[name]))
            eval_fmt = 'mean: [' + ', '.join(eval_fmt) + ']'

            to_log(eval_fmt, link_inited=as_val)

            if as_val and rank == 0:
                for name, value in eval_sums.items():
                    self.writer.add_scalar(name, value, val_iter)

        if testwriter:
            testwriter.close()

    @staticmethod
    def build_model_helper(config_dict=None):
        return get_codec(config_dict.codec, model_builder(config_dict)).cuda()

    def build_model(self):
        return get_codec(self.args.codec, self.models).cuda()

    def show_log(self):
        if not hasattr(self, 'show_log_cnt'):
            self.show_log_cnt = 0
        self.show_log_cnt += 1

        outputs = self._temporaries.pop('last_output')
        losses = []
        for name, value in outputs.items():
            if name.startswith('loss/'):
                losses.append((name, value.item()))
                self.writer.add_scalar(name, value.item(), self.finished_iter)
            elif '/' in name:
                # eval, args, images and other values
                continue
            else:
                # record data distribution for debugging or exploration
                if self.tensorboard_summary_level != 'full':
                    continue  # skip this when level is 'slim'
                self.writer.add_scalar(name + '/max', value.max().item(), self.finished_iter)
                self.writer.add_scalar(name + '/min', value.min().item(), self.finished_iter)
                self.writer.add_histogram(name, value.cpu().detach().numpy(), self.finished_iter)
        self.writer.add_scalar('lr/lr', self.get_scheduler().get_lr()[0], self.finished_iter)

        losses.sort(key=lambda x: x[0])
        loss_fmt = ', '.join(['{}:{:.7f}'.format(k, v) for k, v in losses])

        # TODO log bpp
        to_log(
            ('[epoch:{}, batch:{}]\t[' + loss_fmt + ']\t[lr:{:.7f}]').format(self.cur_epoch(),
                                                                             self.cur_iter,
                                                                             self.get_scheduler().get_lr()[0]))

        if self.args.show_bpp_interval > 0 and rank == 0:
            if self.show_log_cnt % self.args.show_bpp_interval == 0:
                self.evaluate()
                self.get_model().train()

    def add_external_model(cls, name, callable_object):
        raise NotImplementedError

    def convert_model(self, type="skme"):
        raise NotImplementedError

    def get_kestrel_parameter(self):
        raise NotImplementedError

"""
