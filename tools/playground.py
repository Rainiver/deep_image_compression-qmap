import os
import random
import time
from argparse import ArgumentParser
from random import sample

import cv2
import numpy as np
import torch
from torch import nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from codes.prob_table_utils import read_table
from dataset.data_builder import build_dataloader
from dataset.prefetcher import DataPrefetcher

try:
    from integer2.scheduler import QuantEpochScheduler
except:
    print("playground load integer2 failed", flush=True)
    from integer.scheduler import QuantEpochScheduler
from kestrel.act_helper import extract_act
from losses.PSNR_Loss import Loss as PSNR
from losses.SSIM_Loss import msssim
from nets.model_builder import model_builder
from pipelines.models import BaseCodec, CodecStageEnum
from pipelines.models_helper import get_codec
from tools.config import load_yaml
from tools.train_val_helper import load, convert_caffe, convert_nart
from utils import test_time
from utils import git_reader

# if use rank or world_size before init them
# will & should raise an error
from utils.distributed_utils import link, dist_init, reduce_gradients, DistModule

rank, world_size = None, None

plt, sns = None, None

TRAIN_MODE = 0
TEST_MODE = 1
VAL_MODE = -1
CAFFE_MODE = 2
FLOPS_MODE = 3
NART_MODE = 4


def to_log(content, flush=True, link_inited=True):
    if rank == 0:
        print(content, file=log, flush=True)
        print(content, flush=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args, **kwargs):
    global world_size, rank, log, writer, args_, link
    # test arguments
    test_zero_pad = 'test_zero_pad' in kwargs and kwargs['test_zero_pad']
    args.zero_pad = test_zero_pad
    test_real_bpp = 'test_real_bpp' in kwargs and kwargs['test_real_bpp']
    test_quality_entropy = 'test_quality_entropy' in kwargs and kwargs['test_quality_entropy']
    is_test_mode = test_real_bpp or test_quality_entropy
    test_lambda4 = args.lambda4s[kwargs['test_lambda4_id']] if 'test_lambda4_id' in kwargs else None
    test_b = kwargs['test_b'] if 'test_b' in kwargs else None
    args.test_time = kwargs['test_time'] if 'test_time' in kwargs else False
    to_caffe = 'to_caffe' in kwargs and kwargs['to_caffe']
    to_nart = 'to_nart' in kwargs and kwargs['to_nart']
    flops_without_to_caffe = kwargs['flops_without_to_caffe']
    output_act = kwargs['output_act']
    args.out_dir = kwargs['out_dir']
    args.nart_backend = kwargs['nart_backend']
    args.nart_out = kwargs['nart_out']
    args.kept = kwargs['kept']
    tensorboard_summary_level = kwargs['tensorboard_summary_level']
    args.fast_train = kwargs['fast_train']
    if kwargs['test_list'] is not None:
        args.dataset.test.meta_file_list = [kwargs['test_list']]
    quant_cfg_file = kwargs['quant_cfg']
    if quant_cfg_file is not None:
        with open(quant_cfg_file) as f:
            quant_cfg = yaml.load(f)
    else:
        quant_cfg = None

    # Set random seed
    set_random_seed(666)

    # set to deterministic by default
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # or conv_transpose op is non-deterministic

    # run mode
    mode = TEST_MODE if is_test_mode else TRAIN_MODE
    if to_caffe:
        mode = CAFFE_MODE
    if flops_without_to_caffe:
        mode = FLOPS_MODE
    if to_nart:
        mode = NART_MODE

    args.log_dir_ckpt = args.log_dir
    log_dir = args.log_dir if args.out_dir is None else os.path.join(args.out_dir, 'logs')
    args.log_dir = log_dir
    rank, world_size = dist_init() if link else (0, 1)
    args.rank = rank
    if rank == 0 and not os.path.exists(log_dir):
        os.mkdir(log_dir)
    while rank != 0 and not os.path.exists(log_dir):
        time.sleep(0.5)
    log_fname = 'test_log.txt' if is_test_mode else 'train_log.txt'
    log = open(os.path.join(log_dir, log_fname), 'w')

    summary_dir = os.path.join(args.log_dir, 'events')
    if rank == 0 and mode == TRAIN_MODE:
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=summary_dir)

    device = kwargs.get('device', None)
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('cuda is not available')
        device = torch.device('cuda:' + str(torch.cuda.current_device()))
    else:
        device = torch.device('cpu:0')
    print('on device:', device)
    args.device = device

    args_ = args
    to_log(git_reader.read_git(), link_inited=False)
    to_log(args, link_inited=False)

    models = model_builder(args)
    codec = get_codec(args.codec, models)  # codec是模型，调用返回的是data_pool
    if rank == 0:
        for stage in CodecStageEnum:
            codec.print_processes(stage=stage)

    opt_cls = torch.optim.Adam
    if args.fast_train and not args.get('disable_apex'):
        try:
            # using apex FusedAdam for a faster bp
            # see https://nvidia.github.io/apex/optimizers.html#apex.optimizers.FusedAdam
            from apex.optimizers import FusedAdam
            to_log('using apex FusedAdam')

            opt_cls = FusedAdam
        except ImportError:
            to_log('cannot load apex FusedAdam, using default Pytorch implementation')

    opt = opt_cls([{'params': codec.parameters()}], lr=args.lr)

    for name, model in models.items():
        if model:
            to_log(name + ':', link_inited=False)
            to_log(model, link_inited=False)

    sch = torch.optim.lr_scheduler.MultiStepLR(opt, args.lr_milestion, gamma=args.lr_scheduler_gamma)

    loader = build_dataloader(args.dataset, 'train')

    codec.to(device)

    load({'codec': codec}, None, None, args.log_dir_ckpt, args.load_epoch, 0, mode)

    if quant_cfg is not None:
        gsch = QuantEpochScheduler(codec, sch, quant_cfg)
        args.epoch = gsch.total_epoch
        to_log("total_epoch: {}".format(args.epoch), flush=True)
    else:
        gsch = sch

    val_loader = build_dataloader(args.dataset, 'test')
    if mode == TRAIN_MODE:
        train(codec, loader, gsch.optimizer, gsch, args, val_loader,
              summary_level=tensorboard_summary_level)
    elif mode == TEST_MODE:
        if quant_cfg is not None:
            gsch.update_quant_cfg(is_test=True)
        if output_act:
            extract_act(codec, args.log_dir)
        test(codec, val_loader, args, real_bpp=test_real_bpp, quality_entropy=test_quality_entropy,
             test_lambda4=test_lambda4, test_b=test_b)
    elif mode == CAFFE_MODE or mode == FLOPS_MODE:
        convert_caffe(codec.sub_models, mode)
    elif mode == NART_MODE:
        convert_nart(codec.sub_models, args)
    if link:
        link.finalize()
    log.close()
    if rank == 0 and mode == TRAIN_MODE:
        writer.close()


def train(codec, train_loader, opt, sch, args, val_loader, summary_level='full'):
    """
    train given codec model
    """

    global rank
    device = args.device

    # [20201130] hedailan:
    # grad clip for training stability
    # setting it to 0.1 may help when unexpected NaN occurs
    # see: https://github.com/InterDigitalInc/CompressAI/blob/master/examples/train.py
    clip_max_norm = args.get('clip_max_norm', 0.)

    load({}, opt if args.get('load_opt', True) else None, sch, args.log_dir_ckpt, args.load_epoch, 0, TRAIN_MODE)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if args.linklink and torch.cuda.is_available():
        codec = DistModule(codec, True)

    total_iter = sch.last_epoch * len(train_loader)

    codec.train()

    train_prefetcher = DataPrefetcher(train_loader, args.device)

    if args.get('double_lambda_trick'):
        to_log('training with a double lambda trick.')
        double_lambda_trick_epoch = args.get('double_lambda_trick_epoch', args.epoch // 2)
    else:
        double_lambda_trick_epoch = args.epoch

    for epoch in range(sch.last_epoch, args.epoch):
        sch.step()
        for iteration, data in enumerate(train_prefetcher):

            # total_iter: [1, num_total_iter]
            total_iter += 1

            # prefetcher has move data to the target device
            # data = data.to(device)

            # 2x lambda trick by minnen2020 paper
            lambda4_ori = args['lambda4']  # NOTICE: hard-code lambda4 may be removed in the future
            if args.get('double_lambda_trick'):
                if epoch < double_lambda_trick_epoch:
                    # our implementation half the lambda4(which of R) instead
                    args['lambda4'] = lambda4_ori / 2
                elif epoch == double_lambda_trick_epoch:
                    to_log(f'using target lambda4: {lambda4_ori} from now on')

            # wrapped before every iter
            codec, args = attr_controller(codec, args, total_iter, CodecStageEnum.TRAIN)

            # TODO: delete
            if isinstance(data, dict):
                data["cur_epoch"] = epoch
            outputs = codec(data, stage=CodecStageEnum.TRAIN, args=args)

            # recover lambda4
            if args.get('double_lambda_trick'):
                # our implementation half the lambda4(which of R) instead
                args['lambda4'] = lambda4_ori

            loss = outputs['loss/total']
            loss = loss / world_size
            opt.zero_grad()
            loss.backward()
            if args.linklink:
                reduce_gradients(codec, True)
            if clip_max_norm > 0:
                nn.utils.clip_grad_norm_(codec.parameters(), clip_max_norm)
            opt.step()

            # print log and write summary for each `show_interval` iterations
            if total_iter % args.show_interval == 0 and rank == 0:
                losses = []
                for name, value in outputs.items():
                    if name.startswith('loss/'):
                        losses.append((name, value.item()))
                        writer.add_scalar(name, value.item(), total_iter)
                    elif '/' in name:
                        # eval, args, images and other values
                        continue
                    else:
                        # record data distribution for debugging or exploration
                        if summary_level != 'full':
                            continue  # skip this when level is 'slim'
                        writer.add_scalar(name + '/max', value.max().item(), total_iter)
                        writer.add_scalar(name + '/min', value.min().item(), total_iter)
                        writer.add_histogram(name, value.cpu().detach().numpy(), total_iter)
                writer.add_scalar('lr/lr', sch.get_lr()[0], total_iter)

                # this method can be used to watch intermediate variables
                for name, model in codec.module.sub_models.items():
                    if summary_level != 'full':
                        continue  # skip this when level is 'slim'
                    if 'y_quant' in name:
                        if hasattr(model, 'rx'):
                            writer.add_scalar('rx/max', model.rx.max().item(), total_iter)
                            writer.add_scalar('rx/min', model.rx.min().item(), total_iter)
                            writer.add_histogram('rx', model.rx.cpu().detach().numpy(), total_iter)
                    if 'z_quant' in name:
                        if hasattr(model, 'rx'):
                            writer.add_scalar('rx/max', model.rx.max().item(), total_iter)
                            writer.add_scalar('rx/min', model.rx.min().item(), total_iter)
                            writer.add_histogram('rx', model.rx.cpu().detach().numpy(), total_iter)

                losses.sort(key=lambda x: x[0])
                loss_fmt = ', '.join(['{}:{:.7f}'.format(k, v) for k, v in losses])

                # TODO log bpp
                to_log(
                    ('[epoch:{}, batch:{}]\t[' + loss_fmt + ']\t[lr:{:.7f}]').format(epoch, iteration, sch.get_lr()[0]))

            # validate for each `show_bpp_interval` iterations
            # if `show_bpp_interval` is less or equal to zero,
            # will never do that
            if args.show_bpp_interval > 0 and rank == 0:
                if total_iter % args.show_bpp_interval == 0:
                    test(codec, val_loader, args, real_bpp=True, quality_entropy=True,
                         as_val=True, val_iter=total_iter, summary_level=summary_level)

                    # test will set codec to eval model
                    codec.train()

            # if args.linklink and torch.cuda.is_available():
            #     link.synchronize()

        # save checkpoint for each `args.snapshot_interval`
        # always save at the end of the last epoch
        if (epoch == args.epoch - 1 or epoch % args.snapshot_interval == 0) and rank == 0:
            torch.save(codec.state_dict(), os.path.join(args.log_dir, 'codec_epoch-{}.pth'.format(epoch)))
            torch.save(opt.state_dict(), os.path.join(args.log_dir, 'opt_epoch-{}.pth'.format(epoch)))
            torch.save(sch.state_dict(), os.path.join(args.log_dir, 'sch_epoch-{}.pth'.format(epoch)))

    # after the last epoch, test the finally saved model
    if rank == 0:
        test(codec, val_loader, args, real_bpp=True, quality_entropy=True,
             as_val=True, val_iter=total_iter, summary_level=summary_level)


def test(codec: BaseCodec, loader, args, real_bpp, quality_entropy, as_val=False, val_iter=-1, test_lambda4=None,
         test_b=None, summary_level='full'):
    stage = CodecStageEnum.VALID if as_val else CodecStageEnum.TEST
    device = args.device

    if not as_val and not args.test_time:
        # set deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # or conv_transpose op is non-deterministic
    if args.test_time:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        to_log('test-time mode: disable cudnn.deterministic and enable cudnn.benchmark')

    assert not (as_val and val_iter < 0)

    if not as_val and args.get('test_time', False):
        codec.prepare_recursively()  # for a precious running speed evaluation

    log_dir = args.log_dir
    result_dir = os.path.join(log_dir, 'results')  # TODO: argument instead

    # read exported prob table
    if args.get('use_table_file'):
        assert not as_val, 'using a prob table file is not allowed during training'
        table_path = os.path.join(log_dir, 'prob_table.dip')
        for tname, table, tails in read_table(table_path):
            args['prob_table/' + tname] = table
            args['prob_table_tails/' + tname] = tails

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
        codec, args = attr_controller(codec, args, 0, stage, test_lambda4=test_lambda4, test_b=test_b)

        eval_sums = {}

        test_time.clear()

        # a list containing res for each image
        res_list = []
        # TODO: move ycbcr process into codec
        for i, x in enumerate(tqdm(loader)):

            if isinstance(x, dict):
                x["cur_epoch"] = 0

            # if i>0:
            #    break
            x_path = x['path']
            # print(x_path)
            key_list = ["img"]
            if "y_coefficients" in x:
                key_list = ["y_coefficients", "cb_coefficients", "cr_coefficients"]
            for key in key_list:
                # x = _x['img']
                # print(key)
                x["img"] = x[key].to(device)

                test_time.mark(name="", new_pipeline=True)

                # x = x.to(device)

                if draw_grad:
                    x["img"] = x["img"].clone().detach().requires_grad_(True)

                args['test_real_bpp'] = real_bpp
                args['y_save_path'] = os.path.join(result_dir, f'{i:d}_{key}.bin')
                args['z_save_path'] = os.path.join(result_dir, f'{i:d}_{key}_side.bin')
                # for residual compression
                args["ry_save_path"] = os.path.join(result_dir, f"{i:d}_{key}_r.bin")
                args["rz_save_path"] = os.path.join(result_dir, f"{i:d}_{key}_r_side.bin")
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
                        xx = x["img"][:, :, l:r, :]
                        output.append({k: v for k, v in codec(xx, stage=stage, args=args).items()})
                        if r == x["img"].shape[2]:
                            break
                        l += 64 - base
                        r = l + 64
                        if x["img"].shape[2] - (r + 1) < 64:
                            r = x["img"].shape[2]

                    x_hat = torch.zeros_like(x["img"])
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
                    outputs['image/original'] = x["img"]
                    outputs['image/reconstruction'] = x_hat
                    outputs['eval/msssim'] = msssim(x["img"], x_hat)
                    psnr = PSNR()
                    outputs['eval/psnr'] = psnr(x["img"], x_hat)

                if draw_grad:
                    for loss_type in ['total', 'msssim', 'mse', 'rmse', 'yuv', 'entropy_total', 'tv']:
                        codec.zero_grad()
                        x["img"].grad = None
                        outputs['loss/' + loss_type].backward(retain_graph=True)
                        grad = x["img"].grad
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
                        # results of current dataset instead of average of historical datasets
                        eval_fmt.append('{}:{:.7f}'.format(name, value))
                        # eval_fmt.append('{}:{:.7f}'.format(name, eval_sums[name] / (i + 1)))
                    elif name.startswith('image/') and rank == 0:
                        if value.dtype is torch.int16:
                            continue
                        # convert NCHW tensor to HWC numpy array
                        if isinstance(value, np.ndarray):
                            img = value
                        else:
                            img = torch.round(value * 255).int().clamp(0, 255)
                            img = img.detach().cpu().numpy()
                            img = np.array(img, dtype=np.uint8).squeeze(0)
                            img = np.transpose(img, [1, 2, 0])

                        if as_val and summary_level == 'full':
                            continue
                            # writer.add_image('%d/' % i + name[len('image/'):],
                            #                  img_tensor=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            #                  global_step=val_iter,
                            #                  dataformats='HWC')
                        else:
                            # if summary_level is set to 'slim',
                            # do not save images to tsbd event but write em to disk directly
                            name = name[len('image/'):]
                            print(name, img.shape)
                            # cv2.imwrite(result_dir + '/%d_%s.png' % (i, name), img)

                        if draw_grad:
                            testwriter.add_image('%d/' % i + name,
                                                 img_tensor=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                                 global_step=val_iter,
                                                 dataformats='HWC')

                eval_fmt = '[' + ', '.join(eval_fmt) + ']'

                # TODO: add options
                # TODO images
                to_log(eval_fmt, link_inited=as_val)
                res_list.append(' '.join(map(str, (x_path[0], outputs.get('eval/bpp', np.inf),
                                                   outputs.get('eval/psnr', np.inf),
                                                   outputs.get('eval/msssim', np.inf)))))

        test_time.output()

        eval_fmt = []
        for name, value in eval_sums.items():
            eval_sums[name] = value / len(loader)
            eval_fmt.append('{}:{:.7f}'.format(name, eval_sums[name]))
        eval_fmt = 'mean: [' + ', '.join(eval_fmt) + ']'

        to_log(eval_fmt, link_inited=as_val)
        # if meta_file_list has >1 list, we only use the first one as the res_file name
        res_file_name = loader.dataset.meta_file_list[0]
        print(res_file_name, flush=True)
        print(os.path.join(result_dir, res_file_name), flush=True)
        with open(os.path.join(result_dir, res_file_name.split('/')[-1]), 'w') as f:
            f.write('\n'.join(res_list))

        if as_val and rank == 0:
            for name, value in eval_sums.items():
                writer.add_scalar(name, value, val_iter)

    if testwriter:
        testwriter.close()


def attr_controller(codec, args, iteration, stage, test_lambda4=None, test_b=None):
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

        if iteration > 0:
            for name, model in codec.module.sub_models.items():
                if 'y_quant' in name:
                    setattr(model, 'iteration', iteration)
                if 'z_quant' in name:
                    setattr(model, 'iteration', iteration)

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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='the directory of experiment')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--test_quality_entropy', default=False, action='store_true')
    parser.add_argument('--test_real_bpp', default=False, action='store_true')
    parser.add_argument('--test_zero_pad', default=False, action='store_true')
    parser.add_argument('--test_time', default=False, action='store_true')
    parser.add_argument('--test_lambda4_id', type=int, default=0)
    parser.add_argument('--test_b', type=float, default=0.)
    parser.add_argument('--to_caffe', default=False, action='store_true')
    parser.add_argument('--to_nart', default=False, action='store_true')
    parser.add_argument('--flops_without_to_caffe', default=False, action='store_true')
    parser.add_argument('--output_act', default=False, action='store_true')
    parser.add_argument('--out_dir', type=str, default=None, \
                        help='if not provided, same as root')
    parser.add_argument('--kept', type=int, default=None)
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--quant_cfg', type=str, default=None)
    parser.add_argument('--tensorboard-summary-level', default='full', choices=['full', 'slim', 'none'],
                        help='full: record curves, images and distributions to tensorboard event file;'
                             'slim: record curves only and save reconstructed images to disk directly')
    parser.add_argument('--fast-train', default=False, action='store_true',
                        help='fast train mode. reduce useless loss calculation during training')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                        help='specify test device')
    parser.add_argument('--nart_out', default='nart', help='out dir of nart engine')
    parser.add_argument('--nart_backend', default='tensorrt', choices=['flops', 'onnx', 'tensorrt'],
                        help='backend of nart')
    _args = parser.parse_args()
    train_args = load_yaml(_args.root)

    kwargs = {k: getattr(_args, k) for k in [
        'test_quality_entropy',
        'test_real_bpp',
        'test_zero_pad',
        'verbose',
        'test_lambda4_id',
        'test_b',
        'test_time',
        'to_caffe',
        'to_nart',
        'flops_without_to_caffe',
        'output_act',
        'out_dir',
        'kept',
        'test_list',
        'quant_cfg',
        'tensorboard_summary_level',
        'fast_train',
        'device',
        'nart_out',
        'nart_backend'
    ]}
    main(train_args, **kwargs)
