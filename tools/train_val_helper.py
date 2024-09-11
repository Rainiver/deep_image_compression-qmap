import argparse
import functools
import os
import pickle
import random
import time
from random import sample
import subprocess
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from codes.AE.adaptive_arithmetic_compress import compress_with_index
from codes.AE.adaptive_arithmetic_decompress import decompress, decompress_with_index
from codes.AE.adaptive_arithmetic_decompress import to_prob_table_min_max
try:
    from integer2 import module as qm
except:
    print("train_val_helper load integer2 failed", flush=True)
    from integer import module as qm
from losses.Lp_Loss import Loss as lp
from losses.PSNR_Loss import Loss as PSNR
from losses.SSIM_Loss import msssim
from losses.TV_Loss import Loss as TV
from losses.entropy_Loss import Loss as Entropy_loss
from losses.grad_Loss import Loss as Grad
from nets.context import context_models
from nets.decoder import decoders
from nets.encoder import encoders
from nets.entropy import entropy_models
from nets.layers import SignalConv2d, MaskedConv2d, MaskedSignalConv2d, SignalConvTranspose2d
from nets.layers_quant import GGQConvTranspose2d, GGBpAct, GGQReLU, ProAct
from nets.decoder_quant import QZDecoder_GG18
from nets.norm import GDN
from nets.param import param_models
from quant.dequantizator import dequantizators
from quant.quantizator import quantizators
from utils import color_space
from utils.extract_quant_info import extract_quant_info
from utils.flops_helper import flops_cal
from utils.quant_info_to_proto import quant_info_to_proto
try:
    import spring.nart.tools.pytorch as pytorch
    from onnx import load as onnxload
    from onnx import save as onnxsave
    from spring.nart.tools.pytorch.network_utils.common import update_input_names, update_output_names
    from tools.param_flops_config import custom_ops, profile, clever_format
except:
    pytorch=None

plt, sns = None, None


def calc(x, y):
    return x * y


def calc_bpp(image, dir):
    return calc_bpp_by_shape(image.shape, dir)


def calc_bpp_by_shape(shape, dir):
    if isinstance(dir, str):
        size = os.path.getsize(dir)
    elif isinstance(dir, list):
        size = sum(os.path.getsize(d) for d in dir)
    else:
        raise TypeError(dir)
    return size * 8 * 3 / functools.reduce(calc, shape)


# if use rank or world_size before init them
# will & should raise an error
rank, world_size = None, None
try:
    import spring.linklink as link
    from utils.distributed_utils import dist_init, reduce_gradients, DistModule

except:
    link = None

TRAIN_MODE = 0
TEST_MODE = 1
VAL_MODE = -1
CAFFE_MODE = 2

TIME = []


def to_log(content, link_inited=True):
    if rank == 0:
        print(content, file=log, flush=True)
        print(content, flush=True)
    if rank is None:
        print(content, flush=True)


def load(models: dict, opt, sch, log_dir, epoch, offset, mode):
    """
    load the models
    :param models: a dict, models to load, whose pth file named {name}_epoch-{epc}.pth
    :param log_dir: where the pth files are saved
    :param epoch: epoch number to load. if negative, load the latest
    :param offset: deprecated.
    :param mode: train or test
    :return:
    """

    assert mode != VAL_MODE  # should not reload during validation

    # TODO: move to utils
    def _detect_latest(prefix, suffix):
        """
        detect the latest file in log_dir with format <prefix><epoch><suffix>
        :param prefix:
        :param suffix:
        :return: epoch, if here's no checkpoints, return a negative value
        """
        checkpoints = os.listdir(log_dir)
        checkpoints = [f for f in checkpoints if f.startswith(prefix) and f.endswith(suffix)]
        checkpoints = [int(f[len(prefix):-len(suffix)]) for f in checkpoints]
        checkpoints = sorted(checkpoints)
        _epoch = checkpoints[-1] if len(checkpoints) > 0 else None
        return _epoch

    def _remove_dist_module_prefix(k: str):
        return k[7:] if k.startswith('module.') else k

    def _load(model, epoch, prefix, suffix, name):
        if model:
            if epoch < 0:
                epoch = _detect_latest(prefix, suffix)

            if epoch is not None:
                ckpt = torch.load(os.path.join(log_dir, '{}{}{}'.format(prefix, epoch, suffix)),
                                  map_location='cpu')
                ckpt = {_remove_dist_module_prefix(k): v for k, v in ckpt.items()}
                kwargs = {}
                if isinstance(model, nn.Module):
                    kwargs['strict'] = False
                model.load_state_dict(ckpt, **kwargs)
                to_log("loaded {} epoch: {}".format(name, epoch), link_inited=mode == TRAIN_MODE)

    for m_name, m in models.items():
        prefix = m_name + '_epoch-'
        _load(m, epoch, prefix, '.pth', m_name)
        models[m_name] = m

    _load(opt, epoch, 'opt_epoch-', '.pth', 'opt')
    _load(sch, epoch, 'sch_epoch-', '.pth', 'sch')


def to_prob_table(y_scaled: np.ndarray, entropy_model, device, scale_factor, per_channel):
    '''
    output: freq of range(0, table_len).
    The output should only depend on those encoded side information, 
    including shape, table_len, y_scaled_min, scale_factor.

    if per_channel is True, will out put C tables, where C denotes number of channels

    also should be careful with any round op, to make sure this table is recoverable
    on decoding side

    for decomress, the actual process should be:
    entropy_model((range(0, table_len)+y_scaled_min)/scale_factor)*y_scaled_size
    '''
    assert entropy_model is not None
    assert isinstance(y_scaled, np.ndarray)
    assert y_scaled.ndim == 4  # N C W H
    assert y_scaled.shape[0] == 1  # compress one per time

    _min = y_scaled.min()
    _max = y_scaled.max()

    return to_prob_table_min_max(_min, _max, y_scaled.shape, entropy_model, device, scale_factor, per_channel)


def get_y_table_with_sigma(_min, len_range_table, shape, entropy_pre, scale_table, scale_factor):
    _max = _min - 2 + len_range_table
    len_scale_table = len(scale_table)
    # each channel for each sigma
    y_range_map = np.array([list(range(_min, _max + 2))] * len_scale_table) \
        .reshape([1, len_scale_table, 1, len_range_table])
    y_range_rescaled_map = y_range_map / scale_factor
    scale_map = np.repeat(scale_table, len_range_table) \
        .reshape([1, len_scale_table, 1, len_range_table])

    y_range_rescaled_map = torch.tensor(y_range_rescaled_map).float()
    scale_map = torch.tensor(scale_map).float()
    if torch.cuda.is_available():
        y_range_rescaled_map = y_range_rescaled_map.cuda()
        scale_map = scale_map.cuda()

    # now we can get a [LEN_SCALE_TABLE X LEN_Y_SCALED] shaped prob table
    # and we will compress data according to this
    y_prob_table_per_sigma = entropy_pre(y_range_rescaled_map, 0., scale_map)
    y_prob_table = y_prob_table_per_sigma.reshape([len_scale_table, len_range_table])
    y_prob_table = y_prob_table.cpu()
    y_prob_table *= shape[1] * shape[2] * shape[3]
    y_prob_table[y_prob_table < 1] = 1
    y_prob_table = y_prob_table.int().numpy()
    return y_prob_table


def get_sigma_aligned_and_index(sigma, scale_table):
    sigma_aligned_idx = torch.zeros_like(sigma, dtype=torch.int)
    sigma_aligned = torch.full_like(sigma, scale_table[0], dtype=torch.float32)
    for ind, scale in enumerate(scale_table[:-1]):
        cmp_mask = (sigma > scale).int()
        cmp_mask_fp = cmp_mask.float()
        sigma_aligned_idx += cmp_mask
        sigma_aligned = (1 - cmp_mask_fp) * sigma_aligned + cmp_mask_fp * scale_table[ind + 1]
    sigma_aligned_idx = sigma_aligned_idx.cpu().numpy()
    return sigma_aligned, sigma_aligned_idx


def train(models: dict, train_loader, opt, sch, args, val_loader):
    global rank
    for name, model in models.items():
        if not model:
            continue
        model.train()

        if args.linklink and torch.cuda.is_available():
            model = DistModule(model, True)
        models[name] = model

    num_pixels = train_loader.batch_size * args.crop_size ** 2
    encoder = models['encoder']
    decoder = models['decoder']
    entropy_pre = models['entropy_pre']
    z_entropy_pre = models['z_entropy_pre']
    z_encoder = models['z_encoder']
    z_decoder = models['z_decoder']
    y_context = models['y_context']
    y_parameter = models['y_parameter']
    quant = models['y_quant']

    tot = sch.last_epoch * len(train_loader)
    tot_val = 0
    l2 = lp(p=2)
    tv = TV()
    grad = Grad()
    bpp = 0
    entropy_loss = Entropy_loss()
    # quant = quantizators(args.y_quant, args.BIT) # moved to models[]
    dequant = dequantizators(args.y_quant, args.BIT)
    z_quant = quantizators(args.z_quant, args.BIT) if z_encoder else None
    z_dequant = dequantizators(args.z_quant, args.BIT) if z_encoder else None

    for epoch in range(sch.last_epoch, args.epoch):
        sch.step()
        for id, data in enumerate(train_loader):
            if len(args.lambda4s) > 1:
                sampled_lambda = sample(args.lambda4s[1:], 1)[0]
                encoder.module.sampled_lambda = sampled_lambda
                decoder.module.sampled_lambda = sampled_lambda
                if z_encoder:
                    z_encoder.module.sampled_lambda = sampled_lambda
                    z_decoder.module.sampled_lambda = sampled_lambda
            if len(args.b_range) > 1:
                assert len(args.b_range) == 3
                sampled_b = random.uniform(args.b_range[1], args.b_range[2])
                sampled_delta = 2 ** sampled_b
                quant = quantizators(args.y_quant, sampled_delta)
                dequant = dequantizators(args.y_quant, sampled_delta)
                z_quant = quantizators(args.z_quant, sampled_delta) if z_encoder else None
                z_dequant = dequantizators(args.z_quant, sampled_delta) if z_encoder else None

            tot += 1
            if torch.cuda.is_available():
                data = data.cuda()
            if args.y_encode[-3:] == "yuv":
                y = color_space.bgr_to_yuv(data)
                y = color_space.yuv_to_yuv420(y)
                if torch.cuda.is_available():
                    for i in range(3):
                        y[i] = y[i].cuda()
            else:
                y = data.clone()
            y = encoder(y)
            y_tilde = quant(y)
            y_index = quant(y_tilde, True) if args.y_quant == "SOFT" else y_tilde
            if z_encoder:
                z = z_encoder(y) if (y_parameter and not y_context) else z_encoder(torch.abs(y))
                z_tilde = z_quant(z)
                z_likelihoods = z_entropy_pre(z_tilde)
                sigma = z_decoder(z_tilde)
                mu = 0.
                if y_parameter:
                    if y_context:
                        context = y_context(y_tilde)
                        assert context.shape == sigma.shape
                        prior = torch.cat([sigma, context], 1)
                    else:
                        prior = torch.cat([sigma, torch.abs(sigma)], 1)
                    params = y_parameter(prior)
                    n_channel = params.shape[1] // 2
                    sigma = params[:, :n_channel, ...]
                    mu = params[:, n_channel:, ...]
                entropy_pre_extra_args = [mu, sigma]
            else:
                entropy_pre_extra_args = []
                z_likelihoods = 1.
            y_likelihoods = entropy_pre(y_index,
                                        *entropy_pre_extra_args)  # changed to y_index because of soft quant interval
            if args.show_bpp_interval > 0 and rank == 0:
                if tot // args.show_bpp_interval > tot_val:
                    tot_val = tot // args.show_bpp_interval

                    test(models, val_loader, args, real_bpp=True, quality_entropy=True, as_val=True, val_iter=tot)

                    for m in models.values():
                        if not m:
                            continue
                        m.train()

            y_dq = dequant(y_tilde)
            output = decoder(y_dq)
            loss1 = 1 - msssim(output, data, normalize=True)
            loss2 = nn.MSELoss(reduction='mean')(output * 255, data * 255)
            loss3 = tv(output)
            y_entropy_loss = entropy_loss(y_likelihoods) / num_pixels
            z_entropy_loss = entropy_loss(z_likelihoods) / num_pixels if z_encoder else 0.
            loss4 = y_entropy_loss + z_entropy_loss  # bpp
            if len(args.lambda4s) > 1:
                loss = args.lambda1 * loss1 + args.lambda2 * loss2 + sampled_lambda * loss4
            else:
                loss = args.lambda1 * loss1 + args.lambda2 * loss2 + args.lambda4 * loss4
            opt.zero_grad()
            loss.backward()
            if args.linklink and torch.cuda.is_available():
                for model in models.values():
                    if model:
                        reduce_gradients(model, True)
            opt.step()
            if tot % args.show_interval == 0 and rank == 0:
                to_log(
                    '[epoch:{}, batch:{}]\t[loss:{:.7f},{:.7f},{:.7f},{:.7f}]\t[bpp:{:.7f}]\t[lr:{:.7f}]'
                        .format(epoch, id, loss1.item(), loss2.item(), loss3.item(),
                                loss4.item(), bpp, sch.get_lr()[0])
                )
                if rank == 0:
                    writer.add_scalar('Y/Min', y.min().item(), tot)
                    writer.add_scalar('Y/Max', y.max().item(), tot)
                    writer.add_scalar('Y_Likelihoods/Min', y_likelihoods.min().item(), tot)
                    writer.add_scalar('Y_Likelihoods/Max', y_likelihoods.max().item(), tot)
                    writer.add_scalar('Loss/Entropy Y', y_entropy_loss.item(), tot)
                    if z_encoder:
                        writer.add_scalar('Z/Max', z.max().item(), tot)
                        writer.add_scalar('Z/Min', z.min().item(), tot)
                        writer.add_scalar('Z_Likelihoods/Min', z_likelihoods.min().item(), tot)
                        writer.add_scalar('Z_Likelihoods/Max', z_likelihoods.max().item(), tot)
                        writer.add_scalar('Loss/Entropy Z', z_entropy_loss.item(), tot)

                    writer.add_scalar('Out/Min', output.min().item(), tot)
                    writer.add_scalar('Out/Max', output.max().item(), tot)
                    writer.add_scalar('Loss/MS-SSIM', loss1.item(), tot)
                    writer.add_scalar('Loss/MSE', loss2.item(), tot)
                    writer.add_scalar('Loss/TV', loss3.item(), tot)
                    writer.add_scalar('Loss/Entropy Total', loss4.item(), tot)
                    writer.add_scalar('LR/LR', sch.get_lr()[0], tot)

            if args.linklink and torch.cuda.is_available():
                link.synchronize()

        if epoch % args.snapshot_interval == 0 and rank == 0:
            for name, model in models.items():
                if model:
                    torch.save(model.state_dict(), os.path.join(args.log_dir, '{}_epoch-{}.pth'.format(name, epoch)))
            torch.save(opt.state_dict(), os.path.join(args.log_dir, 'opt_epoch-{}.pth'.format(epoch)))
            torch.save(sch.state_dict(), os.path.join(args.log_dir, 'sch_epoch-{}.pth'.format(epoch)))


def pad(x, base=32):
    b, c, h, w = x.shape
    H = (h // base + 1) * base
    W = (w // base + 1) * base
    _y = torch.zeros([b, c, H, W])
    _y[:, :, H - h:H, 0:w] = x.clone()
    _y[:, :, 0:h, W - w:W] = x.clone()
    if x.dtype == torch.int:
        _y = _y.int()
    _y[:, :, 0:h, 0:w] = x
    return _y


# noinspection PyUnresolvedReferences
def test(models: dict, loader, args, real_bpp, quality_entropy, as_val=False, val_iter=-1, test_lambda4=None,
         test_b=None):
    mode = VAL_MODE if as_val else TEST_MODE

    assert not (as_val and val_iter < 0)

    TIME.append(time.time())

    log_dir = args.log_dir
    result_dir = os.path.join(log_dir, 'results')  # TODO: argument instead
    if test_lambda4 is not None:
        result_dir = os.path.join(log_dir, 'results_{}_{}'.format(test_lambda4, test_b))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with torch.no_grad():
        for m in models.values():
            if not m:
                continue
            m.eval()

        encoder = models['encoder']
        decoder = models['decoder']
        entropy_pre = models['entropy_pre']
        z_encoder = models['z_encoder']
        z_decoder = models['z_decoder']
        z_entropy_pre = models['z_entropy_pre']
        y_context = models['y_context']
        y_parameter = models['y_parameter']
        quant = models['y_quant']

        if len(args.lambda4s) > 1:
            encoder.sampled_lambda = args.lambda4s[0] if test_lambda4 is None else test_lambda4
            decoder.sampled_lambda = args.lambda4s[0] if test_lambda4 is None else test_lambda4
            if z_encoder:
                z_encoder.sampled_lambda = args.lambda4s[0] if test_lambda4 is None else test_lambda4
                z_decoder.sampled_lambda = args.lambda4s[0] if test_lambda4 is None else test_lambda4

        cnt, s1, s2, s3, s4, s5 = 0, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), 0
        y_entropy, z_entropy = torch.zeros(1), torch.zeros(1)
        psnr = PSNR()
        tv = TV()
        entropy_loss = Entropy_loss()

        z_quant = quantizators(args.z_quant, args.BIT, train=False) if z_encoder else None

        y_scaled_stream = []
        prob_table_stream = []
        compress_save_path_stream = []
        per_channel_stream = []
        img_shape_stream = []
        real_bpp_calc_path_stream = []
        prob_table_index_stream = []
        for i, x in enumerate(tqdm(loader)):
            cnt += 1

            if torch.cuda.is_available():
                encoder = encoder.cuda()
                decoder = decoder.cuda()
                entropy_pre.cuda()
                x = x.cuda()

            gt = x.clone()

            base = 64 if z_encoder is not None else 32  # TODO: add an option
            b, c, h, w = x.shape
            if x.shape[2] % base != 0 or x.shape[3] % base != 0:
                x = pad(x, base)  # TODO: try reflecting padding ?
                if torch.cuda.is_available():
                    x = x.cuda()

            if args.y_encode[-3:] == "yuv":
                xx = color_space.bgr_to_yuv(x)
                xx = color_space.yuv_to_yuv420(xx)
                if torch.cuda.is_available():
                    for i in range(3):
                        xx[i] = xx[i].cuda()
            else:
                xx = x.clone()

            # quant = quantizators(args.y_quant, args.BIT, train=False)
            dequant = dequantizators(args.y_quant, args.BIT)

            scale_factor = (1 << args.BIT) - 1
            if len(args.b_range) > 1:
                assert len(args.b_range) == 3
                sampled_b = args.b_range[0] if test_b is None else test_b
                sampled_delta = 2 ** sampled_b
                quant = quantizators(args.y_quant, sampled_delta, train=False)
                dequant = dequantizators(args.y_quant, sampled_delta)
                z_quant = quantizators(args.z_quant, sampled_delta, train=False) if z_encoder else None
                z_dequant = dequantizators(args.z_quant, sampled_delta) if z_encoder else None
                scale_factor = 1. / sampled_delta

            y = encoder(xx)
            y_tilde = quant(y)

            if z_encoder:
                z = z_encoder(y) if (y_parameter and not y_context) else z_encoder(torch.abs(y))
                z_tilde = z_quant(z)
                z_likelihoods = z_entropy_pre(z_tilde)
                sigma = z_decoder(z_tilde)  # used to estimate y
                mu = 0.
                if y_parameter:
                    if y_context:
                        context = y_context(y_tilde)
                        assert context.shape == sigma.shape
                        prior = torch.cat([sigma, context], 1)
                    else:
                        prior = torch.cat([sigma, torch.abs(sigma)], 1)
                    params = y_parameter(prior)
                    n_channel = params.shape[1] // 2
                    sigma = params[:, :n_channel, ...]
                    mu = params[:, n_channel:, ...]
                    assert sigma.shape == mu.shape
                    assert sigma.shape == y.shape

                SCALES_MIN = 0.11
                SCALES_MAX = 256
                SCALES_LEVELS = 64

                # We need to limit the number of possible values of sigma to a finite set, because the prob table
                # for the range coder needs to be precomputed and saved to ensure consistency of prob model (like gaussian * U) across platforms
                # (of course, z_decoder and context etc also need to give determined values as stated in GG19I)
                scale_table = list(np.exp(np.linspace(
                    np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)))

                sigma_aligned, sigma_aligned_idx = get_sigma_aligned_and_index(sigma, scale_table)

                entropy_pre_extra_args = [mu, sigma_aligned]
            else:
                entropy_pre_extra_args = []
                z_likelihoods = torch.ones(1)

            y_tilde = quant(y)
            y_index = quant(y_tilde, True) if args.y_quant == "SOFT" else y_tilde

            if entropy_pre:
                y_likelihoods = entropy_pre(y_index, *entropy_pre_extra_args)  # changed to y_index since soft quant
            else:
                y_likelihoods = torch.zeros(1)

            p = y_tilde.clone()  # TODO: refactor here, name `p` is confusion

            if y_parameter or y_context:
                y_scaled = torch.round((y - mu) * scale_factor).int().cpu().numpy()
            else:
                y_scaled = y_index.int().cpu().numpy() if args.y_quant == "SOFT" \
                    else torch.round((p) * scale_factor).int().cpu().numpy()

            if z_encoder:
                z_scaled = torch.round(z_tilde * scale_factor).int().cpu().numpy()

            TIME.append(time.time())

            if real_bpp:
                compress_save_path = os.path.join(result_dir, '%d.txt' % i)

                # TODO: merge the saved files into one
                sigma_save_path = os.path.join(result_dir, '%d_z_side_info.txt' % i)

                if args.entropy_AE:

                    # pack values:
                    code_shapes = [x.shape]
                    per_channel_opts = [args.compress_per_channel, False] if z_encoder else [args.compress_per_channel]

                    # prob_tables:
                    if z_encoder:
                        z_prob_table = to_prob_table(z_scaled, z_entropy_pre, scale_factor, args.compress_per_channel)
                        assert z_scaled.ndim == 4
                        assert z_scaled.shape[0] == 1
                        z_index = np.arange(z_scaled.shape[1]).repeat(z_scaled.shape[2] * z_scaled.shape[3])
                        z_index = z_index.reshape(z_scaled.shape)

                        _min = y_scaled.min()
                        _max = y_scaled.max()
                        len_range_table = _max - _min + 2
                        len_scale_table = len(scale_table)

                        y_prob_table = get_y_table_with_sigma(_min, len_range_table, y_scaled.shape,
                                                              entropy_pre, scale_table, scale_factor)

                        y_index = sigma_aligned_idx
                        assert y_scaled.shape == y_index.shape

                        prob_tables = [z_prob_table, y_prob_table]
                        scaled_values = [z_scaled, y_scaled]
                        prob_table_indexes = [z_index, y_index]
                        save_paths = [sigma_save_path, compress_save_path]
                    else:
                        if args.compress_per_channel:
                            y_index = np.arange(y_scaled.shape[1]).repeat(y_scaled.shape[2] * y_scaled.shape[3])
                        else:
                            y_index = np.zeros(y_scaled.shape)
                        y_index = y_index.reshape(y_scaled.shape)
                        y_prob_table = to_prob_table(y_scaled, entropy_pre, scale_factor, args.compress_per_channel)

                        prob_tables = [y_prob_table]
                        scaled_values = [y_scaled]
                        prob_table_indexes = [y_index]
                        save_paths = [compress_save_path]
                else:
                    prob_tables = [None, None]

                if args.test_n_jobs != 0 and mode == TEST_MODE:
                    y_scaled_stream += scaled_values
                    prob_table_stream += prob_tables
                    compress_save_path_stream += save_paths
                    img_shape_stream += code_shapes
                    per_channel_stream += per_channel_opts
                    real_bpp_calc_path_stream.append(save_paths)
                    prob_table_index_stream += prob_table_indexes
                else:
                    for code, index, save_path, prob_table in \
                            zip(scaled_values, prob_table_indexes, save_paths, prob_tables):
                        compress_with_index(code, index, save_path, args.adapt, prob_table, backend=args.ae_backend)
                    bpp = calc_bpp(x, save_paths)
                    s5 += bpp
            TIME.append(time.time())
            if args.decompress:
                if z_encoder is None:
                    y_hat = decompress(compress_save_path, args.compress_per_channel, args.adapt, True, None)
                    y_hat = torch.tensor(y_hat).float()
                    if torch.cuda.is_available():
                        y_hat = y_hat.cuda()
                else:
                    # using z
                    def z_prob_table_fn(_min, table_len, shape):
                        table = to_prob_table_min_max(_min, _min - 2 + table_len, shape,
                                                      z_entropy_pre,
                                                      scale_factor, True)
                        assert (np.array(table) == np.array(z_prob_table)).all()
                        return table

                    def y_prob_table_fn(_min, table_len, shape):
                        table = get_y_table_with_sigma(_min, table_len, shape,
                                                       entropy_pre,
                                                       scale_table, scale_factor)
                        assert (np.array(table) == np.array(y_prob_table)).all()
                        return table

                    # z_prob_table can be work out with given models before inference
                    def per_channel_idx_fn(_min, _table_len, shape):
                        assert len(shape) == 4
                        assert shape[0] == 1
                        c, w, h = shape[1:]
                        index = np.arange(c).repeat(w * h).reshape(shape)
                        return index

                    z_hat = decompress_with_index(sigma_save_path, per_channel_idx_fn, args.adapt, z_prob_table_fn,
                                                  backend=args.ae_backend)
                    z_hat = torch.tensor(z_hat).float()
                    if torch.cuda.is_available():
                        z_hat = z_hat.cuda()
                    z_hat = z_hat / scale_factor
                    sigma_hat = z_decoder(z_hat)

                    _, y_index = get_sigma_aligned_and_index(sigma_hat, scale_table)
                    assert (y_index == sigma_aligned_idx).all()
                    y_hat = decompress_with_index(compress_save_path, y_index, args.adapt, y_prob_table_fn,
                                                  backend=args.ae_backend)
                    # TODO: add back mu for y_hat
                    y_hat = torch.tensor(y_hat).float()
                    if torch.cuda.is_available():
                        y_hat = y_hat.cuda()
                    y_hat = y_hat / scale_factor

                # end of decompress, use y_hat as y_tilde
                y_tilde = y_hat

            TIME.append(time.time())
            y_tilde = dequant(y_tilde)
            x = decoder(y_tilde)
            x = torch.clamp(x, 0, 1)
            TIME.append(time.time())
            x = x[:, :, 0:h, 0:w]
            num_pixels = x.size(2) * x.size(3)
            if quality_entropy:
                s1 += msssim(x, gt, normalize=False)
                s2 += psnr(x, gt)
                s3 += tv(x)
                y_entropy = entropy_loss(y_likelihoods) / num_pixels
                z_entropy = entropy_loss(z_likelihoods) / num_pixels
                s4 += y_entropy + z_entropy
            to_log(
                '[loss:{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}]\t'.format(
                    s1.item() / cnt, s2.item() / cnt, s3.item() / cnt, s4.item() / cnt, s5 / cnt), link_inited=as_val
            )
            to_log(
                '[entropy:{:.7f},{:.7f}]\t'.format(
                    y_entropy.item(), z_entropy.item()), link_inited=as_val
            )

            mse = (x - gt) ** 2
            mse = mse.detach().cpu().numpy()
            mse = mse.squeeze(0)
            mse = np.transpose(mse, [1, 2, 0])
            mse = np.sum(mse, axis=2)
            mse = mse / mse.max()
            mse_uint8 = np.array(mse * 255, dtype=np.uint8)

            x = torch.round(x * 255).int().abs()

            # saving
            x = x.detach().cpu().numpy()
            x = np.array(x, dtype=np.uint8).squeeze(0)
            x = np.transpose(x, [1, 2, 0])

            gt = torch.round(gt * 255).int().abs()
            gt = gt.detach().cpu().numpy()
            gt = np.array(gt, dtype=np.uint8).squeeze(0)
            gt = np.transpose(gt, [1, 2, 0])

            # visualization y, z, sigma and others
            # TODO: add an option to disable below code
            # for a faster test (it can be unacceptably slow)
            y = y.detach().cpu().numpy()
            y = abs(y)
            to_log(y.shape, link_inited=as_val)
            to_log(y.max(), link_inited=as_val)
            y = y / y.max()
            y = y.squeeze(0)
            y = np.transpose(y, [1, 2, 0])
            y_max = np.max(y, axis=2)
            y_max_uint8 = np.array(y_max * 255, dtype=np.uint8)

            py_viz = y_likelihoods.detach().cpu().numpy()
            py_viz = py_viz.squeeze(0)
            py_viz = np.transpose(py_viz, [1, 2, 0])
            py_viz = np.sum(py_viz, axis=2)
            py_viz = py_viz / py_viz.max()
            py_viz_uint8 = np.array(py_viz * 255, dtype=np.uint8)

            if z_encoder:
                z_tilde = z_tilde.detach().cpu().numpy()
                z_tilde = abs(z_tilde)
                to_log(z_tilde.shape, link_inited=as_val)
                to_log(z_tilde.max(), link_inited=as_val)
                z_tilde = z_tilde / z_tilde.max() * 255
                z_tilde = np.array(z_tilde, dtype=np.uint8).squeeze(0)
                z_tilde = np.transpose(z_tilde, [1, 2, 0])
                z_tilde_max = np.max(z_tilde, axis=2)

                sigma = sigma.detach().cpu().numpy()
                sigma = abs(sigma)
                sigma[sigma < 0.11] = 0.11
                sigma = sigma.squeeze(0)
                sigma = np.transpose(sigma, [1, 2, 0])
                sigma /= sigma.max()

                mu = 0.
                y_sigma_ratio = np.log1p((y - mu) / sigma)
                y_sigma_ratio = (y_sigma_ratio / y_sigma_ratio.max())
                y_sigma_ratio_max = np.max(y_sigma_ratio, axis=2)
                y_sigma_ratio_max_uint8 = np.array(y_sigma_ratio_max * 255, dtype=np.uint8)

                sigma_max = np.max(sigma, axis=2)
                sigma_max_uint8 = (sigma_max * 255).astype(np.uint8)

                pz_viz = z_likelihoods.detach().cpu().numpy()
                pz_viz = pz_viz.squeeze(0)
                pz_viz = np.transpose(pz_viz, [1, 2, 0])
                pz_viz = np.sum(pz_viz, axis=2)
                pz_viz = pz_viz / pz_viz.max()
                pz_viz_uint8 = np.array(pz_viz * 255, dtype=np.uint8)

                # TODO can/should we keep the original filename names?
            # TODO a better way to manage the save paths, such as using a manager object
            save_path = os.path.join(result_dir, '%d.png' % i)
            to_log(save_path, link_inited=as_val)
            gt_save_path = os.path.join(result_dir, '%d_gt.png' % i)
            y_save_path = os.path.join(result_dir, '%d_y.png' % i)
            y_stat_save_path = os.path.join(result_dir, '%d_y_stat.png' % i)
            z_save_path = os.path.join(result_dir, '%d_z.png' % i)
            y_sigma_ratio_save_path = os.path.join(result_dir, '%d_y_sigma.png' % i)
            sigma_vis_save_path = os.path.join(result_dir, '%d_sigma.png' % i)
            sigma_stat_save_path = os.path.join(result_dir, '%d_sigma_stat.png' % i)
            y_sigma_ratio_stat_save_path = os.path.join(result_dir, '%d_y_sigma_stat.png' % i)
            y_prob_save_path = os.path.join(result_dir, '%d_y_prob.png' % i)
            z_prob_save_path = os.path.join(result_dir, '%d_z_prob.png' % i)
            mse_save_path = os.path.join(result_dir, '%d_mse.png' % i)

            if as_val and rank == 0:
                writer.add_image('Val{}/Original'.format(i), cv2.cvtColor(gt, cv2.COLOR_BGR2RGB), val_iter,
                                 dataformats='HWC')
                writer.add_image('Val{}/Decompressed'.format(i), cv2.cvtColor(x, cv2.COLOR_BGR2RGB), val_iter,
                                 dataformats='HWC'
                                 )
            elif not as_val:
                # test mode, should save the image
                global plt, sns
                if plt is None:
                    try:
                        import matplotlib.pyplot as plt
                    except:
                        plt = None
                if sns is None:
                    try:
                        import seaborn as sns
                        sns.set()
                    except:
                        sns = None
                cv2.imwrite(save_path, x)
                cv2.imwrite(gt_save_path, gt)
                cv2.imwrite(y_save_path, y_max_uint8)
                cv2.imwrite(y_prob_save_path, py_viz_uint8)
                cv2.imwrite(mse_save_path, mse_uint8)
                if sns:
                    plt.figure()
                    sns.distplot(np.log1p(y.flatten()))
                    plt.savefig(y_stat_save_path)
                    plt.close()
                if z_encoder:
                    cv2.imwrite(z_save_path, z_tilde_max)
                    cv2.imwrite(y_sigma_ratio_save_path, y_sigma_ratio_max_uint8)
                    cv2.imwrite(sigma_vis_save_path, sigma_max_uint8)
                    cv2.imwrite(z_prob_save_path, pz_viz_uint8)
                    if sns:
                        plt.figure()
                        sns.distplot(sigma.flatten())
                        plt.savefig(sigma_stat_save_path)
                        plt.close()

                        plt.figure()
                        sns.distplot(y_sigma_ratio.flatten())
                        plt.savefig(y_sigma_ratio_stat_save_path)
                        plt.close()

        if real_bpp and args.test_n_jobs != 0 and mode == TEST_MODE:
            assert len(y_scaled_stream) == len(prob_table_stream) \
                   == len(compress_save_path_stream)
            assert len(real_bpp_calc_path_stream) == len(img_shape_stream)

            from joblib import Parallel, delayed

            tasks = (
                (delayed(compress_with_index)(
                    y_scaled, index, compress_save_path, args.adapt, prob_table)
                    for (y_scaled, index, compress_save_path, prob_table)
                    in zip(y_scaled_stream, prob_table_index_stream, compress_save_path_stream, prob_table_stream)))
            Parallel(n_jobs=args.test_n_jobs, verbose=100)(tasks)

            bpps = (
                calc_bpp_by_shape(shape, path) for shape, path in
                zip(img_shape_stream, real_bpp_calc_path_stream)
            )
            s5 = sum(bpps)
        loss1 = s1 / cnt
        loss2 = s2 / cnt
        loss3 = s3 / cnt
        loss4 = s4 / cnt
        mbpp = s5 / cnt

        to_log('mean: [loss:{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}]\t'.format(
            loss1.item(), loss2.item(), loss3.item(), loss4.item(), mbpp), link_inited=as_val
        )

        if as_val and rank == 0:
            writer.add_scalar('Val/MS-SSIM', loss1.item(), val_iter)
            writer.add_scalar('Val/PSNR', loss2.item(), val_iter)
            writer.add_scalar('Val/TV', loss3.item(), val_iter)
            writer.add_scalar('Val/Entropy Y', loss4.item(), val_iter)
            writer.add_scalar('Val/Bits Per Pixel', mbpp, val_iter)
        TIME.append(time.time())

def convert_nart(models, args):
    '''
    convert pytorch model to nart tensorrt engine
    '''
    folder = args.nart_out
    backend = args.nart_backend
    os.makedirs(folder, exist_ok=True)

    with torch.no_grad():
        for name, m in models.items(): # for models, check model_builder.py
            if 'encoder' in name or 'decoder' in name: 
                print("converting {}".format(name), flush=True)

                if 'y_decoder' in name and not args.get('post', 'NONE') in ['NONE', 'usm', 'idct']:
                    m.post = models['post']
                    print("merge post into y_decoder", flush=True)

                for item in m.modules():
                    if hasattr(item, 'to_caffe'):
                        item.to_caffe = True
                    if isinstance(item, GDN):
                        item.prepare()
                    if isinstance(item, SignalConv2d) or isinstance(item, SignalConvTranspose2d):
                        item.prepare()
                    if isinstance(item, (GGBpAct, ProAct)):
                        item.prepare()
                    if isinstance(item, QZDecoder_GG18):
                        item.debug = False

                #torch to onnx
                m.cpu()
                m.eval()
                path = os.path.join(folder, name)

                # TODO: set input shapes by args
                if name == "y_encoder":
                    input_shapes = [(3, 1088, 1920)]
                    # input_shapes = [(3, 512, 768)]
                    # input_shapes = [(3, 256, 256)]
                elif name == "y_decoder":
                    input_shapes = [(192, 68, 120)]
                    # input_shapes = [(192, 32, 48)]
                    # input_shapes = [(192, 16, 16)]
                elif name == "z_encoder":
                    input_shapes = [(192, 68, 120)]
                    # input_shapes = [(192, 32, 48)]
                    # input_shapes = [(192, 16, 16)]
                elif name == "z_decoder":
                    input_shapes = [(192, 17, 30)]
                    # input_shapes = [(192, 8, 12)]
                    # input_shapes = [(192, 4, 4)]
                else:
                    raise ValueError

                dummy_input = torch.randn((1, *input_shapes[0]))
                if backend == 'flops':
                    macs, params = profile(model=m,
                                           inputs=(dummy_input,),
                                           custom_ops=custom_ops)
                    flops = macs * 2
                    macs, flops, params = clever_format([macs, flops, params], "%.4f")
                    print('\033[1;34m net: %s, params: %s, macs: %s, flops: %s \033[0m' %
                          (name , params, macs, flops))
                    continue

                torch.onnx.export(model=m,
                                  args=dummy_input,
                                  f=path + '.onnx',
                                  input_names=['%s_data' % name],
                                  output_names=['%s_out' % name],
                                  opset_version=11)

                #convert z_decoder.onnx to z_decoder_int.onnx
                if 'z_decoder' in name:
                    subprocess.call(['python3', 'nart/convert_to_integer_onnx.py','-path', path + '.onnx'])

                if backend == 'onnx':
                    continue

                print("====convert: start of onnx to %s" % backend)
                if backend == 'tensorrt':
                    #convert z_decoder_int.onnx to z_decoder_int.bin
                    if 'z_decoder' in name:
                        int_onnx_path = path + '_int.onnx'
                        int_engine_path = path + '_int.bin'
                        subprocess.call(['python3', '-m', 'spring.nart.switch', '-v', '-t', 'cuda',
                                         '--skme', int_onnx_path, '-o', int_engine_path])
                    else:
                        # onnx to trt engine
                        onnx_path = path + '.onnx'
                        engine_path = path + '.bin'
                        cfg_path = os.path.join('nart', 'trt_nart_cfg.yml')
                        subprocess.call(['python3', '-m', 'spring.nart.switch', '-v', '-t', 'tensorrt',
                                         '-c', cfg_path, '--skme', onnx_path, '-o', engine_path])
                else:
                    raise NameError
                print("====convert: end of onnx to %s" % backend)

def convert_caffe(models, mode):
    '''mode = CAFFE_MODE or FLOPS_MODE
    in FLOPS_MODE only report flops from flops_cal, 
    otherwise will convert caffe and report flops
    '''
    FLOPS_MODE = 3

    with torch.no_grad():
        for name, m in models.items():
            if not m:
                continue
            if 'z_encoder' in name:
                # if backend = [caffe, ppl], replace abs with pow(pow(x, 2), 0.5)
                m._forward_pre_hooks.clear()
                m.alpha = nn.Parameter(torch.Tensor([2]), requires_grad=False)
                m.epsilon = nn.Parameter(torch.Tensor([0.5]), requires_grad=False)
                m.register_forward_pre_hook(lambda m, inp: torch.pow(torch.pow(*inp, m.alpha), m.epsilon))

            # avoid some models in nets/model_builder.py
            if 'entropy' in name or 'quant' in name or isinstance(m, nn.Identity) or 'data_augment' in name or 'data_collect' in name:
                print("skip {}".format(name), flush=True)
                continue

            print("converting {}".format(name), flush=True)
            m.cpu()
            m.eval()
            extract_quant = False
            for item in m.modules():
                if hasattr(item, 'to_caffe'):
                    item.to_caffe = True

                if isinstance(item, GDN):
                    item.prepare()
                if isinstance(item, SignalConv2d) or isinstance(item, SignalConvTranspose2d):
                    item.prepare()
                if isinstance(item, MaskedSignalConv2d) or isinstance(item, MaskedConv2d):
                    item.prepare()

                # TODO: support converting clamp and round division layer in nart
                if isinstance(item, (GGBpAct, ProAct)):
                    item.prepare()
                if isinstance(item, GGQConvTranspose2d):
                    pass
                if isinstance(item, GGQReLU):
                    pass

                # handling quantized layers using our integer lib
                M_NoBnConv2d = qm.get_quant_conv(False)
                M_NoBnSignalConv2d = qm.get_quant_conv(True)
                M_NoBnConvTranspose2d = qm.NoBnConvTranspose2d
                M_EMAAct = qm.EMAAct

                if isinstance(item, (M_NoBnConv2d, M_NoBnSignalConv2d, M_NoBnConvTranspose2d)):
                    item.prepare2q()
                    item.to_caffe2q = True
                    if hasattr(item, 'to_caffe'):
                        item.to_caffe = False  # avoid changing the behavior of parent class

                if isinstance(item, M_EMAAct):
                    extract_quant = True
                    item.to_caffe2q = True

            if 'encoder' in name:
                data_shape = 16
            elif 'decoder' in name:
                data_shape = 4
            else:
                data_shape = 4

            # flops here
            flops, params, flops_str, params_str = flops_cal(m, (m.caffe_channels, data_shape, data_shape))
            print('flops {}'.format(flops), flush=True)
            print('params {}'.format(params), flush=True)
            print('flops_str {}'.format(flops_str), flush=True)
            print('params_str {}'.format(params_str), flush=True)

            if not mode == FLOPS_MODE:
                model_folder = 'caffe'
                if not os.path.exists(model_folder):
                    os.mkdir(model_folder)
                path = os.path.join(model_folder, name)
                print('converting:', name)
                print(m)
                with pytorch.convert_mode():
                    pytorch.convert(
                            m, [(m.caffe_channels, data_shape, data_shape)],
                            filename=path,
                            input_names=["data"],
                            output_names=["out"],
                            verbose=True
                    )
                caffe_model = name + '.caffemodel'
                cmd = f"cd {model_folder} && python -m spring.nart.tools.caffe.convert -a {caffe_model}"
                print(cmd, flush=True)
                os.system(cmd)

                # for models using our integer lib, extract quantization params and write into prototxt
                if extract_quant:
                    print('extracting quant info for {}'.format(name), flush=True)
                    quant_info, layer_names = extract_quant_info(m)
                    quant_info_path = os.path.join(model_folder, name + "_quant_info.pk")
                    with open(quant_info_path, "wb") as f:
                        # might be usefull for converting to other platforms like trt
                        pickle.dump(quant_info, f, protocol=0)
                        print(f"quant_info exported to: {quant_info_path}", flush=True)

                    # here we use the '-convert.prototxt', which is after the merge ALL process
                    src_proto = os.path.join(model_folder, name + '-convert' + '.prototxt')
                    dst_proto = os.path.join(model_folder, name + '-QUANT' + '.prototxt')
                    quant_info_to_proto(quant_info=quant_info, layer_names=layer_names, src_proto=src_proto,
                                        dst_proto=dst_proto)


def main(args, **kwargs):
    global world_size, rank, log, writer, args_, link
    # test arguments
    test_real_bpp = 'test_real_bpp' in kwargs and kwargs['test_real_bpp']
    test_quality_entropy = 'test_quality_entropy' in kwargs and kwargs['test_quality_entropy']
    is_test_mode = test_real_bpp or test_quality_entropy
    test_lambda4 = args.lambda4s[kwargs['test_lambda4_id']] if 'test_lambda4_id' in kwargs else None
    test_b = kwargs['test_b'] if 'test_b' in kwargs else None
    test_time = kwargs['test_time'] if 'test_time' in kwargs else None
    to_caffe = 'to_caffe' in kwargs and kwargs['to_caffe']

    # run mode
    mode = TEST_MODE if is_test_mode else TRAIN_MODE
    if to_caffe:
        mode = CAFFE_MODE

    log_dir = args.log_dir
    rank, world_size = dist_init() if link else (0, 1)
    if rank == 0 and not os.path.exists(log_dir):
        os.mkdir(log_dir)
    while rank != 0 and not os.path.exists(log_dir):
        time.sleep(0.5)
    log_fname = 'test_log.txt' if is_test_mode else 'train_log.txt'
    log = open(os.path.join(log_dir, log_fname), 'w')

    summary_dir = os.path.join(args.log_dir, 'events')
    if rank == 0 and mode == TRAIN_MODE:
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        writer = SummaryWriter(log_dir=summary_dir)

    args_ = args
    to_log(args, link_inited=False)

    y_encoders_kwargs = args.get('y_encode_args', {})
    z_encoders_kwargs = args.get('z_encode_args', {})
    entropy_model_kwargs = args.get('entropy_model_args', {})
    y_decoders_kwargs = args.get('y_decode_args', {})
    z_decoders_kwargs = args.get('z_decode_args', {})
    y_parameter_kwargs = args.get('y_parameter_args', {})
    y_context_kwargs = args.get('y_context_args', {})

    models = {
        'encoder': encoders(args.y_encode, out_channels=args.num_features_encode,
                            **y_encoders_kwargs),
        'decoder': decoders(args.y_decode, out_channels=args.num_features_decode,
                            **y_decoders_kwargs),
        'entropy_pre': entropy_models(args.entropy_model, args.num_features_entropy_model,
                                      **entropy_model_kwargs),
        'z_encoder': encoders(args.z_encode,
                              in_channels=args.num_features_encode,
                              out_channels=args.num_features_encode,
                              **z_encoders_kwargs),
        'z_decoder': decoders(args.z_decode,
                              in_channels=args.num_features_encode,
                              out_channels=args.num_features_decode,
                              **z_decoders_kwargs),
        'z_entropy_pre': entropy_models(args.z_entropy_model, args.num_features_entropy_model),
        'y_parameter': param_models(args.get('y_parameter', 'NONE'),
                                    in_channels=args.num_features_encode * 4,
                                    out_channels=args.num_features_encode * 2,
                                    **y_parameter_kwargs),
        'y_context': context_models(args.get('y_context', 'NONE'),
                                    in_channels=args.num_features_encode,
                                    out_channels=args.num_features_encode * 2,
                                    **y_context_kwargs),
        'y_quant': quantizators(args.y_quant, train=True)
    }

    opt = torch.optim.Adam([{'params': model.parameters()} for model in models.values() if model], lr=args.lr)

    for name, model in models.items():
        if model:
            to_log(name + ':', link_inited=False)
            to_log(model, link_inited=False)

    sch = torch.optim.lr_scheduler.MultiStepLR(opt, args.lr_milestion, gamma=args.lr_scheduler_gamma)
    loader = get_data_loader(mode, args)

    if torch.cuda.is_available():
        for name, model in models.items():
            if model:
                model = model.cuda()
                models[name] = model

    load(models, opt, sch, args.log_dir, args.load_epoch, 0, mode)

    if mode == TRAIN_MODE:
        val_loader = get_data_loader(VAL_MODE, args)
        train(models, loader, opt, sch, args, val_loader)
    elif mode == TEST_MODE:
        test(models, loader, args, real_bpp=test_real_bpp, quality_entropy=test_quality_entropy,
             test_lambda4=test_lambda4, test_b=test_b)
        if test_time:
            print(TIME)
            print("encode time:{:.7f}\n"
                  "compress time:{:.7f}\n"
                  "decompress time:{:.7f}\n"
                  "decode time:{:.7f}\n"
                  "evaluate time:{:.7f}".format(TIME[1] - TIME[0], TIME[2] - TIME[1], TIME[3] - TIME[2],
                                                TIME[4] - TIME[3], TIME[5] - TIME[4]))
    elif mode == CAFFE_MODE:
        convert_caffe(models)
    if link:
        link.finalize()
    log.close()
    if rank == 0 and mode == TRAIN_MODE:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--data_dir', default='../../../dataset/compression')
    parser.add_argument('--train_classes', default=['train'])
    parser.add_argument('--test_classes', default=['valid'])
    parser.add_argument('--GPUs', default=16)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--num_workers', default=16)
    parser.add_argument('--lr', default=0.0003)
    parser.add_argument('--lr_milestion', default=[20, 40, 70, 120, 200, 500])
    parser.add_argument('--lr_scheduler_gamma', default=0.5)
    parser.add_argument('--epoch', default=1001)
    parser.add_argument('--show_interval', default=1)
    parser.add_argument('--show_bpp_interval', default=100)
    parser.add_argument('--test_interval', default=2)
    parser.add_argument('--snapshot_interval', default=5)
    parser.add_argument('--linklink', default=True)
    parser.add_argument('--load_epoch', default=-1)
    parser.add_argument('--num_features_encode', default=30)
    parser.add_argument('--num_features_decode', default=30)
    parser.add_argument('--num_features_entropy_model', default=30)
    parser.add_argument('--y_encode', default="YGG17")
    parser.add_argument('--y_decode', default="YGG17")
    parser.add_argument('--y_quant', default="RT")
    parser.add_argument('--z_encode', default="NONE")
    parser.add_argument('--z_decode', default="NONE")
    parser.add_argument('--z_quant', default="NONE")
    parser.add_argument('--entropy_model', default="fake")
    parser.add_argument('--entropy_AE', default=False)
    parser.add_argument('--BIT', default=8)
    parser.add_argument('--b_range', default=[0])
    parser.add_argument('--lambda1', default=1)
    parser.add_argument('--lambda2', default=1)
    parser.add_argument('--lambda4', default=1)
    parser.add_argument('--lambda4s', default=[1])
    parser.add_argument('--test_quality_entropy', default=False)
    parser.add_argument('--test_real_bpp', default=False)
    parser.add_argument('--test_time', default=False)

    args = parser.parse_args()
    main(args)
