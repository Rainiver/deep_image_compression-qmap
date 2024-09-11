# -*- coding: utf-8 -*-
# Rundong Li <lirundong@sensetime.com>
# (c) 2018, SenseTime

import logging
from types import MethodType

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from integer import module as qm


M_NoBnConv2d = qm.get_quant_conv(False)
M_NoBnSignalConv2d = qm.get_quant_conv(True)

try:
    import spring.linklink as link
except:
    link = None

__all__ = ["QuantEpochScheduler"]

def to_log(content, flush=True, link_inited=True):
    rank = link.get_rank()
    if rank == 0:
        print(content, flush=flush)

def toggle_quant(scheduler, quant_weight, quant_act):
    '''control enable_quant attr
    '''

    def _toggle(module):
        for n, m in module.named_modules():
            if isinstance(m, qm.EMAAct):
                m.enable_quant = quant_act

                if m.quant_mode == "dorefa":
                    for stat in (m.stat_min, m.stat_max):
                        if stat > 0.:
                            stat.fill_(1.)
                        elif stat < 0.:
                            stat.fill_(-1.)
            elif isinstance(m, (M_NoBnConv2d, M_NoBnSignalConv2d, qm.NoBnConvTranspose2d, qm.EMABnConv2d, qm.QuantLinear)):
                m.enable_quant = quant_weight

    if isinstance(scheduler.model, DistributedDataParallel):
        for copy in scheduler.model._module_copies:
            _toggle(copy)
    else:
        _toggle(scheduler.model)
        to_log('toggle_quant: weight {}, act {}'.format(quant_weight, quant_act), flush=True)


def toggle_running_stat(scheduler, running_stat, use_batch_stat=None, quick_stat=False):
    # stat_mode = 'quick' if quick_stat else 'EMA'

    stat_mode = None
    def _toggle(module):
        for n, m in module.named_modules():
            if isinstance(m, qm.EMAAct):
                m.running_stat = running_stat
                stat_mode = m.stat_mode
                if use_batch_stat is not None:
                    m.use_batch_stat = use_batch_stat

    scheduler.running_stat = running_stat
    if isinstance(scheduler.model, DistributedDataParallel):
        for copy in scheduler.model._module_copies:
            _toggle(copy)
    else:
        _toggle(scheduler.model)
        to_log('toggle_running_stat: running_stat {}, stat_mode {}, use_batch_stat {}'.format(running_stat, stat_mode, use_batch_stat), flush=True)


def toggle_freeze_bn(scheduler, freeze_bn):

    def _toggle(module):
        for n, m in module.named_modules():
            if isinstance(m, qm.EMABnConv2d):
                m.freeze_bn = freeze_bn

    if isinstance(scheduler.model, DistributedDataParallel):
        for copy in scheduler.model._module_copies:
            _toggle(copy)
    else:
        _toggle(scheduler.model)


class QuantEpochScheduler:
    def __init__(self, model, lr_scheduler, quant_cfg):
        self.model = model.sub_models # ModuleDict
        self.lr_scheduler = lr_scheduler
        self.optimizer = lr_scheduler.optimizer
        to_log(quant_cfg, flush=True)
        self.schedule = quant_cfg['training_schedule']
        self.model_cfg = quant_cfg['model_cfg']
        self.test_cfg = quant_cfg['test_cfg']
        self.stat_start = 0
        self.stat_stop = 0
        self.running_stat = False
        self.quick_stat = False
        self.current_phase = None
        self.current_iter = 0
        self.last_epoch = self.lr_scheduler.last_epoch
        to_log('last_epoch: {}'.format(self.last_epoch), flush=True)
        self.toggle_running_stat = MethodType(toggle_running_stat, self)
        self.toggle_quant = MethodType(toggle_quant, self)
        self.toggle_freeze_bn = MethodType(toggle_freeze_bn, self)

        self.update_quant_cfg()

        accu_epoch = 0
        base_lr = self.get_lr()
        for i, phase in enumerate(self.schedule):
            phase["milestone"] = accu_epoch + phase["num_epoch"]
            phase["lr"] = [lr * phase["lr_scale"] for lr in base_lr]
            if self.current_phase is None and \
                    accu_epoch <= self.last_epoch < phase["milestone"]:
                self.current_phase = (i, phase)
                to_log("current phase at init is {} {}".format(i, phase), flush=True)
            accu_epoch += phase["num_epoch"]
        self.total_epoch = accu_epoch

    def update_quant_cfg(self, is_test=False):
        i = 0
        for n, m in self.model.named_modules():
            if isinstance(m, qm.EMAAct):
                to_log(n)
                new_cfg = self.model_cfg['EMAAct']
            elif isinstance(m, (M_NoBnConv2d, M_NoBnSignalConv2d, qm.NoBnConvTranspose2d)):
                new_cfg = self.model_cfg['NoBnConv2d_NoBnConvTranspose2d']
                to_log(n)
            else:
                to_log("unrecognized:" + n)
                continue            

            for item in new_cfg:
                assert item in m.__dict__
            
            if i == 0:
                to_log(m.__dict__, flush=True)

            m.__dict__.update(new_cfg)
            m.conditional_init()
            m = m.cuda()

            if i == 0:
                to_log(m.__dict__, flush=True)

            if is_test:
                new_cfg = self.test_cfg
                m.__dict__.update(new_cfg)
                to_log(m.__dict__, flush=True)

            i += 1
        to_log('total quantized layers: {}'.format(i), flush=True)

    def get_lr(self):
        return self.lr_scheduler.get_lr()

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)

    def step(self, epoch=None):
        self.epoch(epoch)

    def epoch(self, epoch=None):
        self.lr_scheduler.step(epoch)
        self.current_iter = self.lr_scheduler.last_epoch - 1
        to_log('current_iter is {}'.format(self.current_iter), flush=True)
        logger = logging.getLogger("global")
        phase_idx, phase = self.current_phase
        if phase["milestone"] <= self.current_iter and self.current_iter < self.total_epoch:
            while True:
                phase_idx += 1
                if self.schedule[phase_idx]["num_epoch"] != 0:
                    break
            self.current_phase = (phase_idx, self.schedule[phase_idx])
            to_log("current phase at epoch {} is {} {}".format(self.current_iter, phase_idx, self.schedule[phase_idx]), flush=True)
            phase_idx, phase = self.current_phase
        for i, group in enumerate(self.lr_scheduler.optimizer.param_groups):
            group["lr"] = phase["lr"][i]

        use_batch_stat = phase.get("use_batch_stat", False)
        if phase["running_stat"]:
            if phase.get("quick_stat", False):
                self.quick_stat = True
                for n, m in self.model.named_modules():
                    if isinstance(m, qm.EMAAct) and False: # no need to clear, clear is conflict with resume
                        m.stat_min.zero_()
                        m.stat_max.zero_()
                        m.batch_min.zero_()
                        m.batch_max.zero_()
                        m.stat_updated = False
                        to_log(f"clear stats at {n} for quick stat...", flush=True)
            else:
                self.quick_stat = False

            # epoch-grained statistic
            self.toggle_running_stat(True, use_batch_stat, self.quick_stat)
        else:
            self.toggle_running_stat(False, use_batch_stat)
        self.toggle_quant(phase["quant_w"], phase["quant_a"])
