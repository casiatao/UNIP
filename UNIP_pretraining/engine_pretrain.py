# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module, teacher: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, remain_epochs,
                    log_writer=None,
                    args=None,
                    ema_teacher=None,
                    momentum_schedule=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    
    if args.use_ema:
        # common params
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in model.module.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in ema_teacher.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
        params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, imgs in enumerate(metric_logger.log_every(data_loader, print_freq, remain_epochs, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        #samples = samples.to(device, non_blocking=True)
        B, C, H, W = imgs.shape
        N = B
        L = H // 16 * W // 16

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                teacher_qk = teacher(imgs.to(device, non_blocking=True))
            qk_loss = model(imgs.to(device, non_blocking=True), teacher_qk)
        
        loss = qk_loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"train loss: {qk_loss.item()}")
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        norm = loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=args.clip_grad,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            norm_value = norm.item()
            # print(f"grad norm: {norm_value}")
            optimizer.zero_grad()
            
            if args.use_ema:
                it = int((len(data_loader) * epoch + data_iter_step) / accum_iter)  # global training iteration
                # EMA update for the teacher
                with torch.no_grad():
                    m = momentum_schedule[it]  # momentum parameter
                    for param_q, param_k in zip(params_q, params_k):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()

        metric_logger.update(total_loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        if (data_iter_step + 1) % accum_iter == 0:
            metric_logger.update(grad_norm=norm_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
            
        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if (data_iter_step + 1) % accum_iter == 0:
                log_writer.add_scalar('grad_norm', norm_value, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}