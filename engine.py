"""
Train and eval functions used in main.py
"""
import math
import sys
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist

from timm.models import model_parameters
from timm import utils
from timm.utils import accuracy, AverageMeter
from utils.tensors import reduce_tensor
import torch.nn as nn
import numpy as np
     
def train_one_epoch(
        local_rank,
        csv_logger,
        print_rank0,
        student,
        teacher,
        criterion,
        kl,
        data_loader, 
        optimizer,
        device, 
        epoch, 
        loss_scaler, 
        max_norm,
        momentum_scheduler = None,
        lr_scheduler = None,
        student_ema = None, 
        mixup_fn = None,
        set_training_mode=True, 
        args = None
):
    student.train()
    teacher.eval()
    print_freq = 10

    loss_meter = AverageMeter()
    xe_loss_meter = AverageMeter()
    repr_distill_loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()

    forward_timer_meter = AverageMeter()
    backward_timer_meter = AverageMeter()
    forward_backward_timer_meter = AverageMeter()
        
    T = 1.
    num_iters = len(data_loader)
    for iter, data in enumerate(data_loader):
        t0 = time.time()

        fwd_bwd_t0 = time.time()

        # -- map inputs to device
        samples, targets, idx = data
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        bsz = samples.shape[0]
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        fwd_t0 = time.time()
        with torch.cuda.amp.autocast():
            # -- teacher
            with torch.no_grad():
                if args.alpha > 0 and args.gamma > 0:
                    z_t_conv = teacher.forward_features(samples)
                    y_t_conv = teacher.forward_head(z_t_conv)

            if args.alpha == 0 and args.gamma == 0:
                # -- no distillation
                y_s = student.module(samples)
            else:
                z_s_cls, z_s_distill, z_s = student.module.forward_features(samples)
                # -- multi-head prediction
                y_s = student.module.head(z_s_cls)
                y_s_distill = student.module.head_dist(z_s_distill)

            repr_distill_loss = torch.tensor(0.0)
            if args.alpha > 0:
                # pool over token-dim for student and spatial-dims for teacher
                b = z_s.shape[0]

                z_s_pool = z_s.mean(1)
                b, c, h, w = z_t_conv.shape
                z_t_conv_pool = z_t_conv.view(b, c, h * w).mean(-1)
                z_s_pool = student.module.projector(z_s_pool)

                z_t_conv_norm = F.layer_norm(z_t_conv_pool, (z_t_conv_pool.shape[1],))
                repr_distill_loss = args.alpha * F.smooth_l1_loss(z_s_pool, z_t_conv_norm)

            if args.gamma > 0:
                kl_loss = args.gamma * kl(F.log_softmax(y_s_distill / T, dim=-1), F.softmax(y_t_conv / T, dim=-1)) * T * T
            else:
                kl_loss = torch.tensor(0.0)

            xe_loss = criterion(y_s, targets)
            loss = xe_loss + kl_loss + repr_distill_loss

        fwd_t1 = time.time()
        delta = fwd_t1 - fwd_t0
        forward_timer_meter.update(delta, 1)

        # -- update metrics
        loss_meter.update(loss.item(), bsz)
        xe_loss_meter.update(xe_loss.item(), bsz)
        repr_distill_loss_meter.update(repr_distill_loss.item(), bsz)
        kl_loss_meter.update(kl_loss.item(), bsz)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print_rank0("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        bwd_t0 = time.time()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(
            loss, optimizer, clip_grad=max_norm,
            parameters=student.parameters(), create_graph=is_second_order
        )

        optimizer.zero_grad()
        torch.cuda.synchronize()
        lr_scheduler.step_update(epoch * num_iters + iter)
        
        bwd_t1 = time.time()
        delta = bwd_t1 - bwd_t0
        backward_timer_meter.update(delta, 1)

        fwd_bwd_t1 = time.time()
        delta = fwd_bwd_t1 - fwd_bwd_t0
        forward_backward_timer_meter.update(delta, 1)

        # Step 3. momentum update of target encoder
        if student_ema:
            with torch.no_grad():
                m = next(momentum_scheduler)
                for param_q, param_k in zip(student.parameters(), student_ema.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        # -- Logging
        if local_rank == 0:
            csv_logger.log(epoch, iter, xe_loss, repr_distill_loss, kl_loss)
            if (iter % print_freq == 0):
                delta_time = time.time() - t0
                t0 = time.time()
                estimated_time = delta_time * num_iters / print_freq
                # multiply by print_freq because we override every iteration
                delta_time *= print_freq
                estimated_time *= print_freq
                print_rank0(
                    f"Epoch: {epoch}, Iter: [{iter}/{num_iters}], lr: [{optimizer.param_groups[0]['lr']:.6f}], "
                    f"Mem: {torch.cuda.max_memory_allocated() / 1024.**2:.2e}, "
                    f"XE Loss: {xe_loss_meter.avg:.4f}, KL Loss: {kl_loss_meter.avg:.4f}, "
                    f"Repr Loss: {repr_distill_loss_meter.avg:.4f}, "
                    f"t/print freq: {delta_time:.2f}s, t/epoch: {estimated_time:.2f}s, "
                    f"total est: {(estimated_time * 300.) / (60. * 60. * 24.):.3f} days, "
                )

    return loss_value

@torch.no_grad()
def validate(data_loader, model, print_rank0, device, print_freq=50):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target, _) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            # measure accuracy and record loss
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % print_freq == 0 and dist.get_rank() == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print_rank0(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB'
            )
            
    if dist.get_rank() == 0:
        print_rank0(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    return acc1_meter.avg, acc5_meter.avg