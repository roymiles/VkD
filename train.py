# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# fix error with pretrained=True for timm models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import datetime
import numpy as np
import time
import json
import os
os.environ['CURL_CA_BUNDLE'] = ''
import sys
import logging
import copy
import shutil
import yaml

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ApexScaler, NativeScaler

from utils.lr_scheduler import build_scheduler
from utils.optimizer import create_optimizer

from models.deit import DEIT_MODELS
from models.regnet import REGNET_MODELS
MODELS = {**DEIT_MODELS, **REGNET_MODELS}

from datasets import build_dataset
from engine import train_one_epoch, validate
from augment import aug_generator
from utils.logging import CSVLogger
from samplers import SubsetRandomSampler

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --
log_timings = True
log_freq = 10
checkpoint_freq = 20
# --

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args):
    # fix the seed for reproducibility
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device("cuda:{}".format(local_rank))
    torch.cuda.set_device(local_rank)

    seed = 1 + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    print('I am rank %d in this world of size %d!' % (local_rank, world_size))

    def print_rank0(msg):
        if dist.get_rank() == 0:
            logger.info(msg)

    dataset_val, _ = build_dataset(is_train=False, args=args)
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=world_size, rank=local_rank, shuffle=True
    )

    if len(dataset_val) % world_size != 0:
        print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
        
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=world_size, rank=local_rank, shuffle=False
    )

    # indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    # sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    ipe = len(data_loader_train)
    data_loader_train.dataset.transform = aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        drop_last=False
    )

    # -- mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    # -- testing the data loader
    verify_data_loader_speed = False
    if verify_data_loader_speed:
        print_rank0(f"num iterations = {len(data_loader_train)}")
        t0 = time.time()
        for idx, (samples, targets) in tqdm(enumerate(data_loader_train)):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            print_rank0(f"current iteration = {idx}")

        t1 = time.time()
        delta = t1 - t0
        if local_rank == 0:
            print_rank0(f"elapsed time: {delta}s")
            exit("done.")

    print_rank0(f"Creating student: {args.student_model}")

    # custom definition
    student = MODELS[args.student_model](
        num_classes=args.nb_classes,
        img_size=args.input_size
    )
            
    n_parameters = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print_rank0(f'number of params: {n_parameters}')

    momentum_scheduler = None
    student_ema = None
    if args.student_ema:
        student_ema = copy.deepcopy(student)
        for p in student_ema.parameters():
            p.requires_grad = False

    print_rank0(f"Creating CNN teacher model: {args.teacher}")
    # conv_teacher = MODELS[args.teacher]()
    teacher = create_model(
        args.teacher,
        pretrained=True
    )

    for p in teacher.parameters():
        p.requires_grad = False

    if args.teacher_path:
        checkpoint = torch.load(args.teacher_path, map_location='cpu')
        checkpoint = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        teacher.load_state_dict(checkpoint)
    else:
        print(f"{bcolors.WARNING}Warning: Using the default Hugging Face weights for the teacher.")

    teacher.to(device)
    teacher.eval()

    # validate teacher accuracy
    # acc1, acc5 = validate(data_loader_val, teacher, print_rank0, device)
    # print_rank0(f"Accuracy of the teacher network on the {len(dataset_val)} test images: {acc1:.1f}%")
    # exit()

    # -- momentum schedule
    ipe_scale = 1.0
    ema = [0.996, 1.0]
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*args.epochs*ipe_scale)
                          for i in range(int(ipe*args.epochs*ipe_scale)+1))

    # -- add training/distillation components. Moved original code over to torch.parametrizations for code simplicity. 
    student.projector = torch.nn.utils.parametrizations.orthogonal(nn.Linear(student.num_features, teacher.num_features, bias=False))

    # move all to gpu (once)
    student.to(device)
    if args.student_ema:
        student_ema.to(device)

    # NOTE: we do not need distributed data parallel for teacher w/ no gradients
    # optimizer = create_optimizer(student)
    student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[local_rank])
    # student = torch.nn.parallel.DataParallel(student, device_ids=[local_rank])
    student_without_ddp = student.module

    if args.student_ckpt != 'none':
        checkpoint = torch.load(args.student_ckpt, map_location='cpu')
        checkpoint = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        student.load_state_dict(checkpoint['student'], strict=False)

    if args.eval_student:
        acc1, acc5 = validate(data_loader_val, student, print_rank0, device)
        print_rank0(f"Accuracy of the student network on the {len(dataset_val)} test images: {acc1:.1f}%")
        exit()

    # ...
    assert world_size in [1, 2], "May need to adjust hyperparameters when using more GPUs. Currently we only tested with up to 2 GPUs with an effective batch size < 1024."

    if not args.unscale_lr:
        # NOTE: does not scale to effective batch sizes > 1024
        linear_scaled_lr = args.lr * args.batch_size * world_size / 512.0
        args.lr = linear_scaled_lr

    optimizer = create_optimizer(student_without_ddp)
    loss_scaler = NativeScaler()
    # loss_scaler = ApexScaler()

    lr_scheduler = build_scheduler(
        optimizer, ipe,
        epochs=args.epochs, 
        warmup_epochs=0,
        decay_epochs=10, 
        min_lr=5e-6, 
        warmup_lr=5e-6,   
        scheduler_name='cosine',
        decay_rate=0.1    
    )

    criterion = LabelSmoothingCrossEntropy()
    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    # -- saving
    output_dir = Path(args.output_dir)

    if not args.resume and local_rank == 0:
        # only create these if not resuming from previous run
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # copy important .py files for later reference too
        script_dir = str(output_dir) + '/scripts/'
        if not os.path.isdir(script_dir):
            os.mkdir(script_dir)

        def save_script(name):
            dump = str(output_dir) + f'/scripts/{name.replace("/", "_")}.py'
            cur_path = os.path.dirname(os.path.realpath(__file__))
            shutil.copy(f"{cur_path}/{name}.py", dump)

        save_script('train')
        save_script('engine')
        save_script('models/deit')
        save_script('datasets')

        dump = str(output_dir) + '/params.yaml'
        with open(dump, 'w+') as f:
            yaml.dump(args, f)

    dist.barrier()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        student.load_state_dict(checkpoint['student'], strict=False)
        
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch']

        if args.student_ema:
            student_ema.load_state_dict(checkpoint['student_ema'])

        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])

        lr_scheduler.step(args.start_epoch)
        if momentum_scheduler:
            for _ in range(args.start_epoch*ipe):
                next(momentum_scheduler)

    if args.eval:
        acc1, acc5 = validate(data_loader_val, student, print_rank0, device)
        print_rank0(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")  
        dist.barrier()
        return
    
    # -- make csv_logger
    csv_logger = None
    csv_logger_val = None
    if local_rank == 0:
        log_file = os.path.join(output_dir, f'r{local_rank}_train.csv')
        val_log_file = os.path.join(output_dir, f'r{local_rank}_val.csv')
        csv_logger = CSVLogger(log_file,
                                ('%d', 'epoch'),
                                ('%d', 'itr'),
                                ('%.5f', 'xe_loss'),
                                ('%.5f', 'repr_distill_loss'),
                                ('%.5f', 'kl_loss'))
        
        csv_logger_val = CSVLogger(val_log_file,
                                    ('%d', 'epoch'),
                                    ('%.5f', 'acc1'),
                                    ('%.5f', 'acc5'))

    # -- checkpointing
    save_path = lambda epoch : os.path.join(output_dir, f'ep{epoch}.pth.tar')
    latest_path = os.path.join(output_dir, f'latest.pth.tar')
    def save_checkpoint(
        student, student_ema, optimizer,
        loss_scaler, epoch, loss_value, 
        batch_size, world_size, lr
    ):
        # note that we do not save the teacher to save space.
        student_ema_dict = student_ema.state_dict() if student_ema is not None else None
        save_dict = {
            'student': student.state_dict(),
            'student_ema': student_ema_dict,
            'optimizer': optimizer.state_dict(),
            'scaler': loss_scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_value,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }

        if local_rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path(epoch + 1))

    # wait for rank 0
    dist.barrier()
    print_rank0(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    kl = torch.nn.KLDivLoss(reduction='batchmean')
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        loss_value = train_one_epoch(
            local_rank, csv_logger, print_rank0,
            student, teacher, criterion, kl, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, momentum_scheduler, lr_scheduler, student_ema, mixup_fn,
            set_training_mode=args.train_mode,
            args = args
        )

        lr_scheduler.step(epoch)

        if args.output_dir:
            save_checkpoint(
                student, student_ema, optimizer,
                loss_scaler, epoch, loss_value, 
                args.batch_size, world_size, optimizer.param_groups[0]["lr"]
            )

        acc1, acc5 = validate(data_loader_val, student, print_rank0, device)
        print_rank0(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if local_rank == 0:
            csv_logger_val.log(epoch, acc1, acc5)
        
        if local_rank == 0:
            if max_accuracy < acc1:
                max_accuracy = acc1
                if args.output_dir:
                    save_checkpoint(
                        student, student_ema, optimizer,
                        loss_scaler, epoch, loss_value, 
                        args.batch_size, world_size, optimizer.param_groups[0]["lr"]
                    )
            print_rank0(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print_rank0('Training time {}'.format(total_time_str))

    if args.student_ema:
        # final evaluation of ema
        acc1, acc5 = validate(data_loader_val, student_ema, print_rank0, device)
        print_rank0(f"Accuracy of the [ema] network on the {len(dataset_val)} test images: {acc1:.1f}%")
        csv_logger_val.bar()
        csv_logger_val.log(epoch, acc1, acc5)

        dist.barrier()