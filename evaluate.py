# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
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
from timm.utils import NativeScaler

from utils.lr_scheduler import build_scheduler
from utils.optimizer import create_optimizer

from models.deit import DEIT_MODELS
from models.regnet import REGNET_MODELS
MODELS = {**DEIT_MODELS, **REGNET_MODELS}

from datasets import build_dataset
from engine import validate
from utils.logging import CSVLogger

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

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
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

    # -- add projector
    # student.projector = torch.nn.utils.parametrizations.orthogonal(nn.Linear(student.num_features, teacher.num_features))

    # move all to gpu (once)
    student.to(device)
    if args.student_ema:
        student_ema.to(device)

    # NOTE: we do not need distributed data parallel for teacher w/ no gradients
    student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[local_rank])
    student_without_ddp = student.module

    checkpoint = torch.load(args.student_ckpt, map_location='cpu')
    checkpoint = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    student.load_state_dict(checkpoint['student'], strict=False)

    acc1, acc5 = validate(data_loader_val, student, print_rank0, device)
    print_rank0(f"Accuracy of the student network on the {len(dataset_val)} test images: {acc1:.1f}%")