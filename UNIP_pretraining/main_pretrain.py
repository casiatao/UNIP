import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torchvision.transforms.functional import InterpolationMode

import models_unip
import models_vit
import models_teacher

from engine_pretrain import train_one_epoch

from util import lr_sched
from util.log import Logger
from util.infmix import InfMix

def get_args_parser():
    parser = argparse.ArgumentParser('Distill pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters
    parser.add_argument('--model', default='unip_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--infpre_path', default='/path/to/infpre', type=str,
                        help='infpre dataset path')
    parser.add_argument('--in1k_path', default='/path/to/in1k', type=str,
                        help='imagenet dataset path')
    parser.add_argument('--coco_path', default='/path/to/coco', type=str,
                        help='coco dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--pretrain', default='',
                        help='init with pretrain checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    if (torch.__version__).startswith('2'):
        parser.add_argument("--local-rank", default=-1, type=int)
    else:
        parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--dist_on_itp", type=bool, default=False)
    # teacher settings
    parser.add_argument("--teacher_path", type=str)
    parser.add_argument("--teacher_model", type=str)
    parser.add_argument('--intermediate', default=12, type=int,
                        help='Distill intermediate layer of teacher')
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--use_dino', action='store_true', default=False)
    
    
    parser.add_argument('--last_heads', default=6, type=int,
                        help='the number of heads of the last layer of teacher')
    parser.add_argument('--loss_type', default='KL', type=str)
    parser.add_argument('--clip_grad', default=None, type=float)
    
    # dataset settings
    parser.add_argument('--data_ratio', default=1.0, type=float)
    parser.add_argument('--joint_crop', action='store_true', default=False)
    parser.add_argument('--use_in1k', action='store_true', default=False)
    parser.add_argument('--rgb_gray', action='store_true', default=False)
    parser.add_argument('--use_coco', action='store_true', default=False)
    parser.add_argument('--per_cls_num', default=100, type=int)
    parser.add_argument('--spec_dataset', default=None, type=str)
    
    # ema settings
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_momentum', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    
    return parser



def main(args):
    t = time.strftime("-%Y%m%d-%H%M%S", time.localtime()) 
    filename = 'log' + t + '.txt'
    log = Logger(os.path.join(args.output_dir, filename))
    sys.stdout = log
    
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = Compose([
            RandomResizedCrop(size=(args.input_size, args.input_size), scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.3801, 0.3801, 0.3801], std=[0.1871, 0.1871, 0.1871]),
        ])
    if args.spec_dataset is not None:
        if args.spec_dataset == '':
            spec_dataset = []
        else:
            spec_dataset = args.spec_dataset.split(',')
        dataset_train = InfMix(infpre_path=args.infpre_path, in1k_path=args.in1k_path, coco_path=args.coco_path, transforms=transform_train, use_in1k=args.use_in1k, per_cls_num=args.per_cls_num, use_coco=args.use_coco, spec_dataset=spec_dataset, rgb_gray=args.rgb_gray)
    else:
        dataset_train = InfMix(infpre_path=args.infpre_path, in1k_path=args.in1k_path, coco_path=args.coco_path, transforms=transform_train, use_in1k=args.use_in1k, per_cls_num=args.per_cls_num, use_coco=args.use_coco, data_ratio=args.data_ratio, rgb_gray=args.rgb_gray)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    teacher = models_teacher.__dict__[args.teacher_model](intermediate=args.intermediate, temperature=args.temperature)
    if args.use_dino:
        teacher.load_state_dict(torch.load(args.teacher_path, map_location="cpu"), strict=False)
    else:    
        teacher.load_state_dict(torch.load(args.teacher_path, map_location="cpu")["model"], strict=False)
    teacher.eval()


    model = models_unip.__dict__[args.model](last_heads=args.last_heads, loss_type=args.loss_type)
    
    model.to(device)
    teacher.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    if args.use_ema:
        model_name = args.model.replace('unip_', '')
        ema_teacher = models_vit.__dict__[model_name]()
        ema_teacher.to(device)
        ema_teacher.load_state_dict(model_without_ddp.state_dict(), strict=False)
        for p in ema_teacher.parameters():
            p.requires_grad = False
        print(f"len_dataloader: {len(data_loader_train)}, accum_iter: {args.accum_iter}")
        momentum_schedule = lr_sched.cosine_scheduler(args.ema_momentum, 1,
                                            total_iters=int(args.epochs * len(data_loader_train) / args.accum_iter))

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, teacher, data_loader_train,
            optimizer, device, epoch, loss_scaler, args.epochs - epoch,
            log_writer=log_writer,
            args=args,
            ema_teacher=ema_teacher if args.use_ema else None,
            momentum_schedule=momentum_schedule if args.use_ema else None
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            if args.use_ema:
                misc.save_ema_teacher(args=args, epoch=epoch, ema_teacher=ema_teacher)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
