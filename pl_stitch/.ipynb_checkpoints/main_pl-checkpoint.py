import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter

# Local application imports
import utils
import models
from models.head import iBOTHead, TemporalHead
from models.puzzle_decoder import PuzzleDecoder
from loader import ImageFolderMask_puzzle, Temporal_RandStep_dataset, RepeatDataset

def get_args_parser():
    parser = argparse.ArgumentParser('pl_stitch', add_help=False)

    # --- Model parameters ---
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 
                 'swin_tiny','swin_small', 'swin_base', 'swin_large'],
        help="Name of architecture to train.")
    parser.add_argument('--patch_size', default=16, type=int, 
        help="Size in pixels of input square patches.")
    parser.add_argument('--window_size', default=7, type=int, 
        help="Size of window (Swin Transformer only).")
    parser.add_argument('--out_dim', default=8192, type=int, 
        help="Dimensionality of output for [CLS] token.")
    parser.add_argument('--patch_out_dim', default=8192, type=int, 
        help="Dimensionality of output for patch tokens.")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, 
        help="Whether to share the same head for [CLS] token and patch tokens.")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, 
        help="Whether to share the same head for teacher.")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="Whether to weight normalize the last layer of the head.")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, 
        help="Base EMA parameter for teacher update.")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head.")
    parser.add_argument('--act_in_head', default='gelu',
        help="Activation function in projection head.")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (MIM) in backbone.")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', 
        help="Ratio of partial prediction.")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', 
        help="Variance of partial prediction ratio.")
    parser.add_argument('--pred_shape', default='block', type=str, 
        help="Shape of partial prediction.")
    parser.add_argument('--pred_start_epoch', default=0, type=int, 
        help="Start epoch to perform masked image prediction.")
    
    # --- Loss Weights ---
    parser.add_argument('--lambda1', default=1.0, type=float, help="Loss weight for [CLS] tokens.")
    parser.add_argument('--lambda2', default=1.0, type=float, help="Loss weight for masked patch tokens.")
    parser.add_argument('--lambda_video', default=1.0, type=float, help="Loss weight for temporal branch.")
    parser.add_argument('--lambda_puzzle', default=0.4, type=float, help="Loss weight for puzzle branch.")
    parser.add_argument('--lambda_temporal', default=1, type=float, help="Loss weight for temporal token masking.")
        
    # --- Temperature ---
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help="Initial teacher temperature.")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="Final teacher temperature.")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="Initial patch temperature.")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help="Final patch temperature.")
    parser.add_argument('--warmup_teacher_temp_epochs', default=3, type=int, help="Warmup epochs for temperature.")

    # --- Optimization ---
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="Use mixed precision training.")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="Initial weight decay.")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="Final weight decay.")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="Max gradient norm.")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help="Per-GPU batch size (images).")
    parser.add_argument('--batch_size_temporal_per_gpu', default=64, type=int, help="Per-GPU batch size (video).")
    parser.add_argument('--epochs', default=30, type=int, help="Number of epochs.")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="Epochs to freeze output layer.")
    parser.add_argument("--lr", default=0.00005, type=float, help="Learning rate.")
    parser.add_argument("--lr_head", default=0.00035, type=float, help="Head learning rate.")
    
    parser.add_argument("--warmup_epochs", default=3, type=int, help="LR warmup epochs.")
    parser.add_argument("--start_epoch_video", default=3, type=int, help="Start epoch for video loss.")
    parser.add_argument("--start_epoch_puzzle", default=6, type=int, help="Start epoch for puzzle loss.")
    
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target minimum LR.")
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'], help="Optimizer.")
    parser.add_argument('--load_from', default=None, help="Path to checkpoint to resume from.")
    parser.add_argument('--drop_path', type=float, default=0.1, help="Drop path rate.")

    # --- Multi-crop ---
    parser.add_argument('--global_crops_number', type=int, default=2, help="Number of global crops.")
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.6, 1.), help="Scale range for global crops.")
    parser.add_argument('--local_crops_number', type=int, default=0, help="Number of local crops.")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4), help="Scale range for local crops.")

    # --- Misc ---
    parser.add_argument('--data_path', default='/path/to/data/', type=str, help='Path to dataset.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs/checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint frequency.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Num workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="Distributed init URL.")
    parser.add_argument("--local_rank", default=0, type=int, help="Local rank (auto-set).")
    parser.add_argument('--no-pe', default=False, type=utils.bool_flag, help="Disable positional encoding.")

    return parser

def train_pl(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ Preparing Data ============
    transform = DataAugmentation(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    
    # 1. Image Dataset (Puzzle)
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    dataset_image = ImageFolderMask_puzzle(
        args.data_path, 
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1/0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch
    )
    print(f"Data loaded: there are {len(dataset_image)} images.")
    
    # 2. Temporal Dataset (Video)
    dataset_temporal = Temporal_RandStep_dataset(lmdb_path=args.data_path, img_size=224)
    print(f"Data loaded: there are {len(dataset_temporal)} video clips.")

    # Balance iterations
    N_img,  B_img = len(dataset_image),  args.batch_size_per_gpu
    N_vid0, B_vid = len(dataset_temporal), args.batch_size_temporal_per_gpu
    iter_img = max(1, N_img  // B_img)
    iter_vid = max(1, N_vid0 // B_vid)
    repeat_factor = math.ceil(iter_img / iter_vid)
    
    dataset_temporal = RepeatDataset(dataset_temporal, repeat_factor)
    
    # Samplers & Loaders
    sampler_image = torch.utils.data.DistributedSampler(dataset_image, shuffle=True)
    sampler_temporal = torch.utils.data.DistributedSampler(dataset_temporal, shuffle=True)
    
    data_loader_image = torch.utils.data.DataLoader(
        dataset_image, sampler=sampler_image,
        batch_size=args.batch_size_per_gpu, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )
    data_loader_temporal = torch.utils.data.DataLoader(
        dataset_temporal, sampler=sampler_temporal,
        batch_size=args.batch_size_temporal_per_gpu, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )

    # ============ Building Networks ============
    args.arch = args.arch.replace("deit", "vit")
    
    # Backbone
    if 'swin' in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size, return_all_tokens=True, masked_im_modeling=args.use_masked_im_modeling)
        teacher = models.__dict__[args.arch](
            window_size=args.window_size, drop_path_rate=0.0, return_all_tokens=True)
        embed_dim = student.num_features
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size, drop_path_rate=args.drop_path,
            return_all_tokens=True, masked_im_modeling=args.use_masked_im_modeling)
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size, return_all_tokens=True)
        embed_dim = student.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    # Heads & Wrappers
    student = utils.MultiCropWrapper(student, iBOTHead(
        embed_dim, args.out_dim, patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head, act=args.act_in_head, norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))
    teacher = utils.MultiCropWrapper(teacher, iBOTHead(
        embed_dim, args.out_dim, patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head, act=args.act_in_head, shared_head=args.shared_head_teacher,
    ))

    # Auxiliary Modules
    backbone_dim = student.backbone.embed_dim if hasattr(student.backbone,'embed_dim') else student.backbone.num_features
    puzzle_decoder = PuzzleDecoder(attn_mul=4, num_blocks=1, embed_dim=backbone_dim, num_heads=12 if backbone_dim==768 else 8).cuda()
    puzzle_decoder = nn.parallel.DistributedDataParallel(puzzle_decoder, device_ids=[args.gpu], broadcast_buffers=False)

    temporal_head = TemporalHead(backbone_dim=backbone_dim).cuda()
    temporal_head = nn.parallel.DistributedDataParallel(temporal_head, device_ids=[args.gpu], broadcast_buffers=False)
    
    # Move to GPU & Sync
    student, teacher = student.cuda(), teacher.cuda()
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False)
    teacher_without_ddp = teacher.module
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False)
    
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
        
    print(f"Student and Teacher are built: {args.arch}.")

    # ============ Loss & Optimizers ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = utils.iBOTLoss(
        args.out_dim, args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number, args.local_crops_number,
        args.warmup_teacher_temp, args.teacher_temp,
        args.warmup_teacher_patch_temp, args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs, args.epochs,
        lambda1=args.lambda1, lambda2=args.lambda2, mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    pl_loss_fn = utils.PlackettLuceLoss(sample=False).cuda()

    writer = None
    if utils.is_main_process():
        local_runs = os.path.join(args.output_dir, 'tf_logs')
        writer = SummaryWriter(logdir=local_runs)
        
    # Optimizer Groups
    params_groups = utils.get_params_groups(student)
    params_groups.extend(utils.get_params_groups(puzzle_decoder))
    params_groups.extend(utils.get_params_groups(temporal_head))
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # Schedulers
    total_batch_size = args.batch_size_per_gpu * utils.get_world_size()
    lr_schedule = utils.cosine_scheduler(
        args.lr * total_batch_size / 256., args.min_lr,
        args.epochs, len(data_loader_image), warmup_epochs=args.warmup_epochs,
    )
    lr_schedule_head = utils.cosine_scheduler(
        args.lr_head * total_batch_size / 256., args.min_lr * args.lr_head / args.lr,
        args.epochs, len(data_loader_image), warmup_epochs=args.warmup_epochs
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end,
        args.epochs, len(data_loader_image),
    )
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader_image)
    )

    # ============ Resume Training ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student, teacher=teacher,
            temporal_head=temporal_head, puzzle_decoder=puzzle_decoder,
            optimizer=optimizer, fp16_scaler=fp16_scaler, ibot_loss=ibot_loss,
        )
    start_epoch = to_restore["epoch"]

    print("Starting Training!")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        data_loader_image.sampler.set_epoch(epoch)
        data_loader_image.dataset.set_epoch(epoch)
        dataset_temporal.dataset.set_epoch(epoch)
        data_loader_temporal.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            student, teacher, teacher_without_ddp, ibot_loss, temporal_head, puzzle_decoder, pl_loss_fn,
            data_loader_image, data_loader_temporal, optimizer,
            lr_schedule, lr_schedule_head, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args
        )

        # Save Checkpoints
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            "temporal_head": temporal_head.state_dict(),
            'puzzle_decoder': puzzle_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            
        # Logging
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            for k, v in train_stats.items():
                writer.add_scalar(k, v, epoch)
        
    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, temporal_head, puzzle_decoder, pl_loss_fn, 
                    data_loader_image, data_loader_temporal, optimizer, 
                    lr_schedule, lr_schedule_head, wd_schedule, momentum_schedule, 
                    epoch, fp16_scaler, args):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    iter_temporal = iter(data_loader_temporal)
    
    # Common parameters for EMA
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q); params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k); params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    for it_curr, (images, masks, puzzle_images) in enumerate(metric_logger.log_every(data_loader_image, 100, header)):
        try:
            clips = next(iter_temporal)
        except StopIteration:
            iter_temporal = iter(data_loader_temporal)
            clips = next(iter_temporal)

        it = len(data_loader_image) * epoch + it_curr
        
        # Update LR & WD
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it] if i < 4 else lr_schedule_head[it]
            if i in (0, 2, 4): 
                param_group["weight_decay"] = wd_schedule[it]

        # Move to GPU
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]   
        clips = clips.cuda(non_blocking=True)
        
        # Puzzle branch inputs
        current_image = puzzle_images[0].cuda(non_blocking=True)
        past_image = puzzle_images[1].cuda(non_blocking=True)
        future_image = puzzle_images[2].cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # --- iBOT Forward ---
            teacher_output = teacher(images[:args.global_crops_number])
            student_output = student(images[:args.global_crops_number], mask=masks[:args.global_crops_number])  

            # --- Puzzle Forward ---
            # Get current image patch embeddings 
            current_patch = student.module.backbone(current_image, mask=masks[2+args.local_crops_number], no_pe=True)[:,1:,:]
        
            student.module.backbone.masked_im_modeling = False            
            # Local views
            student_local_cls = student(images[args.global_crops_number:])[0] if len(images) > args.global_crops_number else None

            # Neighbor image patch embeddings       
            with torch.no_grad():
                past_neighbor_patch = student.module.backbone(past_image, no_pe=True)[:,1:,:]   
                future_neighbor_patch = student.module.backbone(future_image, no_pe=True)[:,1:,:]   
                neighbor_patch = torch.cat((past_neighbor_patch, future_neighbor_patch), dim=1)
                
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            # Calculate Puzzle Loss
            puzzle_scores = puzzle_decoder(current_patch, neighbor_patch)
            loss_puzzle = pl_loss_fn(puzzle_scores)
            loss_puzzle = cosine_lambda(args.lambda_puzzle, args.start_epoch_puzzle, epoch, it_curr, len(data_loader_image)) * loss_puzzle
            
            # Calculate iBOT Loss
            loss_dict_ibot = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
            loss_ibot = loss_dict_ibot.pop('loss')

            # --- Temporal Branch Forward ---
            B, T, C, H, W = clips.shape
            flat = clips.view(B*T, C, H, W)
            student.module.backbone.masked_im_modeling = False
            feat = student.module.backbone(flat)[:, 0, :].view(B, T, -1)
            student.module.backbone.masked_im_modeling = True
            
            logits = temporal_head(feat).squeeze(-1)
            loss_temporal = pl_loss_fn(logits.float())
            # Note: Temporal schedule was commented out in original. Uncomment if needed:
            # loss_temporal = cosine_lambda(args.lambda_video, args.start_epoch_video, epoch, it_curr, len(data_loader_image)) * loss_temporal
            
            # Total Loss
            loss = loss_ibot + loss_puzzle + loss_temporal

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # Logging acc
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1]) 
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)

        # Optimization
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in loss_dict_ibot.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(loss_puzzle=loss_puzzle.item())
        metric_logger.update(loss_temporal=loss_temporal.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def cosine_lambda(lambda_v, start_epoch, epoch, it, steps_per_epoch):
    if epoch < start_epoch:
        return 0.0
    prog = min(1.0, (epoch - start_epoch + it / steps_per_epoch) / 1.0)
    return lambda_v * (0.5 - 0.5 * math.cos(math.pi * prog))

class DataAugmentation(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.puzzle_transf1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.3
            ),
            transforms.RandomGrayscale(p=0.2),            
            normalize,
        ])

        self.puzzle_transf2 = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.3
            ),
            transforms.RandomGrayscale(p=0.2),            
            normalize,
        ])

        self.global_crops_number = global_crops_number
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image, time_frame=None):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        
        if time_frame:
            puzzles = [self.puzzle_transf1(image), self.puzzle_transf2(time_frame[0]), self.puzzle_transf2(time_frame[1])]

        return (crops, puzzles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('pl_stitch', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_pl(args)