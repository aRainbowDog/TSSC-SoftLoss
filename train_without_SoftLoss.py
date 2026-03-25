# 10个epoch保存,resume form 20epoch
import os
# 临时开启expandable_segments（加载阶段用，训练时再关闭）
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import save_image
from tqdm import tqdm
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models.model_dit import MVIF_models
from models.diffusion.gaussian_diffusion import create_diffusion
from dataloader.data_loader_acdc import data_loader
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.utils import (
    clip_grad_norm_, create_logger, update_ema, requires_grad, 
    cleanup, create_tensorboard, write_tensorboard, setup_distributed
)
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from contextlib import nullcontext
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 全局工具函数 --------------------------
def get_raw_model(model):
    """获取DDP包装后的原始模型，若无DDP则返回原模型"""
    return model.module if hasattr(model, 'module') else model

def set_optimizer_zeros_grad(optimizer, set_to_none=True):
    """安全的optimizer梯度清零，减少显存碎片"""
    if set_to_none:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad()

def safe_load_checkpoint(ckpt_path, device):
    """
    安全加载checkpoint：只加载model/ema，跳过opt，避免OOM
    """
    # 1. 加载到CPU，避免GPU瞬间占用
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # 2. 只保留必要的key，删除opt和冗余数据
    keep_keys = ["model", "ema", "train_steps", "best_val_metric"]
    for k in list(checkpoint.keys()):
        if k not in keep_keys:
            del checkpoint[k]
    # 3. model/ema参数移到GPU（逐个加载）
    for k in ["model", "ema"]:
        if k in checkpoint and checkpoint[k] is not None:
            for param_key in checkpoint[k]:
                if isinstance(checkpoint[k][param_key], torch.Tensor):
                    checkpoint[k][param_key] = checkpoint[k][param_key].to(device, non_blocking=True)
    return checkpoint

# -------------------------- 主训练函数 --------------------------
def main(args):
    # ========== 新增：纯验证模式 ==========
    if getattr(args, 'eval_only', False):
        run_validation_only(args)
        return
    # 1. 基础校验与分布式初始化
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    setup_distributed(backend="nccl")  # 初始化分布式环境
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)  # 绑定当前进程到local_rank对应的GPU

    # 2. 显存优化配置（加载阶段开启expandable_segments，训练时关闭）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # 关闭benchmark减少显存预分配
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # 3. 随机种子
    seed = args.global_seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 4. 实验目录（仅rank0创建）
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{model_string_name}_{args.cur_date}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    val_fold = f"{experiment_dir}/val_pic"
    
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(val_fold, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(os.path.join(experiment_dir, 'runs'))
        OmegaConf.save(args, os.path.join(experiment_dir, 'config_acdc.yaml'))
        logger.info(f"Experiment dir: {experiment_dir}")
        logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # 5. 模型初始化（先CPU创建，再移到GPU）
    assert args.image_size % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.image_size // 8
    additional_kwargs = {'num_frames': args.tar_num_frames, 'mode': 'video'}
    
    # 先在CPU创建模型，减少GPU显存峰值
    model = MVIF_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **additional_kwargs
    )
    # 移到GPU（非阻塞）
    model = model.to(device, non_blocking=True)

    # EMA模型先放CPU（关键！节省8GiB显存）
    ema = deepcopy(model).cpu()
    requires_grad(ema, False)

    # VAE模型（冻结）
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_path, 
        subfolder="sd-vae-ft-mse"
    ).to(device, non_blocking=True)
    vae.requires_grad_(False)
    vae.eval()

    vae.enable_slicing()  # 强制VAE分批处理帧，大幅降低视频解码时的显存峰值
    # 6. 加载预训练权重（仅rank0打印日志）
    # if args.pretrained and os.path.exists(args.pretrained):
    #     if rank == 0:
    #         logger.info(f"Loading pretrained model from {args.pretrained}")
    #     # 优化：预训练权重也先加载到CPU
    #     checkpoint = torch.load(args.pretrained, map_location="cpu")
    #     checkpoint = checkpoint.get("ema", checkpoint)  # 优先加载EMA权重
        
    #     model_dict = model.state_dict()
    #     pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    #     # 逐个参数移到GPU，避免一次性加载
    #     for k in pretrained_dict:
    #         pretrained_dict[k] = pretrained_dict[k].to(device, non_blocking=True)
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)
        
    #     # 清理临时数据
    #     del checkpoint, pretrained_dict
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()  # 强制回收未使用显存
        
    #     if rank == 0:
    #         load_ratio = len(pretrained_dict) / len(model_dict) * 100
    #         logger.info(f"Loaded {load_ratio:.1f}% pretrained weights")
    # 6. 加载预训练权重（彻底修复UnboundLocalError）
    if args.pretrained and os.path.exists(args.pretrained):
        if rank == 0:
            logger.info(f"Loading pretrained model from {args.pretrained}")
        try:
            # 优化：预训练权重也先加载到CPU
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            # 优先加载EMA权重，若无则加载model权重
            model_weights = checkpoint.get("ema", checkpoint.get("model", {}))
            
            # 初始化pretrained_dict，避免未定义
            pretrained_dict = {}
            if model_weights:
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in model_weights.items() if k in model_dict}
                
                # 逐个参数移到GPU，避免一次性加载
                for k in pretrained_dict:
                    pretrained_dict[k] = pretrained_dict[k].to(device, non_blocking=True)
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            
            # 计算加载比例（确保load_ratio始终有值）
            load_ratio = len(pretrained_dict) / len(model.state_dict()) * 100 if pretrained_dict else 0.0
            
            # 清理临时数据
            del checkpoint, model_weights
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # 强制回收未使用显存
            
            if rank == 0:
                logger.info(f"Loaded {load_ratio:.1f}% pretrained weights")
        except Exception as e:
            if rank == 0:
                logger.error(f"Failed to load pretrained weights: {e}")
            load_ratio = 0.0
    else:
        # 未指定预训练权重时，加载比例为0
        load_ratio = 0.0

    # 7. 模型优化配置（延迟编译到续训后）
    if args.gradient_checkpointing:
        if rank == 0:
            logger.info("Enabling gradient checkpointing")
        def enable_ckpt(module):
            if hasattr(module, 'enable_gradient_checkpointing'):
                module.enable_gradient_checkpointing()
            elif hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
        model.apply(enable_ckpt)

    if args.enable_xformers_memory_efficient_attention:
        if rank == 0:
            logger.info("Using Xformers memory-efficient attention")
        model.enable_xformers_memory_efficient_attention()

    # 8. DDP包装（多卡时）
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=True,
            bucket_cap_mb=10
        )
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")


    # 9. 优化器与学习率调度器
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps
    )

    # 10. 数据加载器
    # 训练集
    dataset_train = data_loader(args, stage='train')
    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=False,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2  # 预取优化
    )

    # 验证集
    dataset_val = data_loader(args, stage='val')
    sampler_val = DistributedSampler(
        dataset_val,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2
    )

    if rank == 0:
        logger.info(f"Train dataset size: {len(dataset_train)}")
        logger.info(f"Val dataset size: {len(dataset_val)}")
        logger.info(f"Train steps per epoch: {math.ceil(len(loader_train))}")

    # ========== 断点续训核心优化（终极版） ==========
    # 强制清理所有冗余显存
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    train_steps = 0
    first_epoch = 0
    resume_step = 0
    # 初始化最优指标（断点续训时加载历史最优值）
    best_val_metric = float('inf')  # 越小越好

    if args.resume_from_checkpoint:
        latest_ckpt = os.path.join(checkpoint_dir, "epoch_020.pth")
        if os.path.exists(latest_ckpt):
            if rank == 0:
                logger.info(f"Resuming from checkpoint (memory optimized): {latest_ckpt}")
                # 打印当前显存状态
                logger.info(f"Memory before load: {torch.cuda.memory_allocated(device)/1024**3:.2f} GiB")
            
            # 1. 安全加载checkpoint（只加载model/ema，跳过opt）
            checkpoint = safe_load_checkpoint(latest_ckpt, device)
            
            # 2. 加载model（唯一的GPU显存占用）
            get_raw_model(model).load_state_dict(checkpoint["model"])
            # 强制清理显存
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # 3. EMA先放CPU，训练时再移GPU
            ema.load_state_dict(checkpoint["ema"])
            
            # 4. 读取元数据
            # train_steps = checkpoint["train_steps"]
            # first_epoch = train_steps // len(loader_train)
            # resume_step = train_steps % len(loader_train)
            steps_per_epoch = len(loader_train) 
            train_steps = 20 * steps_per_epoch 
            
            first_epoch = 20  # 强制从 20 开始
            resume_step = 0   # 从该 epoch 的第 0 步开始
            
            # 加载历史最优指标
            if "best_val_metric" in checkpoint and checkpoint["best_val_metric"] != float('inf'):
                best_val_metric = checkpoint["best_val_metric"]
                if rank == 0:
                    logger.info(f"Resumed best validation metric: {best_val_metric:.4f}")
            else:
                best_val_metric = float('inf')
            
            # 5. 强制清理所有临时数据
            del checkpoint
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            if rank == 0:
                logger.info(f"Resumed epoch: {first_epoch}, step: {resume_step}")
                logger.info(f"Memory after load: {torch.cuda.memory_allocated(device)/1024**3:.2f} GiB")
                logger.warning("跳过了优化器状态加载，优化器将从头开始（学习率不变）")
        else:
            if rank == 0:
                logger.warning(f"Checkpoint {latest_ckpt} not found, start from scratch")

    # 续训完成后：EMA移到GPU + 关闭expandable_segments + 模型编译
    # 1. EMA移到GPU
    ema = ema.to(device, non_blocking=True)
    update_ema(ema, get_raw_model(model), decay=0)  # 初始化EMA权重
    # 2. 关闭expandable_segments（训练阶段用）
    torch.backends.cuda.expandable_segments = False
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    # 3. 延迟编译模型（避免加载阶段占用显存）
    if args.use_compile and torch.cuda.is_available():
        if rank == 0:
            logger.info("Compiling model with torch.compile (delayed)")
        model = torch.compile(model, mode="reduce-overhead")
    # 4. 最后一次显存清理
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # 12. 训练循环
    num_update_steps_per_epoch = math.ceil(len(loader_train))
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    model.train()
    ema.eval()
    running_loss = 0.0
    log_steps = 0
    start_time = time()

    if rank == 0:
        logger.info(f"Start training for {num_train_epochs} epochs (first epoch: {first_epoch})")

    for epoch in range(first_epoch, num_train_epochs):
        sampler_train.set_epoch(epoch)  # 分布式采样器epoch同步
        diffusion = create_diffusion(
            timestep_respacing=args.timestep_respacing_train,
            diffusion_steps=args.diffusion_steps
        )
        diffusion.training = True

        pbar = tqdm(loader_train, total=len(loader_train), desc=f"Rank{rank} Epoch {epoch+1}", disable=(rank != 0))
        for step, batch in enumerate(pbar):
            # 跳过断点续训的步骤
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                train_steps += 1
                pbar.update(1)
                continue

            # 数据加载
            video = batch['video'].to(device, non_blocking=True)  # [B, F, C, H, W]
            b, f, c, h, w = video.shape

            # VAE编码（latent空间）
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.mixed_precision):
                video_flat = rearrange(video, 'b f c h w -> (b f) c h w')
                latent = vae.encode(video_flat).latent_dist.sample()
                latent.mul_(0.18215)  # 归一化
                latent = rearrange(latent, '(b f) c h w -> b f c h w', b=b)

            # 构建mask（第一帧和最后一帧不mask）
            b, f, c, h_latent, w_latent = latent.shape  # latent维度：[b, f, c, 32, 32]
            mask = torch.ones(b, f, h_latent, w_latent, device=device)
            mask[:, 0, :, :] = 0  # 第一帧
            mask[:, -1, :, :] = 0  # 最后一帧

            # 随机时间步
            t = torch.randint(0, diffusion.num_timesteps, (b,), device=device)

            # 计算损失
            # --- 修改部分：引入 no_sync 优化 ---
            my_context = model.no_sync() if (step + 1) % args.gradient_accumulation_steps != 0 and world_size > 1 else nullcontext()
            with my_context:
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    loss = diffusion.training_losses(model, latent, t, mask=mask)["loss"].mean() / args.gradient_accumulation_steps
                if args.mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            # with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            #     loss_dict = diffusion.training_losses(model, latent, t, mask=mask)
            #     loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps

            # # 反向传播
            # if args.mixed_precision:
            #     scaler.scale(loss).backward()
            # else:
            #     loss.backward()

            # 梯度裁剪（累计到指定步数）
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.mixed_precision:
                    scaler.unscale_(opt)
                grad_norm = clip_grad_norm_(
                    get_raw_model(model).parameters(),
                    args.clip_max_norm,
                    clip_grad=(train_steps >= args.start_clip_iter)
                )

                # 优化器步进
                if args.mixed_precision:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                lr_scheduler.step()
                set_optimizer_zeros_grad(opt)
                update_ema(ema, get_raw_model(model))

                # 日志记录
                running_loss += loss.item() * args.gradient_accumulation_steps
                log_steps += 1
                train_steps += 1

                # 打印进度
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "grad_norm": f"{grad_norm:.4f}",
                    "lr": f"{opt.param_groups[0]['lr']:.6f}"
                })

                # 定期日志
                if train_steps % args.log_every == 0 and rank == 0:
                    avg_loss = running_loss / log_steps
                    steps_per_sec = log_steps / (time() - start_time)
                    logger.info(
                        f"Step {train_steps:07d} | Loss: {avg_loss:.4f} | "
                        f"Grad Norm: {grad_norm:.4f} | Steps/Sec: {steps_per_sec:.2f}"
                    )
                    write_tensorboard(tb_writer, 'Train/Loss', avg_loss, train_steps)
                    write_tensorboard(tb_writer, 'Train/GradNorm', grad_norm, train_steps)
                    write_tensorboard(tb_writer, 'Train/LR', opt.param_groups[0]['lr'], train_steps)
                    running_loss = 0.0
                    log_steps = 0
                    start_time = time()

            # 终止条件
            if train_steps >= args.max_train_steps:
                break

        # ========== 每20个epoch保存模型 ==========
        if rank == 0 and (epoch + 1) % 10 == 0:
            checkpoint = {
                "model": get_raw_model(model).state_dict(),
                "ema": ema.state_dict(),
                "train_steps": train_steps,
                "epoch": epoch + 1,
                "best_val_metric": best_val_metric
            }
            ckpt_path = f"{checkpoint_dir}/epoch_{epoch+1:03d}.pth"
            torch.save(checkpoint, ckpt_path)
            # 保存后清理
            del checkpoint
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info(f"Saved epoch checkpoint: {ckpt_path}")

        # 保存最新epoch检查点（仅保存model/ema）
        if rank == 0:
            checkpoint = {
                "model": get_raw_model(model).state_dict(),
                "ema": ema.state_dict(),
                "train_steps": train_steps,
                "best_val_metric": best_val_metric
            }
            torch.save(checkpoint, f"{checkpoint_dir}/latest_epoch_train_model.pth")
            # 保存后立即清理
            del checkpoint
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        

        # --- 强制清理显存逻辑开始 ---
        # 1. 在所有 Rank 上同步，确保所有卡都完成了本 Epoch 的最后一步训练
        if world_size > 1:
            dist.barrier() 
        
        # 2. 所有卡同时清理缓存
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # --- 强制清理显存逻辑结束 ---
        
        # -------------------------- 验证阶段 --------------------------
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            vae.eval()
            val_diffusion = create_diffusion(
                timestep_respacing=args.timestep_respacing_test,
                diffusion_steps=args.diffusion_steps
            )
        
            # ========== 1. 初始化本地指标（每张卡独立计算） ==========
            local_mae, local_mse, local_psnr = 0.0, 0.0, 0.0
            val_pbar = tqdm(loader_val, total=len(loader_val), desc=f"Rank{rank} Validation", disable=(rank != 0))
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    for val_step, val_batch in enumerate(val_pbar):
                        video_val = val_batch['video'].to(device, non_blocking=True)
                        b_val, f_val, c_val, h_val, w_val = video_val.shape
        
                        # VAE编码
                        video_val_flat = rearrange(video_val, 'b f c h w -> (b f) c h w')
                        latent_val = vae.encode(video_val_flat).latent_dist.sample().mul_(0.18215)
                        latent_val = rearrange(latent_val, '(b f) c h w -> b f c h w', b=b_val)
        
                        # 构建mask（latent空间）
                        _, _, _, h_latent_val, w_latent_val = latent_val.shape
                        mask_val = torch.ones(b_val, f_val, h_latent_val, w_latent_val, device=device)
                        mask_val[:, 0, :, :] = 0.0  # 第一帧不mask
                        mask_val[:, -1, :, :] = 0.0  # 最后一帧不mask
        
                        # 采样生成
                        z = torch.randn_like(latent_val.permute(0, 2, 1, 3, 4))  # [b, c, f, h, w]
                        samples = val_diffusion.p_sample_loop(
                            get_raw_model(model).forward,
                            z.shape,
                            z,
                            clip_denoised=False,
                            progress=False,
                            device=device,
                            raw_x=latent_val.permute(0, 2, 1, 3, 4),
                            mask=mask_val
                        )
        
                        # 解码
                        samples = samples.permute(1, 0, 2, 3, 4) * mask_val + latent_val.permute(2, 0, 1, 3, 4) * (1 - mask_val)
                        samples = samples.permute(1, 2, 0, 3, 4)
                        samples_flat = rearrange(samples, 'b f c h w -> (b f) c h w') / 0.18215
                        decoded = vae.decode(samples_flat).sample
                        decoded = rearrange(decoded, '(b f) c h w -> b f c h w', b=b_val)
        
                        # 计算本地指标（当前卡的批次指标）
                        gt_np = video_val.detach().cpu().numpy()
                        pred_np = decoded.detach().cpu().numpy()
                        batch_mae = np.mean(np.abs(pred_np - gt_np))
                        batch_mse = mean_squared_error(gt_np.reshape(-1), pred_np.reshape(-1))
                        batch_psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=2)
                        
                        # 累加本地指标（除以全局验证集长度，而非单卡长度）
                        global_val_len = len(loader_val) * dist.get_world_size()
                        local_mae += batch_mae / global_val_len
                        local_mse += batch_mse / global_val_len
                        local_psnr += batch_psnr / global_val_len
        
                        # ========== 2. 仅Rank 0保存验证图片 ==========
                        if rank == 0 and val_step == 0:
                            # 1. 取前3个样本的mask
                            mask_val_small = mask_val[:3]  # [3, f, 32, 32]
                            
                            # 2. 调整维度为[N, C, D, H, W]（适配5维interpolate）
                            mask_val_5d = mask_val_small.unsqueeze(1)  # [3, 1, f, 32, 32]
                            
                            # 3. 上采样：指定3维输出尺寸（帧维度不变，H/W上采样到原始尺寸）
                            mask_val_upsampled = F.interpolate(
                                mask_val_5d,
                                size=(f_val, h_val, w_val),  # 关键：指定3维尺寸（D, H, W）
                                mode='nearest'
                            ).squeeze(1)  # [3, f, 256, 256]
                            
                            # 4. 扩展通道维度（匹配视频的3通道）
                            mask_val_3c = mask_val_upsampled.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # [3, f, 3, 256, 256]
                            
                            # 5. 拼接图片
                            val_pic = torch.cat([
                                video_val[:3],  # 原始视频
                                video_val[:3] * (1 - mask_val_3c),  # mask后的视频
                                decoded[:3]  # 生成的视频
                            ], dim=1)
                            
                            # 6. 调整维度并保存（排版）
                            val_pic_flat = rearrange(val_pic, 'b f c h w -> (b f) c h w')
                            save_image(
                                val_pic_flat,
                                os.path.join(val_fold, f"Epoch_{epoch+1}.png"),
                                nrow=12,
                                normalize=True,
                                value_range=(-1, 1)
                            )
        
            # ========== 3. 分布式指标汇总（核心：all_reduce合并所有卡的指标） ==========
            # 将本地指标转为tensor，用于NCCL通信
            mae_tensor = torch.tensor(local_mae, device=device)
            mse_tensor = torch.tensor(local_mse, device=device)
            psnr_tensor = torch.tensor(local_psnr, device=device)
            
            # 汇总所有卡的指标（SUM模式：所有卡的指标相加）
            dist.all_reduce(mae_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)
            
            # 转为全局平均指标
            global_mae = mae_tensor.item()
            global_mse = mse_tensor.item()
            global_psnr = psnr_tensor.item()
        
            # ========== 4. 仅Rank 0打印全局指标并更新最优值 ==========
            if rank == 0:
                val_metric = (global_mae + global_mse) / 2
                
                # 记录验证指标
                logger.info(
                    f"Epoch {epoch+1} Validation | Global MAE: {global_mae:.4f} | Global MSE: {global_mse:.4f} | "
                    f"Global PSNR: {global_psnr:.4f} | Metric: {val_metric:.4f} | Best Metric: {best_val_metric:.4f}"
                )
                write_tensorboard(tb_writer, 'Val/MAE', global_mae, epoch+1)
                write_tensorboard(tb_writer, 'Val/MSE', global_mse, epoch+1)
                write_tensorboard(tb_writer, 'Val/PSNR', global_psnr, epoch+1)
                write_tensorboard(tb_writer, 'Val/Best_Metric', best_val_metric, epoch+1)
        
                # 更新最优指标并保存最优模型
                if val_metric < best_val_metric:
                    prev_best = best_val_metric
                    best_val_metric = val_metric
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    logger.info(
                        f"New best metric: {val_metric:.4f} (previous: {prev_best:.4f}) | "
                        f"Improvement: {prev_best - val_metric:.4f}"
                    )
                    # 保存最优模型
                    checkpoint = {
                        "model": get_raw_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "val_metric": val_metric,
                        "epoch": epoch+1,
                        "train_steps": train_steps,
                        "best_val_metric": best_val_metric
                    }
                    torch.save(checkpoint, f"{checkpoint_dir}/best_epoch_train_model.pth")
                    del checkpoint  # 清理显存
        
            # 所有卡同步，确保Rank 0完成指标打印/保存后，其他卡再继续训练
            dist.barrier()
        model.train()  # 回到训练模式
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            vae.eval()
            val_diffusion = create_diffusion(
                timestep_respacing=args.timestep_respacing_test,
                diffusion_steps=args.diffusion_steps
            )

            val_mae, val_mse, val_psnr = 0.0, 0.0, 0.0
            val_pbar = tqdm(loader_val, total=len(loader_val), desc="Validation")
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    for val_step, val_batch in enumerate(val_pbar):
                        video_val = val_batch['video'].to(device)
                        b_val, f_val, c_val, h_val, w_val = video_val.shape
    
                        # VAE编码
                        video_val_flat = rearrange(video_val, 'b f c h w -> (b f) c h w')
                        latent_val = vae.encode(video_val_flat).latent_dist.sample().mul_(0.18215)
                        latent_val = rearrange(latent_val, '(b f) c h w -> b f c h w', b=b_val)
    
                        # 构建mask（latent空间）
                        _, _, _, h_latent_val, w_latent_val = latent_val.shape
                        mask_val = torch.ones(b_val, f_val, h_latent_val, w_latent_val, device=device)
                        mask_val[:, 0, :, :] = 0.0  # 第一帧不mask
                        mask_val[:, -1, :, :] = 0.0  # 最后一帧不mask
    
                        # 采样生成
                        z = torch.randn_like(latent_val.permute(0, 2, 1, 3, 4))  # [b, c, f, h, w]
                        samples = val_diffusion.p_sample_loop(
                            get_raw_model(model).forward,
                            z.shape,
                            z,
                            clip_denoised=False,
                            progress=False,
                            device=device,
                            raw_x=latent_val.permute(0, 2, 1, 3, 4),
                            mask=mask_val
                        )
    
                        # 解码
                        samples = samples.permute(1, 0, 2, 3, 4) * mask_val + latent_val.permute(2, 0, 1, 3, 4) * (1 - mask_val)
                        samples = samples.permute(1, 2, 0, 3, 4)
                        samples_flat = rearrange(samples, 'b f c h w -> (b f) c h w') / 0.18215
                        decoded = vae.decode(samples_flat).sample
                        decoded = rearrange(decoded, '(b f) c h w -> b f c h w', b=b_val)
    
                        # 计算指标
                        gt_np = video_val.detach().cpu().numpy()
                        pred_np = decoded.detach().cpu().numpy()
                        val_mae += np.mean(np.abs(pred_np - gt_np)) / len(loader_val)
                        val_mse += mean_squared_error(gt_np.reshape(-1), pred_np.reshape(-1)) / len(loader_val)
                        val_psnr += peak_signal_noise_ratio(gt_np, pred_np, data_range=2) / len(loader_val)
    
                        # 保存验证图片
                        if val_step == 0:
                            # 1. 取前3个样本的mask
                            mask_val_small = mask_val[:3]  # [3, f, 32, 32]
                            
                            # 2. 调整维度为[N, C, D, H, W]（适配5维interpolate）
                            mask_val_5d = mask_val_small.unsqueeze(1)  # [3, 1, f, 32, 32]
                            
                            # 3. 上采样：指定3维输出尺寸（帧维度不变，H/W上采样到原始尺寸）
                            mask_val_upsampled = F.interpolate(
                                mask_val_5d,
                                size=(f_val, h_val, w_val),  # 关键：指定3维尺寸（D, H, W）
                                mode='nearest'
                            ).squeeze(1)  # [3, f, 256, 256]
                            
                            # 4. 扩展通道维度（匹配视频的3通道）
                            mask_val_3c = mask_val_upsampled.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # [3, f, 3, 256, 256]
                            
                            # 5. 拼接图片
                            val_pic = torch.cat([
                                video_val[:3],  # 原始视频
                                video_val[:3] * (1 - mask_val_3c),  # mask后的视频
                                decoded[:3]  # 生成的视频
                            ], dim=1)
                            
                            # 6. 调整维度并保存（排版）
                            val_pic_flat = rearrange(val_pic, 'b f c h w -> (b f) c h w')
                            save_image(
                                val_pic_flat,
                                os.path.join(val_fold, f"Epoch_{epoch+1}.png"),
                                nrow=12,
                                normalize=True,
                                value_range=(-1, 1)
                            )

            # 记录验证指标
            val_metric = (val_mae + val_mse) / 2
            logger.info(
                f"Epoch {epoch+1} Validation | MAE: {val_mae:.4f} | MSE: {val_mse:.4f} | "
                f"PSNR: {val_psnr:.4f} | Metric: {val_metric:.4f} | Best Metric: {best_val_metric:.4f}"
            )
            write_tensorboard(tb_writer, 'Val/MAE', val_mae, epoch+1)
            write_tensorboard(tb_writer, 'Val/MSE', val_mse, epoch+1)
            write_tensorboard(tb_writer, 'Val/PSNR', val_psnr, epoch+1)
            write_tensorboard(tb_writer, 'Val/Best_Metric', best_val_metric, epoch+1)

            if val_metric < best_val_metric:
                prev_best = best_val_metric
                best_val_metric = val_metric
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info(
                    f"New best metric: {val_metric:.4f} (previous: {prev_best:.4f}) | "
                    f"Improvement: {prev_best - val_metric:.4f}"
                )

            dist.barrier() # 确保每一轮 Epoch 结束时所有卡步调一致
        model.train()  # 回到训练模式

        
        
        dist.barrier() # 确保每一轮 Epoch 结束时所有卡步调一致
        # 终止训练
        if train_steps >= args.max_train_steps:
            break

    # 训练结束
    if rank == 0:
        logger.info(f"Training finished! Total steps: {train_steps}")
    cleanup()
def run_validation_only(args):
    """纯验证模式：加载模型，运行一轮验证，保存结果"""
    # 1. 分布式初始化（与训练相同）
    assert torch.cuda.is_available()
    setup_distributed(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # 2. 创建实验目录
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{model_string_name}_{args.cur_date}_eval"
    val_fold = f"{experiment_dir}/val_pic"
    
    if rank == 0:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(val_fold, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(os.path.join(experiment_dir, 'runs'))
        logger.info(f"EVAL ONLY MODE | Experiment dir: {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # 3. 初始化模型（与训练相同）
    latent_size = args.image_size // 8
    additional_kwargs = {'num_frames': args.tar_num_frames, 'mode': 'video'}
    
    model = MVIF_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **additional_kwargs
    ).to(device, non_blocking=True)
    
    # EMA模型（验证必须用EMA）
    ema = deepcopy(model).cpu()
    
    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_path, 
        subfolder="sd-vae-ft-mse"
    ).to(device, non_blocking=True)
    vae.requires_grad_(False)
    vae.eval()
    vae.enable_slicing()

    # 4. 加载预训练权重（优先EMA）
    ckpt_path = args.pretrained if args.pretrained else os.path.join(
        f"{args.results_dir}/{model_string_name}_{args.cur_date}/checkpoints", 
        args.test_ckpt
    )
    
    if rank == 0:
        logger.info(f"Loading checkpoint: {ckpt_path}")
    
    checkpoint = safe_load_checkpoint(ckpt_path, device)
    
    # 加载EMA权重到模型（验证时用EMA）
    ema_state = checkpoint.get("ema", checkpoint.get("model", {}))
    model.load_state_dict(ema_state)
    ema.load_state_dict(ema_state)
    ema = ema.to(device, non_blocking=True)
    
    del checkpoint
    torch.cuda.empty_cache()

    # 5. DDP包装（多卡时）
    world_size = dist.get_world_size()
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 6. 创建验证数据加载器
    dataset_val = data_loader(args, stage='val')
    sampler_val = DistributedSampler(
        dataset_val, num_replicas=world_size, rank=rank, shuffle=False
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False  # 验证时不drop，确保完整评估
    )

    if rank == 0:
        logger.info(f"Val dataset size: {len(dataset_val)}")

    # 7. 运行验证（复用原有验证逻辑）
    model.eval()
    vae.eval()
    
    val_diffusion = create_diffusion(
        timestep_respacing=args.timestep_respacing_test,
        diffusion_steps=args.diffusion_steps
    )

    # 初始化指标
    local_mae, local_mse, local_psnr = 0.0, 0.0, 0.0
    global_val_len = len(loader_val) * world_size
    
    val_pbar = tqdm(loader_val, total=len(loader_val), 
                    desc=f"Rank{rank} Validation", disable=(rank != 0))

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            for val_step, val_batch in enumerate(val_pbar):
                video_val = val_batch['video'].to(device, non_blocking=True)
                b_val, f_val, c_val, h_val, w_val = video_val.shape

                # VAE编码
                video_val_flat = rearrange(video_val, 'b f c h w -> (b f) c h w')
                latent_val = vae.encode(video_val_flat).latent_dist.sample().mul_(0.18215)
                latent_val = rearrange(latent_val, '(b f) c h w -> b f c h w', b=b_val)

                # 构建mask
                _, _, _, h_latent_val, w_latent_val = latent_val.shape
                mask_val = torch.ones(b_val, f_val, h_latent_val, w_latent_val, device=device)
                mask_val[:, 0, :, :] = 0.0
                mask_val[:, -1, :, :] = 0.0

                # 采样生成
                z = torch.randn_like(latent_val.permute(0, 2, 1, 3, 4))
                samples = val_diffusion.p_sample_loop(
                    get_raw_model(model).forward,
                    z.shape, z,
                    clip_denoised=False,
                    progress=False,
                    device=device,
                    raw_x=latent_val.permute(0, 2, 1, 3, 4),
                    mask=mask_val
                )

                # 解码
                samples = samples.permute(1, 0, 2, 3, 4) * mask_val + \
                         latent_val.permute(2, 0, 1, 3, 4) * (1 - mask_val)
                samples = samples.permute(1, 2, 0, 3, 4)
                samples_flat = rearrange(samples, 'b f c h w -> (b f) c h w') / 0.18215
                decoded = vae.decode(samples_flat).sample
                decoded = rearrange(decoded, '(b f) c h w -> b f c h w', b=b_val)

                # 计算指标
                gt_np = video_val.detach().cpu().numpy()
                pred_np = decoded.detach().cpu().numpy()
                
                batch_mae = np.mean(np.abs(pred_np - gt_np))
                batch_mse = mean_squared_error(gt_np.reshape(-1), pred_np.reshape(-1))
                batch_psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=2)
                
                local_mae += batch_mae / global_val_len
                local_mse += batch_mse / global_val_len
                local_psnr += batch_psnr / global_val_len

                # 保存可视化图片（仅rank 0，第一个batch）
                if rank == 0 and val_step == 0:
                    mask_val_small = mask_val[:3]
                    mask_val_5d = mask_val_small.unsqueeze(1)
                    mask_val_upsampled = F.interpolate(
                        mask_val_5d,
                        size=(f_val, h_val, w_val),
                        mode='nearest'
                    ).squeeze(1)
                    mask_val_3c = mask_val_upsampled.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                    
                    val_pic = torch.cat([
                        video_val[:3],
                        video_val[:3] * (1 - mask_val_3c),
                        decoded[:3]
                    ], dim=1)
                    
                    val_pic_flat = rearrange(val_pic, 'b f c h w -> (b f) c h w')
                    save_image(
                        val_pic_flat,
                        os.path.join(val_fold, f"Eval_Result.png"),
                        nrow=12, normalize=True, value_range=(-1, 1)
                    )

    # 汇总指标
    mae_tensor = torch.tensor(local_mae, device=device)
    mse_tensor = torch.tensor(local_mse, device=device)
    psnr_tensor = torch.tensor(local_psnr, device=device)
    
    dist.all_reduce(mae_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(mse_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)

    # 打印结果
    if rank == 0:
        logger.info(
            f"EVALUATION COMPLETE | Global MAE: {mae_tensor.item():.4f} | "
            f"Global MSE: {mse_tensor.item():.4f} | "
            f"Global PSNR: {psnr_tensor.item():.4f}"
        )

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_cta.yaml")
    
    args = parser.parse_args()
    main(OmegaConf.load(args.config))