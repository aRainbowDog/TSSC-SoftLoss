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
import warnings
# ========== 新增mask生成相关依赖 ==========
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import apply_hysteresis_threshold
warnings.filterwarnings('ignore')

# -------------------------- 新增：Mask生成函数 --------------------------
def generate_vessel_mask_adaptive(frames_gray, max_weight=100.0, base_weight=1.0):
    """
    生成自适应血管mask和软权重
    Args:
        frames_gray: list of numpy array, 灰度帧列表 [f, h, w]
        max_weight: float, 血管区域最大权重（最高100倍）
        base_weight: float, 非血管区域基础权重
    Returns:
        mask_final: numpy array, 二值mask [h, w]
        soft_weight: numpy array, 软权重图 [h, w]（值范围[base_weight, max_weight]）
    """
    first_frame = frames_gray[0].astype(np.float32)
    last_frame = frames_gray[-1].astype(np.float32)
    
    # 计算帧差
    diff_map = np.abs(first_frame - last_frame)
    diff_smooth = cv2.GaussianBlur(diff_map, (3, 3), 0)
    
    # 归一化+增强对比度
    diff_norm = (diff_smooth / (diff_smooth.max() + 1e-5) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    diff_enhanced = clahe.apply(diff_norm)
    
    # 自适应阈值
    flat_data = diff_enhanced.flatten()
    dynamic_high = np.percentile(flat_data, 97.5) 
    dynamic_low = np.percentile(flat_data, 96)  
    dynamic_high = max(dynamic_high, 70) 
    dynamic_low = max(dynamic_low, 40)

    # 滞后阈值分割
    mask_binary = apply_hysteresis_threshold(
        diff_enhanced, dynamic_low, dynamic_high
    ).astype(np.uint8) * 255
    
    # 形态学操作优化mask
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_refined = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_connected = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel_close)
    
    # 筛选有效轮廓
    cnts, _ = cv2.findContours(mask_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask_connected)
    h, w = mask_final.shape
    for c in cnts:
        area = cv2.contourArea(c)
        if 20 < area < (h * w * 0.2):  # 过滤过小/过大轮廓
            cv2.drawContours(mask_final, [c], -1, 255, -1)
    
    # 生成软权重（调整到max_weight范围）
    if np.any(mask_final > 0):
        dist = cv2.distanceTransform(mask_final, cv2.DIST_L2, 5)
        # 权重计算：距离越近权重越高，最大值max_weight，最小值base_weight
        soft_weight = np.exp(-dist / 8.0) * (max_weight - base_weight) + base_weight
        # 确保权重不超过最大值
        soft_weight = np.clip(soft_weight, base_weight, max_weight)
    else:
        soft_weight = np.ones_like(diff_map, dtype=np.float32) * base_weight
    
    return mask_final, soft_weight

def prepare_mask_for_latent(soft_weight_np, latent_size, device):
    """
    将256×256的软权重图转换为latent空间的权重tensor
    Args:
        soft_weight_np: numpy array [h, w] (256×256)
        latent_size: int, latent空间尺寸（32）
        device: torch.device
    Returns:
        weight_latent: torch.Tensor [1, 1, latent_size, latent_size]
    """
    # 转换为tensor并调整维度
    weight_tensor = torch.from_numpy(soft_weight_np).float().to(device)  # [256, 256]
    weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]
    
    # 下采样到latent尺寸（使用双线性插值，保留权重分布）
    weight_latent = F.interpolate(
        weight_tensor, 
        size=(latent_size, latent_size), 
        mode='bilinear', 
        align_corners=False
    )  # [1, 1, 32, 32]
    
    return weight_latent

# -------------------------- 全局工具函数（不变） --------------------------
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

# 新增：EMA decay调度函数（关键优化）
def get_ema_decay(train_steps, base_decay=0.9999, min_decay=0.99, warmup_steps=1000):
    """
    动态调整EMA decay值：
    - 热身阶段线性增加decay，避免初始阶段EMA更新过快
    - 训练后期使用高decay，稳定EMA权重
    """
    if train_steps < warmup_steps:
        decay = min_decay + (base_decay - min_decay) * (train_steps / warmup_steps)
    else:
        decay = base_decay
    return decay

def save_val_sample_visualization(epoch, video_val, vae, val_diffusion, device, val_fold, generate_vessel_mask_adaptive, raw_model, ema_model=None):
    """
    保存单个样本的验证可视化图片（同时支持raw model和EMA model）
    Args:
        epoch: 当前epoch
        video_val: 验证视频 [B, F, C, H, W]
        vae: VAE模型
        val_diffusion: 扩散模型
        device: 设备
        val_fold: 可视化保存目录
        generate_vessel_mask_adaptive: 血管mask生成函数
        raw_model: 原始训练模型（必传）
        ema_model: EMA模型（可选，不传则只生成raw model的图）
    """
    b_val, f_val, c_val, h_val, w_val = video_val.shape
    
    # ========== 第一步：通用预处理（只做一次） ==========
    # VAE编码（共用）
    video_val_flat = rearrange(video_val, 'b f c h w -> (b f) c h w')
    latent_val = vae.encode(video_val_flat).latent_dist.sample().mul_(0.18215)
    latent_val = rearrange(latent_val, '(b f) c h w -> b f c h w', b=b_val)

    # 构建mask（共用）
    _, _, _, h_latent_val, w_latent_val = latent_val.shape
    mask_val = torch.ones(b_val, f_val, h_latent_val, w_latent_val, device=device)
    mask_val[:, 0, :, :] = 0.0
    mask_val[:, -1, :, :] = 0.0

    # 生成血管mask（共用，用于可视化）
    video_val_np = video_val.detach().cpu().numpy()
    video_val_gray = np.mean(video_val_np, axis=2)
    sample_idx = 0
    frames_gray = [video_val_gray[sample_idx, t] for t in range(f_val)]
    mask_final, soft_weight = generate_vessel_mask_adaptive(frames_gray)

    # 处理mask_val上采样（共用）
    mask_val_single = mask_val[sample_idx:sample_idx+1]  # [1, F, 32, 32]
    mask_val_5d = mask_val_single.unsqueeze(1)           # [1, 1, F, 32, 32]
    mask_val_upsampled = F.interpolate(
        mask_val_5d,
        size=(f_val, h_val, w_val),
        mode='nearest'
    ).squeeze(1)  # [1, F, H, W]
    mask_val_3c = mask_val_upsampled.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # [1, F, 3, H, W]

    # 血管mask可视化处理（共用）
    vessel_mask_vis = torch.from_numpy(mask_final).float().to(device)  # [256,256]
    vessel_mask_vis = vessel_mask_vis.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,256,256]
    vessel_mask_vis = vessel_mask_vis.repeat(1, f_val, 3, 1, 1)  # [1,F,3,256,256]
    vessel_mask_vis = (vessel_mask_vis / 255.0) * 2 - 1  # 归一化到[-1,1]

    # ========== 第二步：生成raw model的结果并保存 ==========
    with torch.no_grad():
        # 采样生成（raw model）
        z = torch.randn_like(latent_val.permute(0, 2, 1, 3, 4))
        samples_raw = val_diffusion.p_sample_loop(
            raw_model.forward,
            z.shape,
            z,
            clip_denoised=False,
            progress=False,
            device=device,
            raw_x=latent_val.permute(0, 2, 1, 3, 4),
            mask=mask_val
        )

        # 解码（raw model）
        samples_raw = samples_raw.permute(1, 0, 2, 3, 4) * mask_val + latent_val.permute(2, 0, 1, 3, 4) * (1 - mask_val)
        samples_raw = samples_raw.permute(1, 2, 0, 3, 4)
        samples_raw_flat = rearrange(samples_raw, 'b f c h w -> (b f) c h w') / 0.18215
        decoded_raw = vae.decode(samples_raw_flat).sample
        decoded_raw = rearrange(decoded_raw, '(b f) c h w -> b f c h w', b=b_val)

        # 强制转灰度（raw model）
        decoded_raw_gray = decoded_raw.mean(dim=2, keepdim=True)
        decoded_raw = decoded_raw_gray.repeat(1, 1, 3, 1, 1)

        # 拼接并保存raw model的图
        video_val_single = video_val[sample_idx:sample_idx+1]
        decoded_raw_single = decoded_raw[sample_idx:sample_idx+1]
        val_pic_raw = torch.cat([
            video_val_single,                  # 原始视频
            video_val_single * (1 - mask_val_3c),  # mask后的视频
            decoded_raw_single,                # raw model预测视频
            vessel_mask_vis                    # 血管mask
        ], dim=1)  # [1, 4*F, 3, H, W]

        # 保存raw model的图
        val_pic_raw_flat = rearrange(val_pic_raw, 'b f c h w -> (b f) c h w')
        save_image(
            val_pic_raw_flat,
            os.path.join(val_fold, f"Epoch_{epoch+1}_sample_raw.png"),
            nrow=f_val,
            normalize=True,
            value_range=(-1, 1)
        )
        print(f"✅ Epoch {epoch+1} raw model可视化保存完成：{os.path.join(val_fold, f'Epoch_{epoch+1}_sample_raw.png')}")

    # ========== 第三步：生成EMA model的结果并保存（如果传了ema_model） ==========
    if ema_model is not None:
        with torch.no_grad():
            # 采样生成（EMA model）
            z_ema = torch.randn_like(latent_val.permute(0, 2, 1, 3, 4))
            samples_ema = val_diffusion.p_sample_loop(
                ema_model.forward,  # 使用EMA模型
                z_ema.shape,
                z_ema,
                clip_denoised=False,
                progress=False,
                device=device,
                raw_x=latent_val.permute(0, 2, 1, 3, 4),
                mask=mask_val
            )

            # 解码（EMA model）
            samples_ema = samples_ema.permute(1, 0, 2, 3, 4) * mask_val + latent_val.permute(2, 0, 1, 3, 4) * (1 - mask_val)
            samples_ema = samples_ema.permute(1, 2, 0, 3, 4)
            samples_ema_flat = rearrange(samples_ema, 'b f c h w -> (b f) c h w') / 0.18215
            decoded_ema = vae.decode(samples_ema_flat).sample
            decoded_ema = rearrange(decoded_ema, '(b f) c h w -> b f c h w', b=b_val)

            # 强制转灰度（EMA model）
            decoded_ema_gray = decoded_ema.mean(dim=2, keepdim=True)
            decoded_ema = decoded_ema_gray.repeat(1, 1, 3, 1, 1)

            # 拼接并保存EMA model的图
            decoded_ema_single = decoded_ema[sample_idx:sample_idx+1]
            val_pic_ema = torch.cat([
                video_val_single,                  # 原始视频
                video_val_single * (1 - mask_val_3c),  # mask后的视频
                decoded_ema_single,                # EMA model预测视频
                vessel_mask_vis                    # 血管mask
            ], dim=1)  # [1, 4*F, 3, H, W]

            # 保存EMA model的图
            val_pic_ema_flat = rearrange(val_pic_ema, 'b f c h w -> (b f) c h w')
            save_image(
                val_pic_ema_flat,
                os.path.join(val_fold, f"Epoch_{epoch+1}_sample_ema.png"),
                nrow=f_val,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"✅ Epoch {epoch+1} EMA model可视化保存完成：{os.path.join(val_fold, f'Epoch_{epoch+1}_sample_ema.png')}")

# -------------------------- 主训练函数（修改loss计算部分） --------------------------
def main(args):
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
    latent_size = args.image_size // 8  # 256//8=32
    additional_kwargs = {'num_frames': args.tar_num_frames, 'mode': 'video'}
    
    # 先在CPU创建模型，减少GPU显存峰值
    model = MVIF_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **additional_kwargs
    )
    # 移到GPU（非阻塞）
    model = model.to(device, non_blocking=True)

    # ========== EMA模型初始化修复 ==========
    # EMA模型直接创建在GPU（避免后续设备迁移错误）
    ema = deepcopy(model).to(device, non_blocking=True)
    requires_grad(ema, False)
    ema.eval()  # 强制设置为eval模式

    # VAE模型（冻结）
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_path, 
        subfolder="sd-vae-ft-mse"
    ).to(device, non_blocking=True)
    vae.requires_grad_(False)
    vae.eval()

    # 6. 加载预训练权重（仅rank0打印日志）
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
                
                # 同步更新EMA模型（关键：加载预训练权重后更新EMA）
                update_ema(ema, get_raw_model(model), decay=0.0)  # 完全复制
            
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
            broadcast_buffers=True
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
        latest_ckpt = os.path.join(checkpoint_dir, "latest_epoch_train_model.pth")
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
            
            # 3. 加载EMA模型（关键：续训时正确加载EMA）
            if "ema" in checkpoint and checkpoint["ema"] is not None:
                ema.load_state_dict(checkpoint["ema"])
                ema.eval()  # 重新设置为eval模式
            
            # 4. 读取元数据（核心修复：先赋值train_steps）
            train_steps = checkpoint.get("train_steps", 0)  # 先拿到checkpoint里的训练步数
            best_val_metric = checkpoint.get("best_val_metric", float('inf'))
            
            # ========== 核心修复：正确计算epoch和step ==========
            # 梯度累积步数（从args读取，避免硬编码）
            grad_accum = args.gradient_accumulation_steps
            # 1. 把优化器步数转换回原始batch数（优化器步数 × 梯度累积步数）
            total_batch_steps = train_steps * grad_accum
            # 2. 每个epoch的总batch数
            batches_per_epoch = len(loader_train)
            
            # 3. 计算真实的epoch和剩余batch数
            if total_batch_steps >= batches_per_epoch:
                # 已完成完整epoch，续训到下一个epoch的第一步
                first_epoch = total_batch_steps // batches_per_epoch
                resume_step = 0
            else:
                # 未完成当前epoch，续训到剩余batch
                first_epoch = 0
                resume_step = total_batch_steps
            
            # 加载历史最优指标
            if best_val_metric != float('inf') and rank == 0:
                logger.info(f"Resumed best validation metric: {best_val_metric:.4f}")
            
            # 5. 强制清理所有临时数据
            del checkpoint
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # 打印正确的续训日志
            if rank == 0:
                logger.info(f"Resumed actual epoch: {first_epoch}, batch step: {resume_step}")
                logger.info(f"Total trained batches: {total_batch_steps}/{batches_per_epoch}")
                logger.info(f"Memory after load: {torch.cuda.memory_allocated(device)/1024**3:.2f} GiB")
                logger.warning("跳过了优化器状态加载，优化器将从头开始（学习率不变）")
        else:
            if rank == 0:
                logger.warning(f"Checkpoint {latest_ckpt} not found, start from scratch")

    # 续训完成后：关闭expandable_segments + 模型编译
    # 1. 关闭expandable_segments（训练阶段用）
    torch.backends.cuda.expandable_segments = False
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    # 2. 延迟编译模型（避免加载阶段占用显存）
    if args.use_compile and torch.cuda.is_available():
        if rank == 0:
            logger.info("Compiling model with torch.compile (delayed)")
        model = torch.compile(model, mode="reduce-overhead")
    # 3. 最后一次显存清理
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # 12. 训练循环
    num_update_steps_per_epoch = math.ceil(len(loader_train))
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    model.train()
    ema.eval()  # EMA始终保持eval模式
    running_loss = 0.0
    log_steps = 0
    start_time = time()

    if rank == 0:
        logger.info(f"Start training for {num_train_epochs} epochs (first epoch: {first_epoch})")
        logger.info(f"Vessel mask max weight: {getattr(args, 'vessel_max_weight', 10.0)}")

    for epoch in range(first_epoch, num_train_epochs):
        sampler_train.set_epoch(epoch)  # 分布式采样器epoch同步
        diffusion = create_diffusion(
            timestep_respacing=args.timestep_respacing_train,
            diffusion_steps=args.diffusion_steps
        )
        diffusion.training = True

        # pbar = tqdm(loader_train, total=len(loader_train), desc=f"Rank{rank} Epoch {epoch+1}", disable=(rank != 0))
        pbar = tqdm(loader_train, total=len(loader_train), desc=f"Rank{rank} Epoch {epoch + 1}", disable=(rank != 0))
        for step, batch in enumerate(pbar):
            # 跳过断点续训的步骤
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                train_steps += 1
                pbar.update(1)
                continue

            # 数据加载
            video = batch['video'].to(device, non_blocking=True)  # [B, F, C, H, W]
            b, f, c, h, w = video.shape

            # ========== 核心修改1：生成血管mask和软权重 ==========
            # 1. 将tensor转换为numpy（用于mask生成）
            video_np = video.detach().cpu().numpy()  # [B, F, C, H, W]
            # 转为灰度图（取第一个通道，或计算均值）
            video_gray_np = np.mean(video_np, axis=2)  # [B, F, H, W]
            
            # 2. 为每个样本生成软权重mask
            weight_list = []
            for i in range(b):
                frames_gray = [video_gray_np[i, t] for t in range(f)]
                # 生成软权重（max_weight可通过args配置，默认10）
                max_weight = getattr(args, 'vessel_max_weight', 10.0)
                _, soft_weight_np = generate_vessel_mask_adaptive(frames_gray, max_weight=max_weight)
                # 转换为latent空间权重
                weight_latent = prepare_mask_for_latent(soft_weight_np, latent_size, device)
                weight_list.append(weight_latent)
            
            # 3. 拼接batch的权重 [B, 1, latent_h, latent_w]
            weight_batch = torch.cat(weight_list, dim=0)  # [B, 1, 32, 32]
            # 扩展到帧维度 [B, F, 1, 32, 32]
            weight_batch = weight_batch.unsqueeze(1).repeat(1, f, 1, 1, 1)

            # VAE编码（latent空间）
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.mixed_precision):
                video_flat = rearrange(video, 'b f c h w -> (b f) c h w')
                latent = vae.encode(video_flat).latent_dist.sample()
                latent.mul_(0.18215)  # 归一化
                latent = rearrange(latent, '(b f) c h w -> b f c h w', b=b)

            # 构建原始mask（第一帧和最后一帧不mask）
            b, f, c, h_latent, w_latent = latent.shape  # latent维度：[b, f, c, 32, 32]
            mask = torch.ones(b, f, h_latent, w_latent, device=device)
            mask[:, 0, :, :] = 0  # 第一帧
            mask[:, -1, :, :] = 0  # 最后一帧

            # 随机时间步
            t = torch.randint(0, diffusion.num_timesteps, (b,), device=device)

            # 计算损失
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                loss_dict = diffusion.training_losses(model, latent, t, mask=mask)
                base_loss = loss_dict["loss"]  # [B, F, C, 32, 32]
                
                # ========== 优化：带归一化的软权重loss计算 ==========
                # 将权重扩展到通道维度 [B, F, C, 32, 32]
                weight_expanded = weight_batch.repeat(1, 1, c, 1, 1)
                
                # 1. 计算加权loss
                weighted_loss = base_loss * weight_expanded
                
                # 2. 权重归一化：保证加权后loss的均值和原始loss一致
                # 计算权重的均值（用于缩放，保持loss尺度）
                weight_mean = weight_expanded.mean()
                # 归一化系数：让加权loss的均值 ≈ 原始loss均值
                norm_factor = 1.0 / weight_mean
                # 应用归一化
                weighted_loss = weighted_loss * norm_factor
                
                # 3. 最终loss（保持和原代码一致的梯度累积逻辑）
                loss = weighted_loss.mean() / args.gradient_accumulation_steps

            # 反向传播
            if args.mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

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

                # ========== EMA更新修复 ==========
                # 动态计算decay值，避免固定decay的问题
                ema_decay = get_ema_decay(train_steps, base_decay=0.9999, min_decay=0.99, warmup_steps=1000)
                # 确保EMA模型在eval模式下更新
                with torch.no_grad():
                    update_ema(ema, get_raw_model(model), decay=ema_decay)

                # 日志记录
                running_loss += loss.item() * args.gradient_accumulation_steps
                log_steps += 1
                train_steps += 1

                # 打印进度
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "grad_norm": f"{grad_norm:.4f}",
                    "lr": f"{opt.param_groups[0]['lr']:.6f}",
                    "ema_decay": f"{ema_decay:.6f}"  # 新增：监控EMA decay
                })

                # 定期日志
                if train_steps % args.log_every == 0 and rank == 0:
                    avg_loss = running_loss / log_steps
                    steps_per_sec = log_steps / (time() - start_time)
                    logger.info(
                        f"Step {train_steps:07d} | Loss: {avg_loss:.4f} | "
                        f"Grad Norm: {grad_norm:.4f} | Steps/Sec: {steps_per_sec:.2f} | "
                        f"EMA Decay: {ema_decay:.6f}"
                    )
                    write_tensorboard(tb_writer, 'Train/Loss', avg_loss, train_steps)
                    write_tensorboard(tb_writer, 'Train/GradNorm', grad_norm, train_steps)
                    write_tensorboard(tb_writer, 'Train/LR', opt.param_groups[0]['lr'], train_steps)
                    write_tensorboard(tb_writer, 'Train/EMADecay', ema_decay, train_steps)  # 新增
                    running_loss = 0.0
                    log_steps = 0
                    start_time = time()

            # 终止条件
            if train_steps >= args.max_train_steps:
                break

        # ========== 每10个epoch保存模型 ==========
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

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # -------------------------- 验证阶段（同步修改loss加权） --------------------------
        # ========== 1. 每个epoch都执行：单样本可视化保存（轻量） ==========
        # 每个epoch的轻量可视化（调用修改后的函数）
        if rank == 0:
            model.eval()
            vae.eval()
            val_diffusion = create_diffusion(
                timestep_respacing=args.timestep_respacing_test,
                diffusion_steps=args.diffusion_steps
            )
            
            with torch.no_grad():
                # 只取验证集第一个batch
                val_batch = next(iter(loader_val))
                video_val = val_batch['video'].to(device)
                
                # 调用可视化函数（同时传raw model和EMA model）
                save_val_sample_visualization(
                    epoch=epoch,
                    video_val=video_val,
                    vae=vae,
                    val_diffusion=val_diffusion,
                    device=device,
                    val_fold=val_fold,
                    generate_vessel_mask_adaptive=generate_vessel_mask_adaptive,
                    raw_model=get_raw_model(model),  # 原始模型（解包DDP）
                    ema_model=ema                    # EMA模型（可选，传了就生成EMA的图）
                )
            
            model.train()  # 回到训练模式
            torch.cuda.empty_cache()
        
        # ========== 2. 每val_interval个epoch执行：完整验证（指标+最优模型） ==========
        if (epoch + 1) % args.val_interval == 0 and rank == 0:
            model.eval()
            vae.eval()
            val_diffusion = create_diffusion(
                timestep_respacing=args.timestep_respacing_test,
                diffusion_steps=args.diffusion_steps
            )
    
            val_mae, val_mse, val_psnr = 0.0, 0.0, 0.0
            val_pbar = tqdm(loader_val, total=len(loader_val), desc="Full Validation")
            
            with torch.no_grad():
                for val_step, val_batch in enumerate(val_pbar):
                    video_val = val_batch['video'].to(device)
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
    
                    # 强制转灰度
                    decoded_gray = decoded.mean(dim=2, keepdim=True)
                    decoded = decoded_gray.repeat(1, 1, 3, 1, 1)
    
                    # 计算指标（修复数据范围：[-1,1] → 0-255）
                    gt_np = (video_val.detach().cpu().numpy() * 0.5 + 0.5) * 255.0
                    pred_np = (decoded.detach().cpu().numpy() * 0.5 + 0.5) * 255.0
                    val_mae += np.mean(np.abs(pred_np - gt_np)) / len(loader_val)
                    val_mse += mean_squared_error(gt_np.reshape(-1), pred_np.reshape(-1)) / len(loader_val)
                    val_psnr += peak_signal_noise_ratio(gt_np, pred_np, data_range=255) / len(loader_val)
    
            # 记录完整验证指标
            val_metric = (val_mae + val_mse) / 2
            logger.info(
                f"Epoch {epoch+1} Full Validation | MAE: {val_mae:.4f} | MSE: {val_mse:.4f} | "
                f"PSNR: {val_psnr:.4f} | Metric: {val_metric:.4f} | Best Metric: {best_val_metric:.4f}"
            )
            write_tensorboard(tb_writer, 'Val/MAE', val_mae, epoch+1)
            write_tensorboard(tb_writer, 'Val/MSE', val_mse, epoch+1)
            write_tensorboard(tb_writer, 'Val/PSNR', val_psnr, epoch+1)
            write_tensorboard(tb_writer, 'Val/Best_Metric', best_val_metric, epoch+1)
    
            # 保存最优模型
            if val_metric < best_val_metric:
                prev_best = best_val_metric
                best_val_metric = val_metric
                checkpoint = {
                    "model": get_raw_model(model).state_dict(),
                    "ema": ema.state_dict(),
                    "val_metric": val_metric,
                    "epoch": epoch+1,
                    "train_steps": train_steps,
                    "prev_best_metric": prev_best
                }
                best_ckpt_path = f"{checkpoint_dir}/best_epoch_train_model.pth"
                torch.save(checkpoint, best_ckpt_path)
                del checkpoint
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info(
                    f"Saved BEST model to {best_ckpt_path} | "
                    f"New best metric: {val_metric:.4f} (previous: {prev_best:.4f}) | "
                    f"Improvement: {prev_best - val_metric:.4f}"
                )
    
            model.train()  # 回到训练模式
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        # if (epoch + 1) % args.val_interval == 0 and rank == 0:
        #     model.eval()
        #     vae.eval()
        #     val_diffusion = create_diffusion(
        #         timestep_respacing=args.timestep_respacing_test,
        #         diffusion_steps=args.diffusion_steps
        #     )

        #     val_mae, val_mse, val_psnr = 0.0, 0.0, 0.0
        #     val_pbar = tqdm(loader_val, total=len(loader_val), desc="Validation")
            
        #     with torch.no_grad():
        #         for val_step, val_batch in enumerate(val_pbar):
        #             video_val = val_batch['video'].to(device)
        #             b_val, f_val, c_val, h_val, w_val = video_val.shape

        #             # ========== 验证阶段也生成mask（可选，用于可视化） ==========
        #             video_val_np = video_val.detach().cpu().numpy()
        #             video_val_gray = np.mean(video_val_np, axis=2)
                    
        #             # VAE编码
        #             video_val_flat = rearrange(video_val, 'b f c h w -> (b f) c h w')
        #             latent_val = vae.encode(video_val_flat).latent_dist.sample().mul_(0.18215)
        #             latent_val = rearrange(latent_val, '(b f) c h w -> b f c h w', b=b_val)

        #             # 构建mask（latent空间）
        #             _, _, _, h_latent_val, w_latent_val = latent_val.shape
        #             mask_val = torch.ones(b_val, f_val, h_latent_val, w_latent_val, device=device)
        #             mask_val[:, 0, :, :] = 0.0  # 第一帧不mask
        #             mask_val[:, -1, :, :] = 0.0  # 最后一帧不mask

        #             # 采样生成（使用EMA模型进行验证，更准确）
        #             z = torch.randn_like(latent_val.permute(0, 2, 1, 3, 4))  # [b, c, f, h, w]
        #             samples = val_diffusion.p_sample_loop(
        #                 # ema.forward,  # 关键：验证时使用EMA模型
        #                 get_raw_model(model).forward,
        #                 z.shape,
        #                 z,
        #                 clip_denoised=False,
        #                 progress=False,
        #                 device=device,
        #                 raw_x=latent_val.permute(0, 2, 1, 3, 4),
        #                 mask=mask_val
        #             )

        #             # 解码
        #             samples = samples.permute(1, 0, 2, 3, 4) * mask_val + latent_val.permute(2, 0, 1, 3, 4) * (1 - mask_val)
        #             samples = samples.permute(1, 2, 0, 3, 4)
        #             samples_flat = rearrange(samples, 'b f c h w -> (b f) c h w') / 0.18215
        #             decoded = vae.decode(samples_flat).sample
        #             decoded = rearrange(decoded, '(b f) c h w -> b f c h w', b=b_val)

        #             # ========== 新增：强制转灰度 ==========
        #             # 方法A：取平均（推荐，保留亮度信息）
        #             decoded_gray = decoded.mean(dim=2, keepdim=True)  # [B, F, 1, H, W]
        #             decoded = decoded_gray.repeat(1, 1, 3, 1, 1)      # [B, F, 3, H, W]，R=G=B

        #             # 计算指标
        #             gt_np = video_val.detach().cpu().numpy()
        #             pred_np = decoded.detach().cpu().numpy()
        #             val_mae += np.mean(np.abs(pred_np - gt_np)) / len(loader_val)
        #             val_mse += mean_squared_error(gt_np.reshape(-1), pred_np.reshape(-1)) / len(loader_val)
        #             val_psnr += peak_signal_noise_ratio(gt_np, pred_np, data_range=2) / len(loader_val)

        #             # 保存验证图片（新增血管mask可视化）
        #             if val_step == 0:
        #                 # 只可视化第一个样本（避免多样本拼接混乱）
        #                 sample_idx = 0
        #                 # 1. 提取第一个样本的所有数据
        #                 video_val_single = video_val[sample_idx:sample_idx+1]  # [1, 12, 3, 256, 256]
        #                 decoded_single = decoded[sample_idx:sample_idx+1]      # [1, 12, 3, 256, 256]
                        
        #                 # 2. 生成第一个样本的血管mask
        #                 frames_gray = [video_val_gray[sample_idx, t] for t in range(f_val)]
        #                 mask_final, soft_weight = generate_vessel_mask_adaptive(frames_gray)
                        
        #                 # 3. 处理mask_val（只取第一个样本）
        #                 mask_val_single = mask_val[sample_idx:sample_idx+1]  # [1, 12, 32, 32]
        #                 mask_val_5d = mask_val_single.unsqueeze(1)           # [1, 1, 12, 32, 32]
        #                 mask_val_upsampled = F.interpolate(
        #                     mask_val_5d,
        #                     size=(f_val, h_val, w_val),
        #                     mode='nearest'
        #                 ).squeeze(1)  # [1, 12, 256, 256]
        #                 mask_val_3c = mask_val_upsampled.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # [1, 12, 3, 256, 256]
                        
        #                 # 4. 血管mask可视化（只针对第一个样本）
        #                 vessel_mask_vis = torch.from_numpy(mask_final).float().to(device)  # [256,256]
        #                 vessel_mask_vis = vessel_mask_vis.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,256,256]
        #                 vessel_mask_vis = vessel_mask_vis.repeat(1, f_val, 3, 1, 1)  # [1,12,3,256,256]
        #                 vessel_mask_vis = (vessel_mask_vis / 255.0) * 2 - 1  # 归一化到[-1,1]
                        
        #                 # 5. 拼接：只显示第一个样本的4列（原始、masked、预测、血管mask）
        #                 val_pic = torch.cat([
        #                     video_val_single,                  # 原始灰度视频 [1,12,3,256,256]
        #                     video_val_single * (1 - mask_val_3c),  # mask后的视频 [1,12,3,256,256]
        #                     decoded_single,                    # 修复后的灰度预测视频 [1,12,3,256,256]
        #                     vessel_mask_vis                    # 血管mask [1,12,3,256,256]
        #                 ], dim=1)  # 拼接后：[1, 48, 3, 256, 256]（12帧×4列）
                        
        #                 # 6. 调整维度并保存（nrow=12，每行显示12帧，共4行）
        #                 val_pic_flat = rearrange(val_pic, 'b f c h w -> (b f) c h w')  # [48,3,256,256]
        #                 save_image(
        #                     val_pic_flat,
        #                     os.path.join(val_fold, f"Epoch_{epoch+1}.png"),
        #                     nrow=12,  # 每行12帧，4列就是4行（原始、masked、预测、mask）
        #                     normalize=True,
        #                     value_range=(-1, 1)
        #                 )
        #                 print(f"✅ 可视化保存完成：仅显示第1个样本，4列×12帧，共48帧")          

        #     # 记录验证指标
        #     val_metric = (val_mae + val_mse) / 2
        #     logger.info(
        #         f"Epoch {epoch+1} Validation | MAE: {val_mae:.4f} | MSE: {val_mse:.4f} | "
        #         f"PSNR: {val_psnr:.4f} | Metric: {val_metric:.4f} | Best Metric: {best_val_metric:.4f}"
        #     )
        #     write_tensorboard(tb_writer, 'Val/MAE', val_mae, epoch+1)
        #     write_tensorboard(tb_writer, 'Val/MSE', val_mse, epoch+1)
        #     write_tensorboard(tb_writer, 'Val/PSNR', val_psnr, epoch+1)
        #     write_tensorboard(tb_writer, 'Val/Best_Metric', best_val_metric, epoch+1)

        #     # 保存最优模型
        #     if val_metric < best_val_metric:
        #         prev_best = best_val_metric
        #         best_val_metric = val_metric
        #         checkpoint = {
        #             "model": get_raw_model(model).state_dict(),
        #             "ema": ema.state_dict(),
        #             "val_metric": val_metric,
        #             "epoch": epoch+1,
        #             "train_steps": train_steps,
        #             "prev_best_metric": prev_best
        #         }
        #         best_ckpt_path = f"{checkpoint_dir}/best_epoch_train_model.pth"
        #         torch.save(checkpoint, best_ckpt_path)
        #         del checkpoint  # 清理
        #         torch.cuda.empty_cache()
        #         torch.cuda.ipc_collect()
        #         logger.info(
        #             f"Saved BEST model to {best_ckpt_path} | "
        #             f"New best metric: {val_metric:.4f} (previous: {prev_best:.4f}) | "
        #             f"Improvement: {prev_best - val_metric:.4f}"
        #         )

        #     model.train()  # 回到训练模式

        # 终止训练
        if train_steps >= args.max_train_steps:
            break

    # 训练结束
    if rank == 0:
        logger.info(f"Training finished! Total steps: {train_steps}")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_acdc.yaml")
    # ========== 新增参数：血管mask最大权重 ==========
    parser.add_argument("--vessel_max_weight", type=float, default=10.0, help="Max weight for vessel region (default: 10)")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))

