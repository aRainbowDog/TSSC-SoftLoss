import os
import math
import argparse
import datetime
import numpy as np
import torch
import imageio
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
# from dataloader.data_loader_knee import data_loader
from dataloader.data_loader_acdc import data_loader
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from utils.show import *
from utils.utils import load_checkpoint, calculate_metrics
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings
warnings.filterwarnings('ignore')


def main(args):
    # 1. 基础配置与随机种子
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # 2. 目录创建（保存日志和结果）
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{model_string_name}_{args.cur_date}"
    save_dir = f"{experiment_dir}/test_epoch10_ema"
    os.makedirs(save_dir, exist_ok=True)
    # 指标日志文件
    metrics_log = os.path.join(save_dir, "test_epoch_10_.log")
    f_metrics = open(metrics_log, 'w', encoding='utf-8')
    f_metrics.write(f"测试指标计算日志 - {datetime.datetime.now()}\n")
    f_metrics.write(f"测试模型权重: {args.test_ckpt}\n")
    f_metrics.write(f"测试数据集路径: {args.data_path_test}\n\n")

    # 3. 设备与模型加载（复用test.py逻辑）
    os.environ["CUDA_VISIBLE_DEVICES"] = args.test_gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载VAE
    diffusion = create_diffusion(timestep_respacing=args.timestep_respacing_test, diffusion_steps=args.diffusion_steps)
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path, subfolder="sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)
    print(f"VAE加载完成: {args.pretrained_vae_model_path}")

    # 加载MVIF模型
    checkpoint_path = os.path.join(experiment_dir, "checkpoints", args.test_ckpt)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    latent_size = args.image_size // 8
    additional_kwargs = {'num_frames': args.tar_num_frames, 'mode': 'video'}
    model = MVIF_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **additional_kwargs).to(device)


    print('使用EMA权重！')
    f_metrics.write('使用EMA权重！\n')
    checkpoint = checkpoint["ema"]
    # print('使用raw权重！')
    # f_metrics.write('使用raw权重！\n')
    # checkpoint = checkpoint["model"]
    
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"模型加载完成: {checkpoint_path}")

    print(f"Loaded keys: {len(pretrained_dict)} / {len(model_dict)}")
    if len(pretrained_dict) == 0:
        print("警告：模型权重完全没加载上！检查 state_dict 的 Key！")

    # 4. 加载测试数据集
    dataset_test = data_loader(args, stage='test')
    loader_test = DataLoader(
        dataset_test, 
        batch_size=int(args.test_batch_size), 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    print(f"测试集数量: {len(loader_test)}")
    f_metrics.write(f"测试集数量: {len(loader_test)}\n\n")

    # 5. 测试与指标计算
    vae.eval()
    model.eval()
    diffusion.training = False

    # 初始化指标汇总
    total_mse, total_mae, total_psnr, total_ssim = 0.0, 0.0, 0.0, 0.0
    sample_count = 0

    pbar = tqdm(loader_test, total=len(loader_test), desc="计算测试指标")
    for idx, video_data_test in enumerate(pbar):
        with torch.no_grad():
            # 5.1 加载真实视频（GT）
            x_test = video_data_test['video'].to(device)
            x_test_name = video_data_test['video_name'][0] + '.mp4'  # 单样本名称
            x_test_path = video_data_test['video_path'][0]  # 完整路径

            print(f"[{idx+1:04d}] 正在测试: {x_test_path}")

            # 5.2 构建输入（仅保留首尾帧）
            x_test_intp = torch.zeros_like(x_test)
            x_test_intp[:, 0, :, :, :] = x_test[:, 0, :, :, :]
            x_test_intp[:, -1, :, :, :] = x_test[:, -1, :, :, :]

            # 5.3 VAE编码
            b, _, _, _, _ = x_test_intp.shape
            x_ = rearrange(x_test_intp, 'b f c h w -> (b f) c h w').contiguous()
            latent = vae.encode(x_).latent_dist.sample().mul_(0.18215)
            latent = rearrange(latent, '(b f) c h w -> b f c h w', b=b).contiguous()

            # 5.4 生成掩码
            b, f, c, h, w = latent.shape
            mask = torch.ones(b, f, h, w).to(device)
            mask[:, 0, :, :] = 0  # 首帧不掩码
            mask[:, -1, :, :] = 0  # 尾帧不掩码

            # 5.5 扩散模型推理生成视频
            sample_fn = model.forward
            z = torch.randn(latent.shape, dtype=x_.dtype, device=device)
            z = z.permute(0, 2, 1, 3, 4)
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, 
                progress=False, device=device, 
                raw_x=latent.permute(0, 2, 1, 3, 4), mask=mask
            )

            # 5.6 融合首尾帧 + VAE解码
            samples = samples.permute(1, 0, 2, 3, 4) * mask + latent.permute(2, 0, 1, 3, 4) * (1 - mask)
            samples = samples.permute(1, 2, 0, 3, 4)
            samples = rearrange(samples, 'b f c h w -> (b f) c h w') / 0.18215
            decoded_x = vae.decode(samples).sample
            decoded_x = rearrange(decoded_x, '(b f) c h w -> b f c h w', b=b).contiguous()

            # 5.7 数据格式对齐（和test.py一致的后处理）
            x_test[:, :, 0, :, :] = x_test[:, :, 1, :, :]
            x_test[:, :, 2, :, :] = x_test[:, :, 1, :, :]
            decoded_x[:, :, 0, :, :] = decoded_x[:, :, 1, :, :]
            decoded_x[:, :, 2, :, :] = decoded_x[:, :, 1, :, :]

            # 5.8 转换为[0,255]的np.array（适配calculate_metrics）
            # GT视频: [f, h, w, c]
            gt_video = ((x_test[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous().numpy()
            # 预测视频: [f, h, w, c]
            pred_video = ((decoded_x[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous().numpy()

            # 5.9 计算单样本指标
            avg_mse, avg_mae, avg_psnr, avg_ssim = calculate_metrics(pred_video, gt_video)
            
            # 5.10 汇总指标
            total_mse += avg_mse
            total_mae += avg_mae
            total_psnr += avg_psnr
            total_ssim += avg_ssim
            sample_count += 1

            # 5.11 打印并记录单样本指标
            pbar.set_postfix({
                "PSNR": f"{avg_psnr:.4f}",
                "SSIM": f"{avg_ssim:.4f}",
                "MAE": f"{avg_mae:.4f}"
            })
            f_metrics.write(f"样本 {idx+1} - {x_test_name}\n")
            f_metrics.write(f"  MSE: {avg_mse:.6f}\n")
            f_metrics.write(f"  MAE: {avg_mae:.6f}\n")
            f_metrics.write(f"  PSNR: {avg_psnr:.4f}\n")
            f_metrics.write(f"  SSIM: {avg_ssim:.4f}\n\n")

            # 可选：保存预测视频（和test.py一致）
            os.makedirs(os.path.join(save_dir, 'GT'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'Pred'), exist_ok=True)
            imageio.mimwrite(os.path.join(save_dir, 'GT', x_test_name), gt_video, fps=2, codec='libx264')
            imageio.mimwrite(os.path.join(save_dir, 'Pred', x_test_name), pred_video, fps=2, codec='libx264')

    # 6. 计算整体平均指标
    avg_total_mse = total_mse / sample_count
    avg_total_mae = total_mae / sample_count
    avg_total_psnr = total_psnr / sample_count
    avg_total_ssim = total_ssim / sample_count

    # 7. 输出并保存最终指标
    print("\n==================== 测试集平均指标 ====================")
    print(f"平均MSE: {avg_total_mse:.6f}")
    print(f"平均MAE: {avg_total_mae:.6f}")
    print(f"平均PSNR: {avg_total_psnr:.4f}")
    print(f"平均SSIM: {avg_total_ssim:.4f}")
    print("========================================================")

    f_metrics.write("==================== 测试集平均指标 ====================\n")
    f_metrics.write(f"平均MSE: {avg_total_mse:.6f}\n")
    f_metrics.write(f"平均MAE: {avg_total_mae:.6f}\n")
    f_metrics.write(f"平均PSNR: {avg_total_psnr:.4f}\n")
    f_metrics.write(f"平均SSIM: {avg_total_ssim:.4f}\n")
    f_metrics.write("========================================================\n")
    f_metrics.close()

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../scripts/configs/config_cta.yaml")  # 根据数据集选择配置文件
    args = parser.parse_args()
    main(OmegaConf.load(args.config))