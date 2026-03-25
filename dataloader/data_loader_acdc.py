# 相邻3帧的灰度图
import os
import torch
import decord
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import re
import random
from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple
from .video_transforms import *

# 解决多进程 DataLoader 中 PyAV 加载问题（提前导入并初始化）
try:
    import av
except ImportError:
    raise ImportError("请安装 PyAV 库：conda install -c conda-forge av 或 pip install av")

class_labels_map = None
cls_sample_cnt = None

def get_filelist(file_path):
    """递归获取文件列表，过滤非视频文件"""
    Filelist = []
    video_ext = ('.mp4', '.avi', '.mov', '.mkv')  # 仅保留视频文件
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            if filename.lower().endswith(video_ext):
                Filelist.append(os.path.join(home, filename))
    return Filelist

class DecordInit(object):
    """Using Decord to initialize the video_reader (备用视频读取方案)"""
    def __init__(self, num_threads=1, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs

    def __call__(self, filename):
        reader = decord.VideoReader(filename, ctx=self.ctx, num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_threads={self.num_threads})')
        return repr_str

class data_loader(torch.utils.data.Dataset):
    """加载心脏视频数据（ACDC），支持4D（时空+切片）维度"""
    def __init__(self, configs, stage):
        self.stage = stage
        self.configs = configs
        self.target_video_len = self.configs.tar_num_frames
        self.v_decoder = DecordInit()
        self.temporal_sample = TemporalRandomCrop(configs.tar_num_frames)
        
        # 数据变换：区分训练/测试（训练可加随机翻转）
        transform_list = [
            ToTensorVideo(),
            CenterCropResizeVideo(configs.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]
        if self.stage == 'train':
            transform_list.insert(2, RandomHorizontalFlipVideo())  # 训练加水平翻转
        self.transform = transforms.Compose(transform_list)

        # 划分训练/验证/测试集
        if self.stage in ['train', 'val']:
            self.data_path = configs.data_path_train
            all_videos = get_filelist(self.data_path)
            len_train = int(0.9 * len(all_videos))
            if self.stage == 'train':
                self.video_lists = all_videos[:len_train]
            else:
                self.video_lists = all_videos[len_train:]
        elif self.stage == 'test':
            self.data_path = configs.data_path_test
            self.video_lists = get_filelist(self.data_path)
        else:
            raise ValueError(f"stage 必须是 train/val/test，当前为 {self.stage}")
        
        print(f"[{self.stage}] 加载视频数量: {len(self.video_lists)}")

    def _read_video_safe(self, video_path):
        """安全读取视频，兼容不同格式，返回 TCHW 格式张量"""
        try:
            # 使用 torchvision 读取（依赖 PyAV）
            vframes, aframes, info = torchvision.io.read_video(
                filename=video_path, 
                pts_unit='sec', 
                output_format='TCHW',
                pts_per_second=15  # 固定帧率，避免不同视频帧率不一致
            )
            return vframes
        except Exception as e:
            # 备用方案：使用 Decord 读取
            reader = self.v_decoder(video_path)
            frames = reader.get_batch(list(range(len(reader)))).asnumpy()  # THWC
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # 转为 TCHW
            return frames

    def __getitem__(self, index):
        # 1. 读取当前切片视频
        path = self.video_lists[index]
        file_name = os.path.basename(path)
        patient_dir = os.path.dirname(path)
        patient_name = os.path.basename(patient_dir)

        # 解析切片号（增强正则鲁棒性）
        slice_match = re.search(r'slice_(\d+)', file_name)
        if not slice_match:
            raise ValueError(f"文件名 {file_name} 未匹配到 slice_XX 格式")
        slice_num = int(slice_match.group(1))
        ext = os.path.splitext(file_name)[1]
        
        vframes = self._read_video_safe(path)
        
        vframes = vframes.to(torch.uint8)  # 转为uint8，避免归一化异常

        # 7. 时间采样（匹配目标帧数）
        total_frames = len(vframes)
        if self.stage == 'train':
            # 训练：随机采样 + 均匀采样混合
            if random.random() < 0.4:
                frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
            else:
                start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
                frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
        else:
            # 验证/测试：均匀采样（帧数不足则重复）
            if total_frames <= self.target_video_len:
                frame_indice = np.linspace(0, total_frames - 1, self.target_video_len, dtype=int)
            else:
                frame_indice = np.round(np.linspace(0, total_frames - 1, self.target_video_len)).astype(int)

        video = vframes[frame_indice]
        # 8. 数据变换（归一化等）
        video = self.transform(video)

        return {
            'video': video,
            'video_name': f"{patient_name}_slice_{slice_num:02d}",
            'video_gt': video.clone(),  # GT 与输入一致（根据任务需求可修改）
            'video_path': self.video_lists[index],
        }

    def __len__(self):
        return len(self.video_lists)

if __name__ == '__main__':
    from tqdm import tqdm
    import imageio
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    # 加载配置
    config_path = r'../configs/config_acdc.yaml'
    config_dict = OmegaConf.load(config_path)
    dataset = data_loader(config_dict, 'val')

    # 禁用多进程（调试阶段），避免 PyAV 多进程问题
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,  # 调试时设为0，正常训练可设为8
        pin_memory=True,
        drop_last=True
    )

    pbar = tqdm(loader, total=len(loader), desc="Processing")
    for i, batch in enumerate(pbar):
        video = batch['video']
        print(f"批次{i} - 视频形状: {video.shape}, 数值范围: [{video.min():.4f}, {video.max():.4f}]")
        
        # 可视化验证
        video_vis = ((video[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8)
        video_vis = video_vis.permute(0, 2, 3, 1).cpu().contiguous()
        print(f"可视化形状: {video_vis.shape}")
        
        # 保存示例帧（可选）
        if i == 0:
            imageio.imwrite("sample_frame.png", video_vis[0].numpy())
        if i >= 2:  # 仅测试前2个批次
            break