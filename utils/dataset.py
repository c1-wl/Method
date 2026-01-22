import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import cv2
import os
from config import Config


class NUSDataset(Dataset):
    def __init__(self, root_dir, mode='training', crop_size=512):

        self.mode = mode
        self.crop_size = crop_size
        self.root_dir = os.path.join(root_dir, mode)
        self.mat_dir = os.path.join(self.root_dir, 'mat')
        self.png_dir = os.path.join(self.root_dir, 'png')

        camera_data = sio.loadmat(Config.camera_sensitivity_path)
        self.camera_sensitivity = camera_data['CRF'][:, :Config.spectral_bands].astype(np.float32)  # 3xN

        self.filenames = []
        for f in os.listdir(self.mat_dir):
            if f.endswith('.mat'):
                png_path = os.path.join(self.png_dir, f.replace('.mat', '.png'))
                if os.path.exists(png_path):
                    self.filenames.append(f)
                else:
                    print(f"警告: 找不到{png_path}，跳过该MAT文件")
        if not self.filenames:
            raise RuntimeError(f"在{self.mat_dir}中没有找到有效的MAT/PNG文件对")

    def __len__(self):
        return len(self.filenames)

    def _center_crop(self, img):
        h, w = img.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            raise ValueError(f"图像尺寸({h}x{w})小于裁剪尺寸({self.crop_size}x{self.crop_size})")
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        return img[top:top + self.crop_size, left:left + self.crop_size]

    def __getitem__(self, idx):
        # 读取 mat
        mat_path = os.path.join(self.mat_dir, self.filenames[idx])
        mat_data = sio.loadmat(mat_path)
        spectral_img = mat_data['tensor'].astype(np.float32)  # H x W x N
        illuminant = mat_data['illumination'].astype(np.float32)  # N x 1

        # 读取 RGB
        png_path = os.path.join(self.png_dir, self.filenames[idx].replace('.mat', '.png'))
        rgb_img = cv2.imread(png_path, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 中心裁剪
        rgb_img = self._center_crop(rgb_img)               # (S,S,3)
        spectral_img = self._center_crop(spectral_img)     # (S,S,N)

        # 张量化
        rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).contiguous()          # (3,S,S)
        spectral_cube = torch.from_numpy(spectral_img).permute(2, 0, 1).contiguous()  # (N,S,S)

        # 监督RGB白点（illumination × CRF）
        rgb_illuminant = torch.from_numpy(
            np.dot(illuminant.squeeze(), self.camera_sensitivity.T)
        ).float()  # (3,)

        return {
            'rgb_img': rgb_img,
            'spectral_cube': spectral_cube,
            'rgb_illuminant': rgb_illuminant
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("批次中没有有效数据")
    return torch.utils.data.default_collate(batch)


def get_dataloaders(crop_size=512):
    train_dataset = NUSDataset(Config.data_root, mode='training', crop_size=crop_size)
    test_dataset = NUSDataset(Config.data_root, mode='testing', crop_size=crop_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return train_loader, test_loader
