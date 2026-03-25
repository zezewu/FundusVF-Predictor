import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time

# ==========================================
# 1. 核心配置
# ==========================================
CONFIG = {
    # 你的图片总目录 (代码会自动搜索里面的所有子文件夹)
    'image_dir': r'E:\TongGuanChao科研\ACG-完整版-区分左右眼',

    # 你的 CSV 文件
    'csv_path': r'E:\TongGuanChao科研\data.csv',

    # 结果保存位置
    'output_dir': r'E:\TongGuanChao科研',

    'image_size': 256,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 尝试导入 Mamba
try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print(">>> 未检测到 mamba_ssm，切换为纯 CNN 模式。")


# ==========================================
# 2. 智能数据集类 (精准列匹配版)
# ==========================================
class ACGDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform

        print(f"\n>>> [步骤1] 正在建立图片索引 (递归扫描)...")
        # 1. 建立图片索引地图：文件名 -> 完整路径
        self.image_map = {}
        found_images = 0
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    # 无论是 "abc.jpg" 还是 "abc" 都能匹配
                    self.image_map[file] = os.path.join(root, file)
                    found_images += 1

        print(f"    - 硬盘中扫描到图片: {found_images} 张")
        if found_images == 0:
            raise ValueError(f"错误：在 {root_dir} 下没找到任何图片！")

        # 2. 读取 CSV
        print(f">>> [步骤2] 读取 CSV 并清洗数据...")
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except:
            df = pd.read_csv(csv_file, encoding='gbk')

        # === 关键修正：指定列名 ===
        target_filename_col = '眼底照1'  # 你的图片文件名列

        # 自动寻找数字列作为标签 (0, 1, 2 ... 60)
        self.label_cols = [col for col in df.columns if str(col).isdigit()]
        # 按数字大小排序，确保顺序对 (0, 1, 2...)
        self.label_cols.sort(key=lambda x: int(x))

        print(f"    - 锁定文件名列: [{target_filename_col}]")
        print(f"    - 锁定标签数据列: {self.label_cols[0]} ~ {self.label_cols[-1]} (共 {len(self.label_cols)} 个点)")

        self.valid_samples = []
        missing_img_count = 0
        missing_label_count = 0

        # 遍历 CSV
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="匹配数据"):
            # 1. 获取文件名
            fname = str(row[target_filename_col]).strip()
            if fname == 'nan' or fname == '':
                continue  # 这一行没写文件名，跳过

            # 2. 尝试在地图里找图
            # CSV里自带后缀 .jpg，直接找
            if fname in self.image_map:
                full_path = self.image_map[fname]
            # 如果CSV里没后缀，尝试加后缀找
            elif (fname + '.jpg') in self.image_map:
                full_path = self.image_map[fname + '.jpg']
            else:
                missing_img_count += 1
                continue  # 找不到图，跳过

            # 3. 获取标签数值
            # 取出 '0'~'60' 列的值
            vals = row[self.label_cols].values

            # 检查标签是否全是空值 (NaN) -> 如果全是空，这行数据没法训练
            # 注意：pandas 读取空值通常是 float 类型的 nan
            if pd.isna(vals).all():
                missing_label_count += 1
                continue  # 标签全是空的，跳过

            # 将剩余的 NaN 填补为 0，并转为 float32
            label_tensor = np.nan_to_num(list(vals), nan=0.0).astype(np.float32)

            self.valid_samples.append({
                'path': full_path,
                'labels': label_tensor
            })

        print(f"\n>>> [清洗报告]")
        print(f"    - CSV总行数: {len(df)}")
        print(f"    - 图片缺失: {missing_img_count} (CSV有记录但没找到图)")
        print(f"    - 标签缺失: {missing_label_count} (有图但没有视野数据，已剔除)")
        print(f"    - ✅ 有效训练样本: {len(self.valid_samples)}")

        if len(self.valid_samples) == 0:
            raise ValueError("严重错误：有效样本为 0！请检查 CSV 中是否所有行的 '0'~'60' 列都是空的？")

        self.num_points = len(self.valid_samples[0]['labels'])

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        img_path = sample['path']
        labels = sample['labels']

        try:
            # 中文路径支持
            img_np = np.fromfile(img_path, dtype=np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            image = np.zeros((CONFIG['image_size'], CONFIG['image_size'], 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(labels)


# 增强
train_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
    A.CLAHE(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ==========================================
# 3. 模型定义
# ==========================================
class ConvMambaRegressor(nn.Module):
    def __init__(self, output_dim, d_model=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, d_model, 3, 1, 1), nn.BatchNorm2d(d_model), nn.ReLU(),
        )
        if HAS_MAMBA:
            self.mamba = nn.Sequential(
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            )
        else:
            self.mamba = nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.stem(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if HAS_MAMBA: x = self.mamba(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return self.head(x)


# ==========================================
# 4. 主程序
# ==========================================
def train():
    if not os.path.exists(CONFIG['csv_path']):
        print(f"错误: 找不到 CSV 文件")
        return

    print(f"设备: {CONFIG['device']}")

    # 1. 加载数据
    try:
        dataset = ACGDataset(CONFIG['csv_path'], CONFIG['image_dir'], transform=train_transform)
    except ValueError as e:
        print(e)
        return

    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

    # 2. 模型
    model = ConvMambaRegressor(output_dim=dataset.num_points).to(CONFIG['device'])
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

    print(f"开始训练... (共 {CONFIG['epochs']} 轮)")

    for epoch in range(CONFIG['epochs']):
        model.train()
        loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{CONFIG["epochs"]}')
        running_loss = 0.0

        for images, labels in loop:
            images = images.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.2f}")

        # 简单保存策略
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(CONFIG['output_dir'], f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存: {save_path}")


if __name__ == '__main__':
    train()