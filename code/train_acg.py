import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import timm

try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
    print(">>> [环境] Mamba 启用")
except ImportError:
    HAS_MAMBA = False
    print(">>> [环境] 降级为 SE 模式")

# ==========================================
# 1. 配置更新 (支持多数据集与 ResNet)
# ==========================================
CONFIG = {
    'datasets': [
        {
            'image_dir': '/mnt/e/TongGuanChao科研/ACG-完整版-区分左右眼',
            'csv_path': '/mnt/e/TongGuanChao科研/data_ACG.csv'
        },
        {
            'image_dir': '/mnt/e/TongGuanChao科研/OAG-完整版-区分左右眼',
            'csv_path': '/mnt/e/TongGuanChao科研/data_OCG.csv'
        }
    ],
    'output_dir': '/mnt/e/TongGuanChao科研/results_wingloss',

    'backbone': 'resnet50',
    'd_model': 384,
    'mamba_layers': 2,

    'image_size': 384,
    'batch_size': 14,
    'learning_rate': 4e-4,
    'epochs': 60,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixup_alpha': 0.2
}


# ==========================================
# 2. Wing Loss
# ==========================================
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        mask = (target != -1).float()
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()

        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss = torch.where(
            delta_y < self.omega,
            self.omega * torch.log(1 + delta_y / self.epsilon),
            delta_y - C
        )

        return (loss * mask).sum() / (mask.sum() + 1e-8)


# ==========================================
# 3. Mixup 工具
# ==========================================
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==========================================
# 4. 数据处理
# ==========================================
def process_fundus_image(img_path, target_size):
    try:
        img_np = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if image is None: return np.zeros((target_size, target_size, 3), dtype=np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray > 7
        if mask.sum() > 0:
            coords = np.argwhere(mask)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1
            image = image[x0:x1, y0:y1]

        image = cv2.resize(image, (target_size, target_size))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), target_size / 30), -4, 128)
        return image
    except Exception:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)


class ACGDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = process_fundus_image(item['path'], CONFIG['image_size'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image=image)['image']
        return image, torch.tensor(item['labels'])


# ==========================================
# 5. 模型架构 (ResNet + Mamba Hybrid)
# ==========================================
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class MambaAdapter(nn.Module):
    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + res


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, l, c = x.size()
        y = x.transpose(1, 2)
        y = self.fc(self.avg_pool(y).view(b, c)).view(b, 1, c)
        return x * y


class CNNMambaHybrid(nn.Module):
    def __init__(self, backbone_name, output_dim, d_model=512):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        dummy = torch.randn(1, 3, 256, 256)
        in_channels = self.backbone(dummy)[-1].shape[1]

        self.coord_att = CoordAtt(in_channels, in_channels)
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=1)

        self.layers = nn.ModuleList()
        for _ in range(CONFIG['mamba_layers']):
            self.layers.append(MambaAdapter(d_model) if HAS_MAMBA else SEBlock(d_model))

        self.norm_final = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]
        feats = self.coord_att(feats)
        x = self.proj(feats)
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers: x = layer(x)
        x = self.norm_final(x).mean(dim=1)
        return self.head(x)


# ==========================================
# 6. 训练核心 & 验证评估 (修改部分)
# ==========================================
def train_one_epoch(loader, model, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc="Train")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(images, labels, CONFIG['mixup_alpha'], device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


def evaluate(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_list, targets_list = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val"):
            images, labels = images.to(device), labels.to(device)
            out = model(images)

            loss = criterion(out, labels)
            running_loss += loss.item()

            # 收集预测值和真实值用于计算所有指标
            preds_list.append(out.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    # 将批次数据拼接
    preds = np.vstack(preds_list)
    targets = np.vstack(targets_list)

    # 过滤无效标签 (-1)
    mask = targets != -1
    flat_p, flat_t = preds[mask], targets[mask]

    # 计算 MAE, RMSE, R-Squared
    val_mae = mean_absolute_error(flat_t, flat_p)
    val_rmse = np.sqrt(mean_squared_error(flat_t, flat_p))
    val_r2 = r2_score(flat_t, flat_p)

    return running_loss / len(loader), val_mae, val_rmse, val_r2


# ==========================================
# 7. 辅助函数 & 报告生成 (修改部分)
# ==========================================
def parse_data_map(csv_file, root_dir):
    print(f"\n>>> 扫描数据集: {root_dir}")
    image_map = {}
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
                image_map[os.path.splitext(file)[0]] = os.path.join(root, file)

    df = pd.read_csv(csv_file)
    label_cols = [str(i) for i in range(61)]
    available_cols = [c for c in label_cols if c in df.columns]

    valid_samples = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="匹配中"):
        if row[available_cols].isnull().any(): continue
        img_filename = str(row['眼底照1'])
        if img_filename == 'nan': continue
        img_stem = os.path.splitext(img_filename)[0]
        if img_stem in image_map:
            valid_samples.append({'path': image_map[img_stem], 'labels': row[available_cols].values.astype(np.float32)})
    return valid_samples, len(available_cols)


def generate_pdf_report(model, dataloader, device, output_dir):
    print("\n>>> 生成最终报告...")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Inference"):
            images = images.to(device)
            out = model(images)
            preds_list.append(out.cpu().numpy())
            targets_list.append(labels.numpy())

    preds = np.vstack(preds_list)
    targets = np.vstack(targets_list)
    mask = targets != -1
    flat_p, flat_t = preds[mask], targets[mask]

    # 计算最终指标
    mae = mean_absolute_error(flat_t, flat_p)
    mse = mean_squared_error(flat_t, flat_p)
    rmse = np.sqrt(mse)
    r2 = r2_score(flat_t, flat_p)

    # 打印最终结果到控制台
    print(f"\n================ 最终测试结果 ================")
    print(f"✅ MAE (Mean Absolute Error): {mae:.4f}")
    print(f"✅ RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"✅ R-Squared (R2 Score): {r2:.4f}")
    print(f"============================================\n")

    pdf_path = os.path.join(output_dir, 'WingLoss_Report.pdf')
    pdf = pdf_backend.PdfPages(pdf_path)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    txt = (f"ResNet+Mamba + WingLoss Report\n=========================\n\n"
           f"Model: {CONFIG['backbone']} (384px) + Mamba\n"
           f"Scheduler: Cosine Annealing Warm Restarts\n"
           f"Samples: {len(dataloader.dataset)}\n\n"
           f"MAE:  {mae:.4f} dB\n"
           f"RMSE: {rmse:.4f} dB\n"
           f"R2:   {r2:.4f}")
    ax.text(0.1, 0.4, txt, fontsize=14, family='monospace')
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    idx = np.random.choice(len(flat_t), min(5000, len(flat_t)), replace=False)
    sns.scatterplot(x=flat_t[idx], y=flat_p[idx], alpha=0.1, color='purple', ax=ax)
    mi, ma = min(flat_t.min(), flat_p.min()), max(flat_t.max(), flat_p.max())
    ax.plot([mi, ma], [mi, ma], 'r--')
    ax.set_title(f"Truth vs Pred")
    ax.set_xlabel("Truth")
    ax.set_ylabel("Pred")
    pdf.savefig(fig)
    plt.close()

    pdf.close()
    print(f"✅ 报告已保存至: {pdf_path}")


# ==========================================
# 8. 主程序 (修改部分)
# ==========================================
def main():
    if not os.path.exists(CONFIG['output_dir']): os.makedirs(CONFIG['output_dir'])

    all_samples = []
    num_points = 0
    for ds_info in CONFIG['datasets']:
        samples, pts = parse_data_map(ds_info['csv_path'], ds_info['image_dir'])
        if samples:
            all_samples.extend(samples)
            num_points = pts

    print(f"\n>>> 总计匹配成功有效样本数: {len(all_samples)}")
    if not all_samples: return

    train_s, val_s = train_test_split(all_samples, test_size=0.1, random_state=42)

    train_tf = A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_loader = DataLoader(ACGDataset(train_s, train_tf), batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(ACGDataset(val_s, val_tf), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    model = CNNMambaHybrid(CONFIG['backbone'], num_points, CONFIG['d_model']).to(CONFIG['device'])

    criterion = WingLoss(omega=10, epsilon=2)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_mae = float('inf')

    print(f"\n>>> 启动训练 ({CONFIG['backbone']} + 384px + WingLoss + CosineAnnealing)")

    for epoch in range(CONFIG['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion, CONFIG['device'])

        # 接收新增的 RMSE 和 R2
        val_loss, val_mae, val_rmse, val_r2 = evaluate(val_loader, model, criterion, CONFIG['device'])

        scheduler.step()

        # 在控制台打印出全部指标
        print(
            f"Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val R2: {val_r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], "best_wing_model.pth"))
            print(f"★ 最佳模型保存 (MAE: {val_mae:.4f})")

    model.load_state_dict(torch.load(os.path.join(CONFIG['output_dir'], "best_wing_model.pth")))
    generate_pdf_report(model, val_loader, CONFIG['device'], CONFIG['output_dir'])


if __name__ == '__main__':
    main()