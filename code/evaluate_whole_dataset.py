import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ================= 配置区域 =================
CONFIG = {
    'image_dir': '/mnt/e/TongGuanChao科研/ACG-完整版-区分左右眼',
    'csv_path': '/mnt/e/TongGuanChao科研/data.csv',
    'model_path': '/mnt/e/TongGuanChao科研/best_model.pth',
    'image_size': 256,
    'batch_size': 32,
    'num_points': 61,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 绘图字体设置 (PDF 对字体要求较高，SimHei 用于显示中文)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设定 PDF 字体为 Type 42 (TrueType)，避免某些期刊投稿系统报错
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 尝试导入 Mamba
try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


# ================= 数据集类 (复用) =================
class RecursiveACGDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.image_map = {}

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_map[os.path.splitext(file)[0]] = os.path.join(root, file)

        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except:
            df = pd.read_csv(csv_file, encoding='gbk')

        label_cols = [str(i) for i in range(CONFIG['num_points'])]
        self.valid_samples = []

        for idx, row in df.iterrows():
            if row[label_cols].isnull().any(): continue
            img_name = str(row['眼底照1'])
            if img_name == 'nan': continue
            stem = os.path.splitext(img_name)[0]
            if stem in self.image_map:
                self.valid_samples.append({
                    'path': self.image_map[stem],
                    'labels': row[label_cols].values.astype(np.float32)
                })

        print(f"共加载 {len(self.valid_samples)} 个样本用于整体评估。")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        try:
            img_np = np.fromfile(sample['path'], dtype=np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = np.zeros((CONFIG['image_size'], CONFIG['image_size'], 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(sample['labels'])


# ================= 模型定义 =================
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
        x = x.flatten(2).transpose(1, 2)
        if HAS_MAMBA: x = self.mamba(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return self.head(x)


# ================= 主评估逻辑 =================
def evaluate_all():
    print(f"设备: {CONFIG['device']}")

    val_transform = A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    dataset = RecursiveACGDataset(CONFIG['csv_path'], CONFIG['image_dir'], transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    model = ConvMambaRegressor(output_dim=CONFIG['num_points']).to(CONFIG['device'])
    if os.path.exists(CONFIG['model_path']):
        state_dict = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
        model.load_state_dict(state_dict)
    else:
        print("❌ 找不到模型文件！")
        return

    model.eval()

    all_preds = []
    all_targets = []

    print(">>> 正在对全量数据进行预测...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(CONFIG['device'])
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    flat_preds = all_preds.flatten()
    flat_targets = all_targets.flatten()

    mask = flat_targets != -1
    clean_preds = flat_preds[mask]
    clean_targets = flat_targets[mask]

    print(f"\n有效数据点总数: {len(clean_targets)}")

    mae = mean_absolute_error(clean_targets, clean_preds)
    rmse = np.sqrt(mean_squared_error(clean_targets, clean_preds))
    r2 = r2_score(clean_targets, clean_preds)

    print("=" * 30)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print("=" * 30)

    # === 开始绘图 (PDF版) ===
    plt.figure(figsize=(18, 6))

    # 1. 散点图
    plt.subplot(1, 3, 1)
    if len(clean_targets) > 50000:
        idx = np.random.choice(len(clean_targets), 50000, replace=False)
        plt.scatter(clean_targets[idx], clean_preds[idx], alpha=0.1, s=2, c='blue', edgecolors='none')
    else:
        plt.scatter(clean_targets, clean_preds, alpha=0.1, s=2, c='blue', edgecolors='none')

    min_val = min(clean_targets.min(), clean_preds.min())
    max_val = max(clean_targets.max(), clean_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.title(f"True vs Predicted\n(MAE={mae:.2f}, R²={r2:.2f})")
    plt.xlabel("True Sensitivity (dB)")
    plt.ylabel("Predicted Sensitivity (dB)")
    plt.grid(True, alpha=0.3)

    # 2. 误差直方图
    plt.subplot(1, 3, 2)
    errors = clean_preds - clean_targets
    plt.hist(errors, bins=50, color='purple', alpha=0.7, density=True)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error (Pred - True) [dB]")
    plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3)

    # 3. Bland-Altman
    plt.subplot(1, 3, 3)
    means = (clean_targets + clean_preds) / 2
    diffs = clean_preds - clean_targets
    if len(means) > 50000:
        idx = np.random.choice(len(means), 50000, replace=False)
        plt.scatter(means[idx], diffs[idx], alpha=0.1, s=2, c='green', edgecolors='none')
    else:
        plt.scatter(means, diffs, alpha=0.1, s=2, c='green', edgecolors='none')

    plt.axhline(diffs.mean(), color='red', linestyle='-')
    plt.axhline(diffs.mean() + 1.96 * diffs.std(), color='red', linestyle='--')
    plt.axhline(diffs.mean() - 1.96 * diffs.std(), color='red', linestyle='--')
    plt.title("Bland-Altman Plot")
    plt.xlabel("Mean (dB)")
    plt.ylabel("Diff (Pred - True) [dB]")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存为 PDF
    save_path = '../overall_evaluation.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"\n✅ PDF 文件已保存: {save_path}")


if __name__ == '__main__':
    evaluate_all()