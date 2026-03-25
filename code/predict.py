import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# ================= 配置区域 =================
# 1. 模型路径 (已修复：去掉了末尾的逗号)
MODEL_PATH = '/mnt/e/TongGuanChao科研/best_model.pth'

# 2. 测试图片路径
TEST_IMAGE_PATH = '/mnt/e/TongGuanChao科研/ACG-完整版-区分左右眼/00501576张顺钗/左眼/20180907085508_31000901_13535.jpg'

# 3. 默认点数 (代码会自动尝试从权重中获取真实点数，如果获取失败则用这个)
DEFAULT_NUM_POINTS = 61
# ===========================================

# 尝试导入 Mamba
try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print(">>> 未检测到 mamba_ssm，将在无 Mamba 模式下运行。")


# === 模型定义 ===
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
            # 注意：这里我们先不定义最后一层，等确定了 output_dim 再加
            nn.LazyLinear(output_dim) if hasattr(nn, 'LazyLinear') else nn.Identity()
        )
        # 为了兼容性，我们在 forward 里动态处理，或者在加载时重建 head

    # 重新定义一个简单的构建函数，避免 LazyLinear 的兼容性问题
    def build_head(self, d_model, output_dim):
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


def predict():
    # 1. 检查图片
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 错误：找不到测试图片 -> {TEST_IMAGE_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 2. 加载权重
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：找不到模型文件 -> {MODEL_PATH}")
        return

    print(f"正在加载权重: {MODEL_PATH}")
    # map_location 确保在 CPU 上也能加载 CUDA 训练的模型
    state_dict = torch.load(MODEL_PATH, map_location=device)

    # 3. 自动分析 output_dim
    # 我们查看 state_dict 中最后一层 (head.4.bias 或 head.5.bias) 的形状
    # 通常 key 是 'head.4.weight' 或 'head.4.bias'
    keys = list(state_dict.keys())
    last_layer_key = keys[-1]  # 获取最后一层的名字

    actual_num_points = DEFAULT_NUM_POINTS
    if 'head' in last_layer_key and ('weight' in last_layer_key or 'bias' in last_layer_key):
        shape = state_dict[last_layer_key].shape
        actual_num_points = shape[0]
        print(f"✅ 从权重文件中检测到输出维度为: {actual_num_points}")
    else:
        print(f"⚠️ 无法自动检测维度，使用默认值: {actual_num_points}")

    # 4. 初始化模型
    model = ConvMambaRegressor(output_dim=actual_num_points).to(device)
    # 手动构建 head 以匹配维度
    model.build_head(d_model=128, output_dim=actual_num_points)
    model.to(device)

    # 加载参数
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"\n❌ 模型加载失败！这通常是因为 num_points 对不上。")
        print(f"错误详情: {e}")
        return

    model.eval()

    # 5. 读取处理图片 (处理中文路径)
    img_np = np.fromfile(TEST_IMAGE_PATH, dtype=np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if image is None:
        print("❌ 图片读取失败，文件可能已损坏。")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    img_tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    # 6. 预测
    print(">>> 正在进行预测...")
    with torch.no_grad():
        output = model(img_tensor)
        vals = output.cpu().numpy()[0]

    # 7. 打印结果
    print("\n" + "=" * 40)
    print(f"图片: {os.path.basename(TEST_IMAGE_PATH)}")
    print("=" * 40)
    print(f"预测出的 {len(vals)} 个视野光敏感度数值 (dB):")
    print("-" * 40)

    # 更加美观的打印
    for i in range(0, len(vals), 10):
        print("  ".join([f"{v:5.2f}" for v in vals[i:i + 10]]))

    print("=" * 40)
    print("预测完成！")


if __name__ == '__main__':
    predict()