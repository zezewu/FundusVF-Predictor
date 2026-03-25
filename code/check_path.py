import os
import pandas as pd

# ================= 配置 =================
# 请确保这里和你的训练代码配置一模一样
CSV_PATH = r'E:\TongGuanChao科研\data.csv'
IMG_DIR = r'/ACG-完整版-区分左右眼'


# ========================================

def diagnose():
    print("------- 1. 检查文件夹 -------")
    if not os.path.exists(IMG_DIR):
        print(f"❌ 错误：文件夹不存在！\n{IMG_DIR}")
        return

    files = os.listdir(IMG_DIR)
    images = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif'))]
    print(f"文件夹里共有 {len(images)} 张图片。")
    print(f"前 5 张图片的文件名是：{images[:5]}")

    print("\n------- 2. 检查 CSV 文件 -------")
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误：CSV不存在！\n{CSV_PATH}")
        return

    # 尝试不同编码读取
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
    except:
        df = pd.read_csv(CSV_PATH, encoding='gbk')

    print(f"CSV 共有 {len(df)} 行。")
    # 假设第一列是文件名
    first_col = df.columns[0]
    print(f"CSV 第一列的列名是: [{first_col}]")

    # 取出前 5 个非空的 CSV 文件名
    csv_names = df[first_col].dropna().astype(str).head(5).tolist()
    print(f"CSV 里前 5 个文件名是：{csv_names}")

    print("\n------- 3. 模拟匹配测试 -------")
    for name in csv_names:
        # 模拟代码中的拼接逻辑
        clean_name = name.strip()
        # 如果CSV里没后缀，代码会自动加 .jpg
        if not clean_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            predicted_path = os.path.join(IMG_DIR, clean_name + ".jpg")
            suffix_status = "(代码自动添加了 .jpg)"
        else:
            predicted_path = os.path.join(IMG_DIR, clean_name)
            suffix_status = "(CSV自带后缀)"

        exists = os.path.exists(predicted_path)
        status = "✅ 成功匹配" if exists else "❌ 匹配失败"
        print(
            f"CSV名字: [{clean_name}] -> 尝试路径: ...\\{os.path.basename(predicted_path)} {suffix_status} -> {status}")


if __name__ == '__main__':
    diagnose()