import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_boss_health(csv_file_path):
    """绘制 Boss 血量曲线图。"""
    if not os.path.isfile(csv_file_path):
        print(f"CSV 文件 {csv_file_path} 不存在。")
        return

    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    if 'Episode' not in df.columns or 'Boss Health' not in df.columns:
        print("CSV 文件中缺少 'Episode' 或 'Boss Health' 列。")
        return

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df['Boss Health'], marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Boss Health')
    plt.title('Boss Health per Episode')
    plt.grid(True)

    # 设置保存路径为与data目录同级的image目录
    image_dir = os.path.join(os.path.dirname(csv_file_path), '../image')
    os.makedirs(image_dir, exist_ok=True)  # 如果image目录不存在，则创建它

    # 保存绘图结果
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    plot_file_path = os.path.join(image_dir, f'boss_health_plot_{timestamp}.png')
    plt.savefig(plot_file_path)
    plt.close()

    print(f"Boss 血量曲线图已保存到 {plot_file_path}")

# 使用示例
if __name__ == "__main__":
    csv_file_path = './data/boss_healths.csv'  # 替换为实际文件路径
    plot_boss_health(csv_file_path)
