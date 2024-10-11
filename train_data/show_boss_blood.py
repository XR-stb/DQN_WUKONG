import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_boss_health(csv_file_path):
    """绘制 Boss 血量曲线图及每10局的平均血量曲线。"""
    if not os.path.isfile(csv_file_path):
        print(f"CSV 文件 {csv_file_path} 不存在。")
        return

    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    if 'Episode' not in df.columns or 'Boss Health' not in df.columns:
        print("CSV 文件中缺少 'Episode' 或 'Boss Health' 列。")
        return

    # 绘制 Boss 血量曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df['Boss Health'], marker='o', linestyle='-', label='Boss Health per Episode')

    # 计算每10局的平均 Boss Health
    if len(df) >= 10:
        averages = [df['Boss Health'][i:i+10].mean() for i in range(0, len(df), 10)]
        average_episodes = [df['Episode'].iloc[i] for i in range(0, len(df), 10)]
        plt.plot(average_episodes, averages, marker='x', linestyle='--', color='red', label='Average Boss Health per 10 Episodes')

    plt.xlabel('Episode')
    plt.ylabel('Boss Health')
    plt.title('Boss Health per Episode and Average Boss Health per 10 Episodes')
    plt.grid(True)
    plt.legend()  # 添加图例

    # 设置保存路径为与 data 目录同级的 image 目录
    image_dir = os.path.join(os.path.dirname(csv_file_path), '../image')
    os.makedirs(image_dir, exist_ok=True)  # 如果 image 目录不存在，则创建它

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
