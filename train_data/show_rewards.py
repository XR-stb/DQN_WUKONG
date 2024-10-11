import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_data_from_directory(directory, prefix):
    """从指定目录加载以指定前缀命名的 CSV 数据文件，并累加每个文件的 Reward 列的总和。"""
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv') and f.startswith(prefix)]
    rewards = []
    for file in all_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path)
            if 'Reward' in df.columns:
                total_reward = df['Reward'].sum()  # 累加整个文件中的 Reward 值
                rewards.append(total_reward)
                print(f"读取文件: {file_path}, 总 Reward 值: {total_reward}")
            else:
                print(f"警告: 文件中不包含 'Reward' 列")
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
    return rewards

def plot_total_rewards(rewards, save_path):
    """绘制每局结束时的总奖励分曲线和每10局的平均奖励曲线。"""
    if rewards:  # 确保有数据可以绘制
        plt.figure(figsize=(10, 6))
        
        # 绘制每局的总奖励曲线
        plt.plot(range(1, len(rewards) + 1), rewards, marker='o', label='Total Reward per Episode')
        
        # 计算并绘制每10局的平均奖励曲线
        if len(rewards) >= 10:
            averages = [sum(rewards[i:i+10]) / 10 for i in range(0, len(rewards), 10)]
            average_episodes = [i + 1 for i in range(0, len(rewards), 10)]
            plt.plot(average_episodes, averages, marker='x', linestyle='--', color='red', label='Average Reward per 10 Episodes')

        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Rewards per Episode and Average Rewards per 10 Episodes')
        plt.grid(True)
        plt.legend()  # 添加图例
        plt.savefig(save_path)
        plt.close()
        print(f"奖励曲线图已保存到 {save_path}")
    else:
        print("没有奖励数据可绘制。")

def main(directory, prefix):
    """主函数，生成奖励曲线图和数据文件。"""
    rewards = load_data_from_directory(directory, prefix)
    
    # 获取当前目录的父目录
    parent_directory = os.path.dirname(directory)
    
    # 在与 data 同级的目录下创建 'image' 文件夹
    image_directory = os.path.join(parent_directory, 'image')
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
    
    if rewards:
        # 生成奖励曲线图
        plot_total_rewards(rewards, os.path.join(image_directory, f'total_rewards_curve_{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    else:
        print(f"没有找到以 '{prefix}' 前缀命名的有效奖励数据文件。")

if __name__ == "__main__":
    # 替换为你保存数据的目录
    data_directory = './data'
    
    # 以所有局reward绘制曲线
    prefix_type = 'episode_'  # 或 'overall_rewards' 以每10局的reward和绘制曲线
    
    main(data_directory, prefix_type)
