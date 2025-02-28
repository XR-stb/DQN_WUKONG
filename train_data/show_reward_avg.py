import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def calculate_running_average(directory, prefix):
    """计算累积平均奖励"""
    all_files = [
        f for f in os.listdir(directory) if f.endswith(".csv") and f.startswith(prefix)
    ]
    # 按时间戳排序文件
    all_files.sort(key=lambda x: x.split("_")[-2] + x.split("_")[-1])

    total_reward = 0
    episode_rewards = []
    running_averages = []

    for idx, file in enumerate(all_files, 1):
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path)
            if "Reward" in df.columns:
                # 计算这一局的总奖励
                episode_reward = df["Reward"].sum()
                if episode_reward > -10000:  # 过滤异常值
                    total_reward += episode_reward
                    episode_rewards.append(episode_reward)
                    # 计算到目前为止的平均奖励
                    running_avg = total_reward / idx
                    running_averages.append(running_avg)

                    print(
                        f"Episode {idx}: Total Reward = {episode_reward:.2f}, "
                        f"Running Average = {running_avg:.2f}"
                    )

        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
            continue

    return episode_rewards, running_averages


def plot_rewards(episode_rewards, running_averages, save_path):
    """绘制奖励曲线"""
    plt.figure(figsize=(12, 6))

    # 绘制每局的奖励
    episodes = range(1, len(episode_rewards) + 1)
    plt.plot(episodes, episode_rewards, "b-", alpha=0.3, label="Episode Reward")

    # 绘制累积平均奖励
    plt.plot(episodes, running_averages, "r-", linewidth=2, label="Running Average")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards and Running Average")
    plt.grid(True)
    plt.legend()

    # 保存图片
    plt.savefig(save_path)
    plt.close()

    print(f"奖励曲线图已保存到 {save_path}")


def main():
    data_directory = "./data"  # 数据目录
    prefix = "episode_"  # 文件前缀

    # 获取当前目录的父目录
    parent_directory = os.path.dirname(os.path.dirname(data_directory))

    # 创建image目录
    image_directory = os.path.join(parent_directory, "image")
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    # 计算奖励
    episode_rewards, running_averages = calculate_running_average(
        data_directory, prefix
    )

    if episode_rewards:
        # 生成图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_rewards(
            episode_rewards,
            running_averages,
            os.path.join(image_directory, f"running_average_{timestamp}.png"),
        )

        # 保存数据到CSV
        # data = {
        #     "Episode": range(1, len(episode_rewards) + 1),
        #     "Reward": episode_rewards,
        #     "Running_Average": running_averages,
        # }
        # df = pd.DataFrame(data)
        # df.to_csv(
        #     os.path.join(image_directory, f"running_average_{timestamp}.csv"),
        #     index=False,
        # )

    else:
        print("没有找到有效的奖励数据")


if __name__ == "__main__":
    main()
