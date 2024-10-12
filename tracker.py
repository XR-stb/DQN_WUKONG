import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from log import log

class RewardTracker:
    def __init__(self, train_data_dir):
        self.train_data_dir = train_data_dir
        self.episode_rewards = []  # 保存每局的总奖励分
        self.total_rewards = []  # 保存每个 action 的累积奖励
        self.boss_healths = []  # 保存每局结束时 Boss 的血量
        self.episode_num = 0  # 记录当前的局数

        # 创建目录
        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)

        # 初始化 Boss 血量数据文件路径
        self.boss_healths_file_path = os.path.join(train_data_dir, 'boss_healths.csv')

        # 如果文件不存在，则创建文件并写入表头
        if not os.path.exists(self.boss_healths_file_path):
            pd.DataFrame(columns=['Episode', 'Boss Health']).to_csv(self.boss_healths_file_path, index=False)

    def add_reward(self, reward):
        """在每个 action 后调用，保存累积的奖励值。"""
        self.total_rewards.append(reward)

    def end_episode(self, boss_health):
        """每局结束时调用，保存总奖励、Boss血量并重置累积奖励。"""
        total_reward = sum(self.total_rewards)
        self.episode_rewards.append(total_reward)
        self.boss_healths.append(boss_health)
        self.episode_num += 1

        # 将当前局的奖励数据保存到文件
        self.save_episode_data()

        # 清空当前局的奖励数据
        self.total_rewards.clear()

        # 每局结束时保存 Boss 血量数据
        self.save_boss_health_data()

    def save_episode_data(self):
        """将每局的奖励数据保存到 CSV 文件，文件名使用时间戳避免覆盖。"""
        episode_df = pd.DataFrame({
            'Action': range(len(self.total_rewards)),
            'Reward': self.total_rewards
        })
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 使用时间戳
        episode_file_path = os.path.join(self.train_data_dir, f'episode_{self.episode_num}_rewards_{timestamp}.csv')
        episode_df.to_csv(episode_file_path, index=False)
        log.debug(f"奖励数据已保存到 {episode_file_path}")

    def save_boss_health_data(self):
        """保存 Boss 血量数据到同一个 CSV 文件中。"""
        # 创建包含当前局数和对应 Boss 血量的数据帧
        current_data = pd.DataFrame({
            'Episode': [self.episode_num],
            'Boss Health': [self.boss_healths[-1]]  # 只保存当前局的 Boss 血量
        })

        # 追加写入文件
        current_data.to_csv(self.boss_healths_file_path, mode='a', header=False, index=False)
        log.debug(f"Boss 血量数据已保存到 {self.boss_healths_file_path}")

    def save_overall_data(self):
        """保存所有局的总奖励数据，文件名使用时间戳避免覆盖。"""
        overall_df = pd.DataFrame({'Episode': range(1, len(self.episode_rewards) + 1),
                                   'Reward': self.episode_rewards})
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 使用时间戳
        overall_file_path = os.path.join(self.train_data_dir, f'overall_rewards_{timestamp}.csv')
        overall_df.to_csv(overall_file_path, mode='a', index=False)
        log.debug(f"整体奖励数据已保存到 {overall_file_path}")

        self.episode_rewards.clear()

