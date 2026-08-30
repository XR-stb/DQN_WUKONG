"""
实时训练仪表板 - 训练过程中在另一个终端运行此脚本即可实时查看训练进展

使用方法:
    python train_data/live_dashboard.py

功能:
    - 每5秒自动刷新一次数据
    - 12个子图实时展示训练核心指标（3行4列布局）
    - 支持训练中途启动，自动加载所有历史数据
    - 覆盖维度：奖励、Boss血量、胜率、命中率、DPS、闪避、
      受伤次数、存活时间、伤害统计、动作多样性、技能使用、综合评分
"""

import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import numpy as np

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 从配置文件加载参数 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'dashboard_conf.yaml')

def _load_config():
    """加载仪表板配置"""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {'data': {'train_data_dir': 'train_data/data'}, 'dashboard': {'refresh_interval': 5000}}

_config = _load_config()

# 数据文件路径 (从配置读取)
CSV_PATH = os.path.join(PROJECT_ROOT, _config['data']['train_data_dir'], 'training_log.csv')

# 刷新间隔（毫秒，从配置读取）
REFRESH_INTERVAL = _config['dashboard']['refresh_interval']


def load_data():
    """加载训练日志CSV"""
    if not os.path.exists(CSV_PATH):
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def calc_moving_avg(series, window):
    """计算移动平均，不足window时用已有数据的均值"""
    return series.rolling(window=window, min_periods=1).mean()


def safe_col(df, col, default=0):
    """安全获取DataFrame列，不存在则返回默认值填充的Series"""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index)


class LiveDashboard:
    """实时训练仪表板 — 3行4列 12宫格布局"""

    def __init__(self):
        self.fig, self.axes = plt.subplots(3, 4, figsize=(22, 13))
        self.fig.suptitle('🎮 悟空DQN训练实时仪表板', fontsize=16, fontweight='bold')
        self.fig.set_facecolor('#1a1a2e')

        # 设置所有子图的暗色主题
        for ax_row in self.axes:
            for ax in ax_row:
                ax.set_facecolor('#16213e')
                ax.tick_params(colors='#e0e0e0', labelsize=8)
                ax.xaxis.label.set_color('#e0e0e0')
                ax.yaxis.label.set_color('#e0e0e0')
                ax.title.set_color('#ffffff')
                for spine in ax.spines.values():
                    spine.set_color('#3a3a5c')
                ax.grid(True, alpha=0.2, color='#ffffff')

        self.fig.patch.set_facecolor('#1a1a2e')
        plt.tight_layout(rect=[0, 0.02, 1, 0.94])

        self.last_episode = 0

    def update(self, frame):
        """每帧更新数据"""
        df = load_data()
        if df is None or len(df) == 0:
            return

        n = len(df)
        episodes = df['episode']

        # 如果数据没有变化则跳过绘制
        if n == self.last_episode:
            return
        self.last_episode = n

        # 更新标题显示当前状态
        latest = df.iloc[-1]
        hit_rate_val = safe_col(df, 'hit_rate').iloc[-1] if 'hit_rate' in df.columns else 0
        dps_val = safe_col(df, 'dps').iloc[-1] if 'dps' in df.columns else 0
        self.fig.suptitle(
            f'🎮 悟空DQN训练实时仪表板  |  '
            f'回合: {int(latest["episode"])}  |  '
            f'最新奖励: {latest["total_reward"]:+.0f}  |  '
            f'命中率: {hit_rate_val:.0%}  |  '
            f'DPS: {dps_val:.1f}  |  '
            f'近10局胜率: {safe_col(df, "win_rate_10").iloc[-1]:.0%}',
            fontsize=13, fontweight='bold', color='#ffffff'
        )

        # ============ 子图 [0,0]：回合总奖励 ============
        ax = self.axes[0][0]
        ax.clear()
        self._style_ax(ax)

        rewards = df['total_reward']
        ax.plot(episodes, rewards, color='#4cc9f0', alpha=0.35, linewidth=0.8, label='每局奖励')
        if n >= 10:
            ax.plot(episodes, calc_moving_avg(rewards, 10), color='#f72585', linewidth=2, label='10局均线')
        if n >= 50:
            ax.plot(episodes, calc_moving_avg(rewards, 50), color='#ffd60a', linewidth=2, label='50局均线')
        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('总奖励', fontsize=8)
        ax.set_title('📈 回合总奖励', fontsize=10)
        ax.legend(loc='upper left', fontsize=7, facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [0,1]：Boss剩余血量 ============
        ax = self.axes[0][1]
        ax.clear()
        self._style_ax(ax)

        boss_hp = df['boss_health']
        ax.bar(episodes, boss_hp, color='#e63946', alpha=0.6, width=1.0, label='Boss血量')
        if n >= 10:
            ax.plot(episodes, calc_moving_avg(boss_hp, 10), color='#ffd60a', linewidth=2, label='10局均线')
        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('Boss剩余血量 (%)', fontsize=8)
        ax.set_title('🩸 Boss剩余血量 (越低越好)', fontsize=10)
        ax.legend(loc='upper right', fontsize=7, facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [0,2]：胜率趋势 ============
        ax = self.axes[0][2]
        ax.clear()
        self._style_ax(ax)

        wr10 = safe_col(df, 'win_rate_10')
        wr50 = safe_col(df, 'win_rate_50')
        ax.plot(episodes, wr10 * 100, color='#06d6a0', linewidth=2, label='近10局胜率')
        ax.plot(episodes, wr50 * 100, color='#118ab2', linewidth=2, label='近50局胜率')
        ax.axhline(y=50, color='#ffffff', linestyle='--', alpha=0.3, label='50%基准线')
        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('胜率 (%)', fontsize=8)
        ax.set_ylim(-5, 105)
        ax.set_title('🏆 胜率趋势', fontsize=10)
        ax.legend(loc='upper left', fontsize=7, facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [0,3]：攻击命中率 ============
        ax = self.axes[0][3]
        ax.clear()
        self._style_ax(ax)

        hit_rate = safe_col(df, 'hit_rate') * 100
        attack_count = safe_col(df, 'attack_count')
        boss_hit_count = safe_col(df, 'boss_hit_count')
        miss_count = safe_col(df, 'miss_count')

        ax.plot(episodes, hit_rate, color='#06d6a0', alpha=0.4, linewidth=0.8, label='命中率')
        if n >= 10:
            ax.plot(episodes, calc_moving_avg(hit_rate, 10), color='#06d6a0', linewidth=2, label='10局均线')
        # 在右侧Y轴显示攻击次数
        ax2 = ax.twinx()
        ax2.bar(episodes, attack_count, color='#4cc9f0', alpha=0.2, width=1.0, label='攻击次数')
        ax2.bar(episodes, boss_hit_count, color='#06d6a0', alpha=0.3, width=1.0, label='命中次数')
        ax2.set_ylabel('次数', fontsize=8, color='#e0e0e0')
        ax2.tick_params(colors='#e0e0e0', labelsize=8)

        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('命中率 (%)', fontsize=8)
        ax.set_ylim(-5, 105)
        ax.set_title('� 攻击命中率 & 攻击次数', fontsize=10)
        # 合并两个轴的图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6,
                  facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [1,0]：DPS（每秒伤害输出） ============
        ax = self.axes[1][0]
        ax.clear()
        self._style_ax(ax)

        dps = safe_col(df, 'dps')
        ax.plot(episodes, dps, color='#ff6b6b', alpha=0.4, linewidth=0.8, label='每局DPS')
        if n >= 10:
            ax.plot(episodes, calc_moving_avg(dps, 10), color='#ff6b6b', linewidth=2, label='10局均线')
        if n >= 50:
            ax.plot(episodes, calc_moving_avg(dps, 50), color='#ffd60a', linewidth=2, label='50局均线')
        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('DPS (%/秒)', fontsize=8)
        ax.set_title('⚔️ DPS 每秒伤害输出 (越高越好)', fontsize=10)
        ax.legend(loc='upper left', fontsize=7, facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [1,1]：闪避成功率 ============
        ax = self.axes[1][1]
        ax.clear()
        self._style_ax(ax)

        dodge_rate = safe_col(df, 'dodge_success_rate') * 100
        dodge_total = safe_col(df, 'dodge_total')
        dodge_success = safe_col(df, 'dodge_success')

        ax.plot(episodes, dodge_rate, color='#a8dadc', alpha=0.4, linewidth=0.8, label='闪避成功率')
        if n >= 10:
            ax.plot(episodes, calc_moving_avg(dodge_rate, 10), color='#a8dadc', linewidth=2, label='10局均线')
        # 双Y轴：显示闪避次数
        ax2 = ax.twinx()
        ax2.bar(episodes, dodge_total, color='#457b9d', alpha=0.2, width=1.0, label='总闪避')
        ax2.bar(episodes, dodge_success, color='#a8dadc', alpha=0.3, width=1.0, label='成功闪避')
        ax2.set_ylabel('次数', fontsize=8, color='#e0e0e0')
        ax2.tick_params(colors='#e0e0e0', labelsize=8)

        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('闪避成功率 (%)', fontsize=8)
        ax.set_ylim(-5, 105)
        ax.set_title('🛡️ 闪避成功率 & 次数', fontsize=10)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6,
                  facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [1,2]：受伤次数 ============
        ax = self.axes[1][2]
        ax.clear()
        self._style_ax(ax)

        injuries = df['injured_count']
        ax.plot(episodes, injuries, color='#ef476f', alpha=0.4, linewidth=0.8, label='每局受伤')
        if n >= 10:
            ax.plot(episodes, calc_moving_avg(injuries, 10), color='#ffd166', linewidth=2, label='10局均线')
        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('受伤次数', fontsize=8)
        ax.set_title('💥 受伤次数 (越少越好)', fontsize=10)
        ax.legend(loc='upper right', fontsize=7, facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [1,3]：存活时间 ============
        ax = self.axes[1][3]
        ax.clear()
        self._style_ax(ax)

        duration = df['duration']
        ax.plot(episodes, duration, color='#8338ec', alpha=0.4, linewidth=0.8, label='每局时长')
        if n >= 10:
            ax.plot(episodes, calc_moving_avg(duration, 10), color='#3a86ff', linewidth=2, label='10局均线')
        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('持续时间 (秒)', fontsize=8)
        ax.set_title('⏱ 存活时间 (越长越好)', fontsize=10)
        ax.legend(loc='upper left', fontsize=7, facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [2,0]：伤害输出 vs 伤害承受 ============
        ax = self.axes[2][0]
        ax.clear()
        self._style_ax(ax)

        dmg_dealt = safe_col(df, 'damage_dealt')
        dmg_taken = safe_col(df, 'damage_taken')
        bar_width = 0.4
        ax.bar(episodes - bar_width / 2, dmg_dealt, width=bar_width, color='#06d6a0', alpha=0.7, label='输出伤害')
        ax.bar(episodes + bar_width / 2, dmg_taken, width=bar_width, color='#e63946', alpha=0.7, label='承受伤害')
        if n >= 10:
            ax.plot(episodes, calc_moving_avg(dmg_dealt, 10), color='#06d6a0', linewidth=2, linestyle='--', label='输出10均线')
            ax.plot(episodes, calc_moving_avg(dmg_taken, 10), color='#e63946', linewidth=2, linestyle='--', label='承受10均线')
        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('伤害 (血量%)', fontsize=8)
        ax.set_title('💊 伤害输出 vs 承受', fontsize=10)
        ax.legend(loc='upper left', fontsize=6, facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [2,1]：每步平均奖励 ============
        ax = self.axes[2][1]
        ax.clear()
        self._style_ax(ax)

        avg_rps = safe_col(df, 'avg_reward_per_step')
        ax.plot(episodes, avg_rps, color='#4cc9f0', alpha=0.4, linewidth=0.8, label='每步平均奖励')
        if n >= 10:
            ax.plot(episodes, calc_moving_avg(avg_rps, 10), color='#f72585', linewidth=2, label='10局均线')
        ax.axhline(y=0, color='#ffffff', linestyle='--', alpha=0.3, label='零线')
        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('平均奖励/步', fontsize=8)
        ax.set_title('📊 每步平均奖励', fontsize=10)
        ax.legend(loc='upper left', fontsize=7, facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [2,2]：动作多样性 & 技能使用 ============
        ax = self.axes[2][2]
        ax.clear()
        self._style_ax(ax)

        diversity = safe_col(df, 'action_diversity')
        skill_use = safe_col(df, 'skill_use_count')

        ax.bar(episodes, diversity, color='#b5838d', alpha=0.6, width=1.0, label='动作种类数')
        ax2 = ax.twinx()
        ax2.plot(episodes, skill_use, color='#ffd60a', linewidth=1.5, alpha=0.8, label='技能使用次数')
        if n >= 10:
            ax2.plot(episodes, calc_moving_avg(skill_use, 10), color='#ffd60a', linewidth=2, linestyle='--', label='技能10均线')
        ax2.set_ylabel('技能次数', fontsize=8, color='#e0e0e0')
        ax2.tick_params(colors='#e0e0e0', labelsize=8)

        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('动作种类', fontsize=8)
        ax.set_title('🎮 动作多样性 & 技能使用', fontsize=10)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6,
                  facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        # ============ 子图 [2,3]：AI综合能力评分 ============
        ax = self.axes[2][3]
        ax.clear()
        self._style_ax(ax)

        if n >= 5:
            boss_damage = 100 - boss_hp
            window = min(n, 50)
            r_norm = self._normalize_series(calc_moving_avg(rewards, window))
            d_norm = self._normalize_series(calc_moving_avg(boss_damage, window))
            t_norm = self._normalize_series(calc_moving_avg(duration, window))
            i_norm = self._normalize_series(calc_moving_avg(injuries, window))
            h_norm = self._normalize_series(calc_moving_avg(hit_rate, window))
            dps_norm = self._normalize_series(calc_moving_avg(dps, window))

            # 综合评分: 奖励20% + Boss伤害20% + 存活20% - 受伤15% + 命中率15% + DPS10%
            score = (r_norm * 0.20 + d_norm * 0.20 + t_norm * 0.20
                     - i_norm * 0.15 + h_norm * 0.15 + dps_norm * 0.10)
            score = score.clip(0, 1) * 100

            ax.fill_between(episodes, 0, score, color='#06d6a0', alpha=0.3)
            ax.plot(episodes, score, color='#06d6a0', linewidth=2, label='综合评分')
            ax.axhline(y=60, color='#ffd60a', linestyle='--', alpha=0.4, label='及格线(60)')
            ax.set_ylim(-5, 105)
        else:
            ax.text(0.5, 0.5, '数据不足\n至少需要5局', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#888888')

        ax.set_xlabel('回合', fontsize=8)
        ax.set_ylabel('评分', fontsize=8)
        ax.set_title('🎯 AI综合能力评分', fontsize=10)
        ax.legend(loc='upper left', fontsize=7, facecolor='#16213e', edgecolor='#3a3a5c', labelcolor='#e0e0e0')

        self.fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    def _style_ax(self, ax):
        """统一设置子图暗色风格"""
        ax.set_facecolor('#16213e')
        ax.grid(True, alpha=0.2, color='#ffffff')
        ax.tick_params(colors='#e0e0e0', labelsize=8)

    @staticmethod
    def _normalize_series(series):
        """Min-Max归一化"""
        s_min = series.min()
        s_max = series.max()
        if s_max == s_min:
            return pd.Series(0.5, index=series.index)
        return (series - s_min) / (s_max - s_min)

    def run(self):
        """启动实时仪表板"""
        print("=" * 60)
        print("🎮 悟空DQN训练实时仪表板 (3×4 全面监控)")
        print(f"📂 监控文件: {CSV_PATH}")
        print(f"🔄 刷新间隔: {REFRESH_INTERVAL / 1000:.0f}秒")
        print("-" * 60)
        print("📊 监控维度:")
        print("   行1: 总奖励 | Boss血量 | 胜率 | 攻击命中率")
        print("   行2: DPS   | 闪避率  | 受伤 | 存活时间")
        print("   行3: 伤害统计 | 每步奖励 | 动作多样性 | 综合评分")
        print("=" * 60)

        if not os.path.exists(CSV_PATH):
            print("⚠ 训练数据文件尚不存在，将在训练开始后自动显示...")

        self.anim = FuncAnimation(
            self.fig, self.update,
            interval=REFRESH_INTERVAL,
            cache_frame_data=False
        )
        plt.show()


if __name__ == '__main__':
    dashboard = LiveDashboard()
    dashboard.run()
