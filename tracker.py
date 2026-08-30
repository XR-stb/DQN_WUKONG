import os
import csv
import json
import time
import numpy as np
from datetime import datetime
from log import log


def _safe_json_default(obj):
    """JSON序列化钩子：自动将numpy类型转换为Python原生类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class RewardTracker:
    """训练数据追踪器
    
    统一记录每回合的所有关键指标到单个CSV文件，支持实时追加写入。
    训练过程中可以通过 live_dashboard.py 实时查看训练进展。
    """

    # CSV 列名定义
    COLUMNS = [
        'episode',           # 回合编号
        'total_reward',      # 回合总奖励
        'steps',             # 回合步数
        'boss_health',       # 回合结束时Boss血量
        'self_health',       # 回合结束时自身血量
        'injured_count',     # 受伤次数
        'duration',          # 回合持续时间(秒)
        'result',            # 胜负结果: win/lose
        # --- 新增：战斗详细统计 ---
        'attack_count',      # AI总攻击次数（轻攻击/重攻击/组合攻击等）
        'boss_hit_count',    # Boss被命中次数（根据Boss血量下降判定）
        'miss_count',        # 攻击打空次数 = attack_count - boss_hit_count
        'hit_rate',          # 命中率 = boss_hit_count / attack_count
        'dodge_total',       # 总闪避次数
        'dodge_success',     # 成功闪避次数（闪避后未受伤）
        'dodge_success_rate',# 闪避成功率
        'dps',               # 每秒伤害输出 (damage_dealt / duration)
        'skill_use_count',   # 技能总使用次数
        'damage_dealt',      # 对Boss造成的总伤害（血量百分比）
        'damage_taken',      # 自身受到的总伤害（血量百分比）
        'avg_reward_per_step', # 平均每步奖励
        'action_diversity',  # 动作多样性（使用了多少种不同动作）
        # --- 滑动统计 ---
        'avg_reward_10',     # 最近10局平均奖励
        'avg_reward_50',     # 最近50局平均奖励
        'win_rate_10',       # 最近10局胜率
        'win_rate_50',       # 最近50局胜率
        'best_reward',       # 历史最高单局奖励
        'timestamp',         # 时间戳
    ]

    def __init__(self, train_data_dir, milestone_interval=20):
        self.train_data_dir = train_data_dir
        self.milestone_interval = milestone_interval
        self.episode_num = 0

        # 当前回合的累积奖励列表（每步一个值）
        self.current_rewards = []

        # 历史数据（用于计算滑动平均等）
        self.reward_history = []
        self.result_history = []  # 'win' / 'lose'
        self.best_reward = float('-inf')

        # 创建目录
        os.makedirs(train_data_dir, exist_ok=True)

        # CSV 文件路径
        self.csv_path = os.path.join(train_data_dir, 'training_log.csv')

        # 如果已有记录文件，加载历史数据以继续训练
        if os.path.exists(self.csv_path):
            self._load_history()
        else:
            self._init_csv()

        # 实时状态 JSON 文件路径（供 OSD 叠加层读取）
        self.realtime_path = os.path.join(train_data_dir, 'realtime_state.json')
        self._realtime_state = {}  # 缓存最新实时状态

        log.info(f"训练数据追踪器已初始化 | 历史回合数: {self.episode_num} | CSV: {self.csv_path}")

    def _init_csv(self):
        """初始化CSV文件，写入表头"""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)

    def _load_history(self):
        """从已有CSV加载历史数据，支持断点续训。
        如果CSV列与当前COLUMNS不匹配（版本升级），自动迁移数据。
        """
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_columns = reader.fieldnames or []
                rows = list(reader)

            # 检查CSV列是否与当前COLUMNS匹配
            if set(existing_columns) != set(self.COLUMNS):
                log.info(f"检测到CSV列变更（旧列{len(existing_columns)}个 → 新列{len(self.COLUMNS)}个），自动迁移数据...")
                # 先加载历史数据到内存
                for row in rows:
                    self.episode_num = int(row['episode'])
                    reward = float(row['total_reward'])
                    self.reward_history.append(reward)
                    self.result_history.append(row['result'])
                    if reward > self.best_reward:
                        self.best_reward = reward

                # 用新的COLUMNS重写CSV（旧数据中没有的列填默认值0）
                self._init_csv()
                with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                    for row in rows:
                        migrated_row = {col: row.get(col, 0) for col in self.COLUMNS}
                        writer.writerow(migrated_row)
                    f.flush()
                log.info(f"CSV数据迁移完成，共迁移 {len(rows)} 条记录")
            else:
                # 列匹配，正常加载
                for row in rows:
                    self.episode_num = int(row['episode'])
                    reward = float(row['total_reward'])
                    self.reward_history.append(reward)
                    self.result_history.append(row['result'])
                    if reward > self.best_reward:
                        self.best_reward = reward

            log.info(f"已加载 {self.episode_num} 条历史训练记录")
        except Exception as e:
            log.error(f"加载历史数据失败: {e}，将重新初始化")
            self.episode_num = 0
            self.reward_history.clear()
            self.result_history.clear()
            self.best_reward = float('-inf')
            self._init_csv()

    def add_reward(self, reward):
        """每步调用，记录当前步的奖励"""
        self.current_rewards.append(reward)

    def update_realtime_state(self, episode, step, action_name, reward,
                              self_blood, boss_blood, injured_count,
                              episode_reward, duration, epsilon=0.0,
                              ui_status=None, episode_stats=None):
        """每步调用，将实时训练状态写入 JSON 文件，供 OSD 叠加层读取
        
        Args:
            episode: 当前回合号
            step: 当前步数
            action_name: 当前执行的动作名称
            reward: 当前步奖励
            self_blood: 玩家血量
            boss_blood: Boss血量
            injured_count: 累计受伤次数
            episode_reward: 回合累计奖励
            duration: 回合已经过时间(秒)
            epsilon: 当前探索率
            ui_status: 额外的UI状态字典（法力、精力、技能等），可选
            episode_stats: 本回合的详细统计字典，包含以下字段：
                - action_counts: dict  各动作使用次数
                - attack_count: int    AI总攻击次数（轻攻击/重攻击/组合攻击等）
                - boss_hit_count: int  Boss被命中次数（根据血量下降判定）
                - skill_usage: dict    技能使用次数统计
                - dodge_count: int     成功闪避次数
                - dodge_success_rate: float  闪避成功率
                - reward_history: list  本回合每步奖励列表（最近N步）
                - self_blood_history: list  本回合玩家血量变化列表（最近N步）
                - boss_blood_history: list  本回合Boss血量变化列表（最近N步）
                - blood_damage_dealt: float  本回合对Boss造成的总伤害
                - blood_damage_taken: float  本回合自身受到的总伤害
                - avg_reward_per_step: float 本回合平均每步奖励
                - dps: float           每秒伤害输出（DPS）
        """
        self._realtime_state = {
            'episode': episode,
            'step': step,
            'action': action_name,
            'reward': round(reward, 2),
            'episode_reward': round(episode_reward, 2),
            'self_blood': round(self_blood, 1),
            'boss_blood': round(boss_blood, 1),
            'injured_count': injured_count,
            'duration': round(duration, 1),
            'epsilon': round(epsilon, 4),
            'best_reward': round(self.best_reward, 2) if self.best_reward != float('-inf') else 0,
            'total_episodes': self.episode_num,
            'win_rate_10': round(self._calc_win_rate(self.result_history, 10), 4),
            'avg_reward_10': round(self._calc_avg(self.reward_history, 10), 2),
            'timestamp': time.time(),
        }
        # 合并额外的 UI 状态数据（法力、精力、技能等）
        if ui_status:
            self._realtime_state['ui'] = ui_status
        # 合并本回合详细统计数据（动作分布、血量曲线等）
        if episode_stats:
            self._realtime_state['episode_stats'] = episode_stats
        self._write_realtime_state()

    def _write_realtime_state(self):
        """原子写入实时状态 JSON（先写临时文件再rename，防止读到半截数据）"""
        tmp_path = self.realtime_path + '.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(self._realtime_state, f, ensure_ascii=False, default=_safe_json_default)
            # 原子替换
            if os.path.exists(self.realtime_path):
                os.replace(tmp_path, self.realtime_path)
            else:
                os.rename(tmp_path, self.realtime_path)
        except Exception as e:
            log.debug(f"写入实时状态JSON失败: {e}")

    def end_episode(self, boss_health, self_health=0, injured_count=0,
                    duration=0.0, steps=0, result='lose', **episode_stats):
        """回合结束时调用，汇总并写入本回合数据
        
        Args:
            boss_health: Boss剩余血量
            self_health: 自身剩余血量
            injured_count: 受伤次数
            duration: 回合持续时间(秒)
            steps: 回合总步数
            result: 'win' 或 'lose'
            **episode_stats: 本回合的详细战斗统计，包含:
                - attack_count: int       AI总攻击次数
                - boss_hit_count: int     Boss被命中次数
                - dodge_total: int        总闪避次数
                - dodge_success: int      成功闪避次数
                - skill_use_count: int    技能总使用次数
                - damage_dealt: float     对Boss造成的总伤害
                - damage_taken: float     自身受到的总伤害
                - action_diversity: int   使用了多少种不同动作
        """
        self.episode_num += 1
        total_reward = sum(self.current_rewards)

        # 更新历史记录
        self.reward_history.append(total_reward)
        self.result_history.append(result)
        if total_reward > self.best_reward:
            self.best_reward = total_reward

        # 计算滑动平均
        avg_10 = self._calc_avg(self.reward_history, 10)
        avg_50 = self._calc_avg(self.reward_history, 50)
        wr_10 = self._calc_win_rate(self.result_history, 10)
        wr_50 = self._calc_win_rate(self.result_history, 50)

        # 从 episode_stats 提取战斗统计
        attack_count = episode_stats.get('attack_count', 0)
        boss_hit_count = episode_stats.get('boss_hit_count', 0)
        miss_count = max(attack_count - boss_hit_count, 0)
        hit_rate = round(boss_hit_count / attack_count, 4) if attack_count > 0 else 0.0
        dodge_total = episode_stats.get('dodge_total', 0)
        dodge_success = episode_stats.get('dodge_success', 0)
        dodge_success_rate = round(dodge_success / dodge_total, 4) if dodge_total > 0 else 0.0
        damage_dealt = episode_stats.get('damage_dealt', 0.0)
        damage_taken = episode_stats.get('damage_taken', 0.0)
        dps = round(damage_dealt / max(duration, 0.1), 2)
        skill_use_count = episode_stats.get('skill_use_count', 0)
        avg_reward_per_step = round(total_reward / max(steps, 1), 2)
        action_diversity = episode_stats.get('action_diversity', 0)

        # 构造一行数据
        row = {
            'episode': self.episode_num,
            'total_reward': round(total_reward, 2),
            'steps': steps,
            'boss_health': round(boss_health, 2),
            'self_health': round(self_health, 2),
            'injured_count': injured_count,
            'duration': round(duration, 1),
            'result': result,
            # --- 新增战斗统计 ---
            'attack_count': attack_count,
            'boss_hit_count': boss_hit_count,
            'miss_count': miss_count,
            'hit_rate': round(hit_rate, 4),
            'dodge_total': dodge_total,
            'dodge_success': dodge_success,
            'dodge_success_rate': round(dodge_success_rate, 4),
            'dps': dps,
            'skill_use_count': skill_use_count,
            'damage_dealt': round(damage_dealt, 2),
            'damage_taken': round(damage_taken, 2),
            'avg_reward_per_step': avg_reward_per_step,
            'action_diversity': action_diversity,
            # --- 滑动统计 ---
            'avg_reward_10': round(avg_10, 2),
            'avg_reward_50': round(avg_50, 2),
            'win_rate_10': round(wr_10, 4),
            'win_rate_50': round(wr_50, 4),
            'best_reward': round(self.best_reward, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # 实时追加写入CSV（每局一行，flush确保实时可读）
        self._append_row(row)

        # 控制台实时打印训练进度摘要
        result_emoji = "🏆" if result == 'win' else "💀"
        log.info(
            f"{result_emoji} 回合 {self.episode_num} 汇总 | "
            f"奖励: {total_reward:+.1f} | "
            f"步数: {steps} | "
            f"Boss血量: {boss_health:.1f}% | "
            f"受伤: {injured_count}次 | "
            f"耗时: {duration:.1f}s | "
            f"攻击: {attack_count} 命中: {boss_hit_count} 命中率: {hit_rate:.0%} | "
            f"闪避: {dodge_success}/{dodge_total} | "
            f"DPS: {dps:.1f} | "
            f"近10局均奖: {avg_10:+.1f} | "
            f"近10局胜率: {wr_10:.0%}"
        )

        # 每N局打印一次详细汇报
        if self.episode_num % self.milestone_interval == 0:
            self._print_milestone_report()

        # 清空当前回合数据
        self.current_rewards.clear()

    def _append_row(self, row):
        """追加一行数据到CSV文件"""
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writerow(row)
                f.flush()  # 立即刷新到磁盘，确保每局数据实时可读
        except Exception as e:
            log.error(f"写入训练数据失败: {e}")

    def _calc_avg(self, data, window):
        """计算最近N局的平均值"""
        if not data:
            return 0.0
        recent = data[-window:]
        return sum(recent) / len(recent)

    def _calc_win_rate(self, results, window):
        """计算最近N局的胜率"""
        if not results:
            return 0.0
        recent = results[-window:]
        wins = sum(1 for r in recent if r == 'win')
        return wins / len(recent)

    def _print_milestone_report(self):
        """每隔N局打印里程碑报告"""
        total_wins = sum(1 for r in self.result_history if r == 'win')
        total_episodes = len(self.result_history)
        overall_wr = total_wins / total_episodes if total_episodes > 0 else 0

        log.info("=" * 60)
        log.info(f"📊 训练里程碑报告 | 第 {self.episode_num} 回合")
        log.info(f"   总胜率: {overall_wr:.1%} ({total_wins}/{total_episodes})")
        log.info(f"   历史最高奖励: {self.best_reward:+.1f}")
        log.info(f"   近10局平均奖励: {self._calc_avg(self.reward_history, 10):+.1f}")
        log.info(f"   近50局平均奖励: {self._calc_avg(self.reward_history, 50):+.1f}")
        log.info(f"   近10局胜率: {self._calc_win_rate(self.result_history, 10):.0%}")
        log.info(f"   近50局胜率: {self._calc_win_rate(self.result_history, 50):.0%}")
        log.info("=" * 60)
