import time
import yaml
from log import log
from tracker import RewardTracker
from reward_calculator import RewardCalculator


def _load_dashboard_config():
    """加载监控配置（获取 train_data_dir）"""
    with open('./config/dashboard_conf.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class ActionJudge:
    """
    动作评判器 — 连接训练循环与奖励函数的桥梁。

    职责:
      1. 接收 process_handler 传入的每步状态
      2. 委托 RewardCalculator 计算奖励（所有奖励逻辑在 reward_calculator.py）
      3. 管理 RewardTracker 进行数据记录

    ⚠️ 如需调整奖励逻辑，请修改 reward_calculator.py，而非本文件。
    """

    def __init__(self):
        dc = _load_dashboard_config()

        # 数据追踪器（CSV记录 + 实时状态JSON）
        self.reward_tracker = RewardTracker(
            train_data_dir=dc['data']['train_data_dir'],
            milestone_interval=dc['report']['milestone_interval'],
        )

        # 奖励计算器（所有奖励逻辑在这里）
        self.calculator = RewardCalculator()

    def reset(self):
        """每局开始时重置状态"""
        self.calculator.reset()

    def judge(
        self,
        action_name,
        injured,
        cur_status,
        next_status,
        events,
        survival_time,
        done,
        injured_cnt,
        steps=0,
    ):
        """
        每步调用，计算奖励并记录数据。

        Args:
            action_name:   本步执行的动作名称
            injured:       本步是否受伤
            cur_status:    动作前状态字典
            next_status:   动作后状态字典
            events:        本步事件列表
            survival_time: 存活时间(秒)
            done:          回合是否结束
            injured_cnt:   累计受伤次数
            steps:         当前步数

        Returns:
            float: 本步总奖励
        """
        # 委托 RewardCalculator 计算奖励
        reward = self.calculator.calc(
            action_name=action_name,
            injured=injured,
            cur_status=cur_status,
            next_status=next_status,
            events=events,
            survival_time=survival_time,
            done=done,
            injured_cnt=injured_cnt,
            steps=steps,
        )

        # 回合结束时记录到CSV
        if done:
            real_boss_blood = float(next_status['boss_blood'])
            real_self_blood = float(next_status['self_blood'])
            is_win = real_boss_blood < 10 and real_self_blood > 10

            self.reward_tracker.end_episode(
                boss_health=real_boss_blood,
                self_health=real_self_blood,
                injured_count=injured_cnt,
                duration=survival_time,
                steps=steps,
                result='win' if is_win else 'lose',
            )

            log.debug(
                f"GAME OVER | "
                f"BOSS HP: {real_boss_blood:.2f} | "
                f"SELF HP: {real_self_blood:.2f} | "
                f"SURVIVAL TIME: {survival_time:.2f} | "
                f"RESULT: {'WIN' if is_win else 'LOSE'}"
            )
        else:
            log.debug(
                f"TIME: {survival_time:.2f} | "
                f"BOSS HP: {next_status['boss_blood']:.2f} | "
                f"SELF HP: {next_status['self_blood']:.2f} | "
                f"ACTION: {action_name} | "
                f"REWARD: {reward:+.2f}"
            )

        # 记录每步奖励
        self.reward_tracker.add_reward(reward)
        return reward
