"""
╔══════════════════════════════════════════════════════════════════╗
║                    🎯 奖励函数 v3 — 收敛优先设计                  ║
║                                                                  ║
║  v3 设计原则（针对 v2 收敛失败的全面修正）:                       ║
║                                                                  ║
║  1. 简化信号 — 奖励维度从9个降到4个核心维度，减少噪声             ║
║  2. 正向激励为主 — 攻击/伤害给大正奖励，惩罚保持克制             ║
║  3. 移除CD硬惩罚 — 不再因CD惩罚agent，交给环境自然反馈          ║
║  4. 线性缩放 — 所有奖励控制在 [-50, +100] 区间，避免极端值      ║
║  5. 攻击即奖励 — 任何攻击动作都给正奖励，降低进攻门槛           ║
║                                                                  ║
║  v2 失败原因分析:                                                ║
║    - 84回合训练数据显示: 攻击次数从ep10起降为0                   ║
║    - 动作多样性降到2（只用IDLE+TISHEN）                          ║
║    - 根因: IDLE每步约-140~-170惩罚，但攻击风险更大(-200+)       ║
║    - CD系统(-60)叠加硬拦截 → agent无法逃脱IDLE陷阱               ║
║    - 受伤递增 INJURY_SCALING=-15*cnt → 后期每次受伤-100+          ║
║                                                                  ║
║  对外接口:                                                       ║
║    使用 RewardJudge 类即可，它整合了奖励计算 + 数据记录。         ║
║    >> from reward_calculator import RewardJudge                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import time
import yaml
from log import log
from tracker import RewardTracker


# ============================================================
#  动作分类常量
# ============================================================

# 攻击型动作
ATTACK_ACTIONS = frozenset({
    'LIGHT_ATTACK', 'HEAVY_ATTACK', 'ATTACK_DODGE',
    'FIVE_HIT_COMBO', 'QIESHOU', 'STEALTH_CHARGE',
})

# 闪避型动作
DODGE_ACTIONS = frozenset({
    'DODGE', 'DODGE_TWO', 'DODGE_THREE',
})

# 技能型动作
SKILL_ACTIONS = frozenset({
    'SKILL_1', 'SKILL_3', 'SKILL_4', 'STEALTH_CHARGE',
})

# 移动型动作
MOVE_ACTIONS = frozenset({
    'GO_FORWARD', 'GO_BACK',
})

# 安全攻击（后摇短）
SAFE_ATTACK_ACTIONS = frozenset({
    'LIGHT_ATTACK', 'ATTACK_DODGE',
})

# 技能名 → 状态字段的映射
SKILL_STATUS_MAP = {
    'SKILL_1':        'skill_1',
    'SKILL_2':        'skill_2',
    'SKILL_3':        'skill_3',
    'SKILL_4':        'skill_4',
    'STEALTH_CHARGE': 'skill_2',
    'TISHEN':         'skill_ts',
    'FABAO':          'skill_fb',
}


# ============================================================
#  奖励计算器 v3 — 收敛优先
# ============================================================

class RewardCalculator:
    """
    奖励函数计算器 v3 — 收敛优先设计。

    核心改进（vs v2）:
      1. 大幅简化: 从9个奖励模块降到4个核心模块
      2. 正向驱动: 攻击=正奖励，受伤=适度负奖励，比例约3:1
      3. 无CD惩罚: 移除所有CD检查和硬拦截，agent自由探索
      4. 窄幅奖励: 单步奖励控制在 [-50, +100]，训练稳定
      5. 简单势函数: 只看血量差变化，不叠加复杂因子

    使用方式:
        calculator = RewardCalculator()
        calculator.reset()
        reward = calculator.calc(...)
    """

    # ==================== 核心参数 ====================
    # --- 攻击奖励（正向，鼓励进攻） ---
    ATTACK_BASE_REWARD = 8.0        # 任何攻击动作的基础正奖励
    SAFE_ATTACK_BONUS = 5.0         # 安全攻击(平A/攻击闪避)额外奖励
    DAMAGE_PER_PERCENT = 30.0       # Boss每掉1%血的奖励
    SAFE_OUTPUT_BONUS = 15.0        # 造成伤害且未受伤

    # --- 闪避奖励 ---
    DODGE_SUCCESS_REWARD = 10.0     # 闪避成功（威胁窗口内）
    DODGE_BASE_REWARD = 2.0         # 闪避基础奖励（任何时候闪避都不应被惩罚太重）

    # --- 受伤惩罚（适度，不递增） ---
    INJURY_PENALTY = -20.0          # 固定受伤惩罚（不再递增！）
    ATTACK_INJURED_EXTRA = -10.0    # 攻击时受伤的额外惩罚

    # --- 行为引导（轻量级） ---
    IDLE_PENALTY = -3.0             # 每次IDLE的小惩罚（推动探索）
    LONG_IDLE_PENALTY = -8.0        # 连续IDLE (>3次)的惩罚
    DIVERSITY_BONUS = 1.0           # 使用新动作的小奖励

    # --- 势函数 ---
    POTENTIAL_SCALE = 2.0           # 势函数缩放（比v2更小，避免噪声）
    POTENTIAL_GAMMA = 0.99

    # --- 回合结束 ---
    WIN_REWARD = 500.0              # 降低胜利奖励（v2的3000太大，造成梯度不稳）
    LOSE_PENALTY = -100.0           # 降低失败惩罚（v2的-1500太大）
    BOSS_DAMAGE_BONUS = 5.0         # 失败时每打掉Boss 1%血的补偿

    # --- 资源管理 ---
    DRINK_LOW_HP_REWARD = 10.0      # 低血量喝药奖励
    DRINK_HIGH_HP_PENALTY = -10.0   # 高血量喝药惩罚

    # --- 威胁窗口 ---
    THREAT_WINDOW_SIZE = 3          # 比v2小（4→3），响应更快

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        """内部状态初始化"""
        self.prev_action = ''
        self.prev_injured = False
        self.consecutive_idle_count = 0
        self.steps_since_last_attack = 0
        self.total_attack_count = 0
        self.used_actions = set()       # 本局使用过的动作集合

        # 威胁窗口
        self.recent_injury_history = []
        self.boss_threat_active = False

        # 势函数
        self.prev_potential = 0.0

        # 替身一局一次
        self.tishen_used = False

    def reset(self):
        """每局开始时调用"""
        self._reset_state()

    # ================================================================
    #  威胁窗口（简化版）
    # ================================================================

    def _update_threat_window(self, injured):
        """更新威胁窗口"""
        self.recent_injury_history.append(injured)
        if len(self.recent_injury_history) > self.THREAT_WINDOW_SIZE * 2:
            self.recent_injury_history = self.recent_injury_history[-(self.THREAT_WINDOW_SIZE * 2):]
        window = self.recent_injury_history[-self.THREAT_WINDOW_SIZE:]
        self.boss_threat_active = any(window)

    # ================================================================
    #  势函数（简化版）
    # ================================================================

    def _potential_reward(self, cur_status, next_status):
        """F = γ * Φ(s') - Φ(s), 其中 Φ = (boss_lost - self_lost) * scale"""
        boss_blood = float(next_status['boss_blood'])
        self_blood = float(next_status['self_blood'])
        new_potential = ((100.0 - boss_blood) - (100.0 - self_blood)) * self.POTENTIAL_SCALE
        reward = self.POTENTIAL_GAMMA * new_potential - self.prev_potential
        self.prev_potential = new_potential
        return reward

    # ================================================================
    #  主入口
    # ================================================================

    def calc(self, action_name, injured, cur_status, next_status,
             events, survival_time, done, injured_cnt, steps=0):
        """
        计算单步奖励。v3 简化版 — 只有4个核心模块。
        """
        reward = 0.0

        if done:
            reward += self._on_episode_end(cur_status, next_status, survival_time, injured_cnt)
        else:
            # ① 势函数（持续的血量优势信号）
            reward += self._potential_reward(cur_status, next_status)

            # ② 攻击与伤害奖励（核心正向激励）
            reward += self._attack_reward(action_name, cur_status, next_status, injured)

            # ③ 受伤惩罚（固定值，不递增）
            reward += self._injury_penalty(injured, action_name)

            # ④ 行为引导（轻量级，防止IDLE循环）
            reward += self._behavior_guide(action_name, cur_status, next_status)

        # 更新内部状态
        self._update_state(action_name, injured)

        return reward

    # ================================================================
    #  ① 攻击与伤害奖励
    # ================================================================

    def _attack_reward(self, action_name, cur_status, next_status, injured):
        """
        v3核心设计: 攻击就给正奖励，造成伤害给大奖励。
        
        关键改变（vs v2）:
        - 攻击miss不再惩罚（v2的-3.0移除）
        - 任何攻击都有正基础奖励
        - 安全攻击(平A)有额外奖励
        - 造成Boss血量下降 → 大奖励
        """
        reward = 0.0
        boss_blood_drop = float(cur_status['boss_blood']) - float(next_status['boss_blood'])

        if action_name in ATTACK_ACTIONS:
            # 基础攻击奖励（永远为正）
            reward += self.ATTACK_BASE_REWARD

            # 安全攻击额外奖励
            if action_name in SAFE_ATTACK_ACTIONS:
                reward += self.SAFE_ATTACK_BONUS

            self.steps_since_last_attack = 0
            self.total_attack_count += 1

        # Boss血量下降 → 造成了实际伤害
        if boss_blood_drop > 0.5:
            reward += boss_blood_drop * self.DAMAGE_PER_PERCENT
            # 安全输出加成：造成伤害且没受伤
            if not injured:
                reward += self.SAFE_OUTPUT_BONUS

        # 闪避动作
        if action_name in DODGE_ACTIONS:
            if self.boss_threat_active:
                # 威胁窗口内闪避 → 好的防御行为
                reward += self.DODGE_SUCCESS_REWARD
            else:
                # 非威胁时闪避 → 小正奖励（不惩罚，让agent自由探索）
                reward += self.DODGE_BASE_REWARD

        return reward

    # ================================================================
    #  ② 受伤惩罚（固定，不递增）
    # ================================================================

    def _injury_penalty(self, injured, action_name):
        """
        v3关键改变: 受伤惩罚固定为 -20，不再随受伤次数递增。
        
        v2的问题: INJURY_SCALING = -15 * injured_cnt → 第5次受伤惩罚=-90
        这导致agent学到"存活越久惩罚越大"，所以干脆不动等死。
        """
        if not injured:
            return 0.0

        reward = self.INJURY_PENALTY  # 固定 -20

        # 攻击时受伤只加一小点额外惩罚（不要太大，否则agent不敢攻击）
        if action_name in ATTACK_ACTIONS:
            reward += self.ATTACK_INJURED_EXTRA

        return reward

    # ================================================================
    #  ③ 行为引导（轻量级）
    # ================================================================

    def _behavior_guide(self, action_name, cur_status, next_status):
        """
        v3 设计: 只用最小力度引导行为方向。
        
        - IDLE给小惩罚（-3），连续IDLE给稍大惩罚（-8）
        - 使用新动作给小奖励（+1）
        - 长时间不攻击给小惩罚（-5）
        - 喝药管理
        
        所有惩罚都很小，不会压过攻击的正奖励。
        """
        reward = 0.0

        # IDLE惩罚
        if action_name == 'IDLE':
            self.consecutive_idle_count += 1
            if self.consecutive_idle_count > 3:
                reward += self.LONG_IDLE_PENALTY
            else:
                reward += self.IDLE_PENALTY
        else:
            self.consecutive_idle_count = 0

        # 动作多样性奖励
        if action_name not in self.used_actions:
            self.used_actions.add(action_name)
            reward += self.DIVERSITY_BONUS

        # 长时间不攻击的惰性惩罚（比v2温和得多）
        if action_name not in ATTACK_ACTIONS:
            self.steps_since_last_attack += 1

        if self.steps_since_last_attack > 6:
            # 超过6步不攻击给 -5（v2是-50，太大了）
            reward += -5.0

        # 喝药管理
        if action_name == 'DRINK_POTION':
            self_blood = float(cur_status['self_blood'])
            hulu = float(cur_status['hulu']) if cur_status['hulu'] else 100.0
            if hulu <= 0:
                reward += -15.0  # 没药还喝
            elif self_blood < 35:
                reward += self.DRINK_LOW_HP_REWARD
            elif self_blood > 70:
                reward += self.DRINK_HIGH_HP_PENALTY

        # 替身管理
        if action_name == 'TISHEN':
            if self.tishen_used:
                reward += -20.0  # 已用过
            elif not cur_status['skill_ts']:
                reward += -20.0  # 不可用
            else:
                self.tishen_used = True
                reward += 5.0    # 正常使用

        # 技能CD管理（只给提示性小惩罚，不硬拦截）
        if action_name in SKILL_STATUS_MAP and action_name != 'TISHEN':
            skill_key = SKILL_STATUS_MAP[action_name]
            if not cur_status[skill_key]:
                reward += -10.0  # 技能CD中使用，小惩罚

        return reward

    # ================================================================
    #  回合结束
    # ================================================================

    def _on_episode_end(self, cur_status, next_status, survival_time, injured_cnt):
        """
        v3: 大幅降低回合结束奖惩的量级。
        
        v2 的 WIN=3000, LOSE=-1500 导致梯度极不稳定。
        v3 控制在 WIN=500, LOSE=-100 范围内。
        """
        reward = 0.0
        boss_blood = float(next_status['boss_blood'])
        self_blood = float(next_status['self_blood'])

        is_win = boss_blood < 10 and self_blood > 10

        if is_win:
            reward += self.WIN_REWARD
            # 低受伤奖励
            if injured_cnt <= 3:
                reward += 100.0
            # 残余血量奖励
            reward += self_blood * 2.0
        else:
            reward += self.LOSE_PENALTY
            # Boss掉血越多，惩罚越轻
            boss_damage_dealt = 100.0 - boss_blood
            reward += boss_damage_dealt * self.BOSS_DAMAGE_BONUS

            # 攻击次数奖励（v3: 正向补偿代替v2的-300惩罚）
            # 打了很多次，虽然输了但努力了，给一点补偿
            if self.total_attack_count >= 10:
                reward += 20.0
            elif self.total_attack_count >= 5:
                reward += 10.0
            # 没攻击不额外惩罚，已经有LOSE_PENALTY了

        return reward

    # ================================================================
    #  状态更新
    # ================================================================

    def _update_state(self, action_name, injured):
        """每步结束后更新内部追踪状态"""
        self.prev_action = action_name
        self.prev_injured = injured
        self._update_threat_window(injured)

    # ================================================================
    #  动作CD查询接口（保留接口兼容性，但不再惩罚）
    # ================================================================

    def is_action_on_cd(self, action_name, survival_time):
        """
        v3: 保留接口但总是返回 (False, 0)。
        不再做CD硬拦截，让agent自由探索。
        如果agent选了CD中的技能，游戏本身会忽略输入，
        这本身就是自然的负反馈（浪费了一步），不需要额外惩罚。
        """
        return False, 0.0

    def get_cd_status_all(self, survival_time):
        """兼容接口"""
        return {}


# ============================================================
#  对外接口 — RewardJudge
# ============================================================

def _load_dashboard_config():
    """加载监控配置"""
    with open('./config/dashboard_conf.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class RewardJudge:
    """
    奖励评判器 — 训练循环的唯一对接入口。
    整合 RewardCalculator + RewardTracker。
    """

    def __init__(self):
        dc = _load_dashboard_config()
        self.reward_tracker = RewardTracker(
            train_data_dir=dc['data']['train_data_dir'],
            milestone_interval=dc['report']['milestone_interval'],
        )
        self._calculator = RewardCalculator()

    def reset(self):
        """每局开始时重置"""
        self._calculator.reset()

    def is_action_on_cd(self, action_name, survival_time):
        """v3: 不再做CD拦截"""
        return self._calculator.is_action_on_cd(action_name, survival_time)

    def get_cd_status_all(self, survival_time):
        """兼容接口"""
        return self._calculator.get_cd_status_all(survival_time)

    def judge(self, action_name, injured, cur_status, next_status,
              events, survival_time, done, injured_cnt, steps=0,
              episode_stats=None):
        """每步调用，计算奖励并记录数据。"""
        reward = self._calculator.calc(
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
                **(episode_stats or {}),
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

        self.reward_tracker.add_reward(reward)
        return reward
