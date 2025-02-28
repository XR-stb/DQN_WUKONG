import time
from log import log
from tracker import RewardTracker


class ActionJudge:
    def __init__(self):
        self.reward_tracker = RewardTracker(train_data_dir="train_data/data")
        self.prev_status = {}
        self.last_attack_time = 0
        self.combo_counter = 0  # 自主维护连击计数器
        self.last_dodge_time = 0  # 自主维护闪避时间
        self.attack_count = 0  # 新增攻击次数计数器
        self.last_boss_attack_time = 0  # 新增BOSS攻击时间记录
        self.consecutive_forward = 0  # 连续前进计数器
        self.last_meaningful_action = time.time()  # 有效动作时间戳
        self.attack_pattern = {
            "last_attack_type": None,
            "combo_streak": 0,  # 连击计数
            "last_boss_attack_time": 0,  # BOSS上次攻击时间
        }
        self.dodge_cooldown = 0  # 闪避冷却
        self.post_dodge_phase = 0  # 闪避后阶段 (0:未闪避 1:0-0.8s 2:0.8-2.8s)
        self.safe_steps = 0  # 安全步数计数器
        self.tishan_used = False
        self.tishan_effective = False
        # 技能冷却跟踪器
        self.skill_cooldowns = {
            "SKILL_1": {"last_used": 0, "cooldown": 30},
            "SKILL_3": {"last_used": 0, "cooldown": 30},
        }
        self.skill_effectiveness = {}  # 记录技能是否有效

    def reset(self):
        """重置每局状态"""
        self.prev_status = {}
        self.prev_status["boss_blood"] = 100
        self.prev_status["self_blood"] = 100
        self.last_attack_time = 0
        self.combo_counter = 0
        self.last_dodge_time = 0
        self.attack_count = 0
        self.last_boss_attack_time = 0
        self.consecutive_forward = 0
        self.last_meaningful_action = time.time()
        self.attack_pattern = {
            "last_attack_type": None,
            "combo_streak": 0,  # 连击计数
            "last_boss_attack_time": 0,  # BOSS上次攻击时间
        }
        self.dodge_cooldown = 0  # 闪避冷却
        self.post_dodge_phase = 0  # 闪避后阶段 (0:未闪避 1:0-0.8s 2:0.8-2.8s)
        self.safe_steps = 0  # 安全步数计数器
        self.tishan_used = False
        self.tishan_effective = False
        # 每局重置冷却状态
        for skill in self.skill_cooldowns:
            self.skill_cooldowns[skill]["last_used"] = 0
        self.skill_effectiveness.clear()

    def _is_skill_action(self, action_name):
        """判断是否为需要冷却的技能"""
        return action_name.startswith("SKILL_") or action_name in self.skill_cooldowns

    def judge(
        self,
        action_name,
        injured,
        b_status,
        a_status,
        events,
        survival_time,
        done,
        injured_cnt,
    ):
        reward = 0.0
        current_time = time.time()

        # 实时检测BOSS攻击状态（新增方法）
        boss_attacking = self._detect_boss_attack(b_status, a_status)

        # 统一技能处理
        if self._is_skill_action(action_name):
            skill_data = self.skill_cooldowns.get(action_name, {"cooldown": 30})
            last_used = skill_data["last_used"]
            cooldown = skill_data["cooldown"]

            # 冷却检测
            if (current_time - last_used) < cooldown:
                reward -= 120  # 冷却中使用基础惩罚
                log.debug(f"%s 冷却中 扣分！", action_name)
                if (b_status["boss_blood"] - a_status["boss_blood"]) < 1:
                    reward -= 60  # 空放追加惩罚
                return reward

            # 有效性检测
            damage_dealt = b_status["boss_blood"] - a_status["boss_blood"]
            is_effective = damage_dealt > 3  # 造成3%以上伤害

            # 更新使用记录
            self.skill_cooldowns[action_name]["last_used"] = current_time
            self.skill_effectiveness[action_name] = is_effective

            # 有效性奖励
            if is_effective:
                reward += 100
                if action_name == "SKILL_3":  # 特殊技能额外奖励
                    reward += 50
            else:
                reward -= 50  # 未命中基础惩罚

        # 动作有效性检测
        reward += self._check_action_quality(action_name, boss_attacking)

        # 自主维护连击计数器
        if action_name in ["LIGHT_ATTACK", "HEAVY_ATTACK"]:
            if time.time() - self.last_attack_time < 2.0:
                self.combo_counter += 1
            else:
                self.combo_counter = 1
            self.last_attack_time = time.time()
        else:
            self.combo_counter = 0

        # 动态平衡系数（根据当前血量调整）
        health_ratio = a_status["self_blood"] / 100.0
        attack_weight = min(2.0, 0.5 + (1 - health_ratio) * 1.5)  # 血量越低攻击权重越高
        defense_weight = 1.2 - health_ratio  # 血量越低防御权重越高

        # 攻击策略奖励（修改核心逻辑）
        reward += self._calculate_attack_reward(action_name, boss_attacking)

        # 防御逻辑（动态调整的闪避奖励）
        reward += self._calculate_defense_reward(
            action_name, injured, defense_weight, a_status
        )

        # 生存奖励（随时间递减）
        reward += self._calculate_survival_bonus(survival_time, health_ratio)

        # 核心计分逻辑
        reward += self._calculate_boss_damage(b_status, a_status)
        reward += self._calculate_self_damage(b_status, a_status)
        reward += self._handle_episode_end(a_status, done)
        reward += self._calculate_dodge_reward(action_name, injured)

        # 连击奖励（新增机制）
        reward += self._calculate_combo_reward(action_name)

        # 基础行为激励（新增）
        reward += self._basic_action_reward(action_name)

        # 记录攻击频率
        if action_name in ["LIGHT_ATTACK", "HEAVY_ATTACK"]:
            self.attack_count += 1

        # 调试日志
        self._log_debug_info(b_status, a_status, reward)

        # 记录奖励
        self.reward_tracker.add_reward(reward)

        if done:
            # 保持原有的时间奖励计算
            reward += survival_time * 8.0

            # 保持原有的状态获取方式
            real_boss_blood = self.prev_status["boss_blood"]
            real_self_blood = self.prev_status["self_blood"]

            # 保持原有的日志输出
            log.debug(
                f"对弈结束: boss_blood:{real_boss_blood:.2f} self_blood:{real_self_blood:.2f}"
            )

            # 保持原有的伤害奖励/惩罚计算
            reward += (100 - real_boss_blood) * 4.0
            reward -= (100 - real_self_blood) * 8.0

            # 保持原有的数据保存逻辑
            self.reward_tracker.end_episode(real_boss_blood)
            if self.reward_tracker.episode_num % 10 == 0:
                self.reward_tracker.save_overall_data()

        # 阶段检测
        self.post_dodge_phase = 0
        if current_time - self.last_dodge_time < 0.8:
            self.post_dodge_phase = 1
        elif current_time - self.last_dodge_time < 2.8:
            self.post_dodge_phase = 2

        # 闪避后行为奖励
        if self.post_dodge_phase > 0:
            reward += self._post_dodge_behavior_reward(action_name, injured)

        # 更新闪避时间戳
        if action_name == "DODGE":
            self.last_dodge_time = current_time
            self.safe_steps = 0

        # TISHEN技能处理
        if action_name == "TISHEN":
            boss_damage = b_status["boss_blood"] - a_status["boss_blood"]

            if not self.tishan_used:
                # 首次使用
                if boss_damage > 3:  # 造成显著伤害
                    reward += 150
                    self.tishan_effective = True
                else:
                    reward -= 50  # 未命中惩罚
                self.tishan_used = True
            else:
                # 重复使用
                reward -= 200  # 严厉惩罚
                if boss_damage < 1:
                    reward -= 100  # 空放追加惩罚

        return reward

    def _detect_boss_attack(self, before, after):
        """仅通过血量变化检测BOSS攻击"""
        # 有效BOSS攻击的条件：
        # 1. BOSS血量未明显变化（排除玩家攻击导致的BOSS掉血）
        # 2. 玩家血量下降超过3%
        boss_blood_stable = abs(before["boss_blood"] - after["boss_blood"]) < 1.0
        health_lost = before["self_blood"] - after["self_blood"]

        if boss_blood_stable and health_lost > 3:
            self.last_boss_attack_time = time.time()
            return True
        return False

    def _check_action_quality(self, action_name, boss_attacking):
        """动作质量检测"""
        penalty = 0

        # 连续前进惩罚（3次以上开始惩罚）
        if action_name == "GO_FORWARD":
            self.consecutive_forward += 1
            if self.consecutive_forward > 3:
                penalty -= 10 * (self.consecutive_forward - 3)  # 递增惩罚
        else:
            self.consecutive_forward = 0

        # 无效闪避检测（非攻击状态下闪避）
        if action_name == "DODGE" and not boss_attacking:
            if time.time() - self.attack_pattern["last_boss_attack_time"] > 2:
                penalty -= 30  # 无意义闪避惩罚

        # 全局动作有效性检测（10秒无有效动作）
        if time.time() - self.last_meaningful_action > 10:
            penalty -= 50  # 全局消极惩罚

        return penalty

    def _calculate_boss_damage(self, before, after):
        """BOSS掉血奖励"""
        damage = max(0.0, before["boss_blood"] - after["boss_blood"])
        return damage * 50.0  # 每1%血量奖励50分

    def _calculate_self_damage(self, before, after):
        """自身掉血惩罚"""
        damage = max(0.0, before["self_blood"] - after["self_blood"])
        return damage * (-80.0)  # 每1%血量扣除80分

    def _handle_episode_end(self, status, done):
        """终局结算"""
        if not done:
            return 0.0

        # 胜利奖励/失败惩罚
        if status["boss_blood"] <= 0:
            return 5000.0 + status["self_blood"] * 50  # 剩余血量加成
        else:
            return (
                -3000.0 + (status["boss_blood"] - 100) * 30
            )  # 根据BOSS剩余血量追加惩罚

    def _calculate_attack_reward(self, action, is_boss_attacking):
        """动态平衡的攻防奖励"""
        reward = 0

        # 根据血量调整策略
        defense_priority = max(
            0, 1 - (self.prev_status["self_blood"] / 100)
        )  # 血量越低防御优先级越高

        # 攻击奖励调整
        if action in ["LIGHT_ATTACK", "HEAVY_ATTACK"]:
            base_reward = 25 if action == "LIGHT_ATTACK" else 15
            reward += base_reward * (1 - defense_priority)  # 防御优先级高时降低攻击奖励

            # 危险攻击惩罚
            if is_boss_attacking:
                reward -= 30 * defense_priority  # 防御优先级越高惩罚越重

        # 闪避奖励增强
        elif action == "DODGE":
            reward += 40 * defense_priority  # 防御优先级高时增加闪避奖励

        return reward

    def _calculate_defense_reward(self, action_name, injured, weight, a_status):
        """动态防御奖励（修正参数传递）"""
        reward = 0

        # 成功闪避基础奖励
        dodge_reward = self._calculate_dodge_reward(action_name, injured)
        reward += dodge_reward * weight * 1.2

        # 危险状态额外奖励（使用传入的a_status）
        if a_status["self_blood"] < 30 and not injured:
            reward += 20 * weight

        return reward

    def _calculate_survival_bonus(self, survival_time, health_ratio):
        """生存奖励（每小时段不同策略）"""
        # 基础生存奖励（随时间递减）
        base = max(0, 10 - survival_time // 30)

        # 健康血量奖励曲线（50%血量时最高）
        health_bonus = 20 * (1 - abs(health_ratio - 0.5))

        return (base + health_bonus) * 0.5

    def _calculate_dodge_reward(self, action, injured):
        """基于受伤状态的闪避奖励"""
        reward = 0
        current_time = time.time()
        time_since_attack = current_time - self.last_boss_attack_time

        # 攻击后0.8秒内
        if time_since_attack < 0.8:
            if action == "IDLE":
                reward += 40  # 正确保持静止
                if not injured:
                    reward += 20  # 成功规避奖励
            else:
                reward -= 50  # 过早行动惩罚

        # 攻击后0.8-2.8秒
        elif time_since_attack < 2.8:
            if action == "GO_BACKWARD":
                reward += 30  # 正确远离
                if not injured:
                    reward += 10 * (2.8 - time_since_attack)  # 时间递减奖励
            elif action in ["LIGHT_ATTACK", "HEAVY_ATTACK"]:
                reward -= 60  # 危险攻击惩罚

        return reward

    def _calculate_combo_reward(self, action):
        """连击奖励"""
        reward = 0
        if action in ["LIGHT_ATTACK", "HEAVY_ATTACK"]:
            if time.time() - self.last_attack_time < 1.5:  # 连击窗口
                self.attack_pattern["combo_streak"] += 1
                # 连击奖励递增
                reward += 5 * self.attack_pattern["combo_streak"]
            else:
                self.attack_pattern["combo_streak"] = 0
        else:
            self.attack_pattern["combo_streak"] = 0  # 中断连击
        return reward

    def _basic_action_reward(self, action_name):
        """基础动作激励"""
        rewards = {
            "IDLE": -2,  # 消极惩罚
            "LIGHT_ATTACK": 5,  # 轻攻击基础奖励
            "HEAVY_ATTACK": 3,  # 重攻击基础奖励（蓄力时间更长）
            "DODGE": 8,  # 闪避基础奖励
        }
        return rewards.get(action_name, 0)

    def _log_debug_info(self, before, after, reward):
        """调试日志添加闪避信息"""
        log.debug(
            f"BOSS伤害: {before['boss_blood'] - after['boss_blood']:.2f}% | "
            f"自伤: {before['self_blood'] - after['self_blood']:.2f}% | "
            f"总奖励: {reward:.2f}"
        )

    def _post_dodge_behavior_reward(self, action, injured):
        """闪避后行为奖励"""
        reward = 0

        # 阶段1：0-0.8秒（强制冷静期）
        if self.post_dodge_phase == 1:
            if action == "IDLE":
                reward += 25  # 正确行为奖励
                if not injured:
                    reward += 15  # 安全奖励
            else:
                reward -= 40  # 过早行动惩罚
                if injured:
                    reward -= 30  # 受伤追加惩罚

        # 阶段2：0.8-2.8秒（安全移动期）
        elif self.post_dodge_phase == 2:
            # 理想行为：远离BOSS（假设GO_BACKWARD是有效动作）
            if action == "GO_BACK":
                reward += 20
                self.safe_steps += 1
            elif action in ["LIGHT_ATTACK", "HEAVY_ATTACK"]:
                reward -= 50  # 危险攻击惩罚

            # 安全持续奖励
            if not injured:
                reward += 10 * self.safe_steps  # 递增奖励
            else:
                reward -= 80  # 安全期受伤重罚
                self.safe_steps = 0

        return reward

    # 注释掉以下原始方法（保留结构）
    # def injured_index_penalty(self, injured, injured_cnt):
    #     pass
    # ... 其他被注释的方法 ...


# 调整闪避奖励参数
DODGE_WINDOW = 2.5  # 奖励窗口时长（秒）
DODGE_REWARD = 80  # 奖励分数
PENALTY = -30  # 闪避后仍然受伤的惩罚
