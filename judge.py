import time
from log import log
from tracker import RewardTracker


class ActionJudge:
    def __init__(self):
        self.reward_tracker = RewardTracker(train_data_dir="train_data/data")
        self.prev_action_name = ""
        self.prev_survival_time = 0
        self.prev_status = {}
        self.prev_injured = False
        self.same_action_count = 0  # 添加连续动作计数器

    def reset(self):
        self.prev_action_name = ""
        self.prev_survival_time = 0
        self.prev_status = {}
        self.prev_status["boss_blood"] = 100
        self.prev_status["self_blood"] = 100
        self.prev_injured = False

    # 受到攻击的次数越多，扣分越多， 加速完美闪避学习
    def injured_index_penalty(self, injured, injured_cnt):
        if injured:
            log.info(f"INJURED | COUNT: {injured_cnt}")
            return injured_cnt * 50
        return 0

    def _calculate_blood_advantage_reward(self, self_blood, boss_blood):
        """计算血量优势奖励"""
        if self_blood > boss_blood:
            return 200  # 保持血量优势的奖励
        else:
            return -50  # 血量劣势的惩罚

    def _calculate_done_reward(self, survival_time):
        """计算对局结束时的奖励"""
        reward = survival_time * 8.0  # 时间奖励

        real_boss_blood = self.prev_status["boss_blood"]
        real_self_blood = self.prev_status["self_blood"]

        # 结算时的胜负判定奖励
        if real_boss_blood < 10 and real_self_blood > 10:
            reward += 3000  # 增加胜利奖励
        else:
            reward -= 2000  # 增加失败惩罚

        # 最终血量差距奖励
        blood_diff = real_self_blood - real_boss_blood
        reward += blood_diff * 10.0  # 根据最终血量差距给予额外奖励/惩罚

        log.debug(
            f"GAME OVER | "
            f"BOSS HP: {real_boss_blood:.2f} | "
            f"SELF HP: {real_self_blood:.2f} | "
            f"SURVIVAL TIME: {survival_time:.2f}"
        )
        return reward, real_boss_blood

    def _process_events(self, events):
        """处理事件队列的奖励"""
        reward = 0
        for event in events:
            if event["event"] == "self_blood":
                # 受到伤害时的惩罚加大
                reward += event["relative_change"] * 2.0  # 原来是1.0
            elif event["event"] == "boss_blood":
                # 对boss造成伤害时的奖励加大
                reward -= event["relative_change"] * 6.0  # 原来是4.0
        return reward

    def _check_skill_cooldown(self, cur_status, action_name):
        """检查技能冷却并计算奖励"""
        reward = 0
        skill_mapping = {
            "SKILL_1": "skill_1",
            "SKILL_2": "skill_2",
            "SKILL_3": "skill_3",
            "SKILL_4": "skill_4",
            "TISHEN": "skill_ts",
            "FABAO": "skill_fb",
            "STEALTH_CHARGE": "skill_2",
        }

        if action_name in skill_mapping:
            skill_key = skill_mapping[action_name]
            if not cur_status[skill_key]:
                reward -= 100
                # 特别针对STEALTH_CHARGE的额外惩罚
                if action_name == "STEALTH_CHARGE":
                    reward -= 200  # 增加使用冷却中技能的惩罚
            elif cur_status[skill_key]:
                # 降低STEALTH_CHARGE的奖励
                if action_name == "STEALTH_CHARGE":
                    reward += 30  # 降低奖励
                else:
                    reward += 100  # 其他技能保持原有奖励

        return reward

    def _handle_special_actions(self, action_name, cur_status, injured, injured_cnt):
        """处理特殊动作的奖励"""
        reward = 0

        # 记录连续使用相同动作的次数
        if action_name == self.prev_action_name:
            self.same_action_count = getattr(self, "same_action_count", 0) + 1
        else:
            self.same_action_count = 0

        # 对连续使用相同动作施加惩罚
        if self.same_action_count > 2:  # 连续使用3次以上同一动作
            reward -= self.same_action_count * 20  # 递增惩罚

        if action_name == "DRINK_POTION":
            reward += self._handle_drink_potion(cur_status)
        elif action_name == "DODGE":
            reward += self._handle_dodge(injured, injured_cnt)
        elif action_name == "QIESHOU":
            reward += self._handle_qieshou(cur_status, injured)
        elif action_name == "HEAVY_ATTACK":
            reward += self._handle_heavy_attack(cur_status, injured, injured_cnt)
        elif action_name in ["LIGHT_ATTACK", "ATTACK_DODGE", "FIVE_HIT_COMBO"]:
            reward += self._handle_light_attacks(cur_status)
        elif action_name == "SKILL_3":
            reward += self._handle_skill_3(injured, injured_cnt)
        elif action_name == "STEALTH_CHARGE":
            reward += self._handle_stealth_charge(cur_status, injured)

        return reward

    def _handle_drink_potion(self, cur_status):
        """处理喝药动作的奖励"""
        if cur_status["self_blood"] > 90:
            return -100  # 惩罚满血喝药
        elif cur_status["self_blood"] < 40:
            return 50  # 奖励血量低时喝药
        elif cur_status["hulu"] < 10:
            return -100  # 惩罚药水不足时喝药
        return 0

    def _handle_dodge(self, injured, injured_cnt):
        """处理闪避动作的奖励"""
        reward = -self.injured_index_penalty(injured, injured_cnt)
        if not injured:
            reward += 50  # 闪避成功奖励
        else:
            reward -= 20  # 闪避失败惩罚

        if self.prev_injured:
            reward += 30  # 受伤后优先闪避奖励
        return reward

    def _handle_qieshou(self, cur_status, injured):
        """处理切手动作的奖励"""
        reward = 0
        if cur_status["gunshi1"]:
            reward += 20  # 有豆时使用切手奖励
            if not injured:
                reward += 30  # 未受伤额外奖励
        return reward

    def _handle_heavy_attack(self, cur_status, injured, injured_cnt):
        """处理重击动作的奖励"""
        reward = 0
        if not cur_status["gunshi1"]:
            reward -= 20  # 无豆重击惩罚
        elif cur_status["gunshi3"]:
            reward += 20  # 三豆重击奖励

        if cur_status["gunshi1"] and self.prev_action_name == "QIESHOU":
            reward -= self.injured_index_penalty(injured, injured_cnt)
            if not injured:
                reward += 30  # 切手追击成功奖励
        return reward

    def _handle_light_attacks(self, cur_status):
        """处理轻攻击相关动作的奖励"""
        if not cur_status["gunshi3"]:
            return 5  # 未满三豆时的攒豆奖励
        return 0

    def _handle_skill_3(self, injured, injured_cnt):
        """处理技能3的奖励"""
        reward = 0
        if self.prev_action_name == "SKILL_1":
            reward += 10  # 定身后召唤猴子奖励

        reward -= self.injured_index_penalty(injured, injured_cnt)
        if injured:
            reward -= 10  # 召唤时受伤惩罚
        return reward

    def _handle_stealth_charge(self, cur_status, injured):
        """处理隐身蓄力攻击的奖励"""
        reward = 0

        # 如果受伤则给予更大惩罚
        if injured:
            reward -= 100

        # 根据当前血量状态调整使用策略
        if cur_status["self_blood"] < 50:
            reward -= 50  # 血量低时不建议使用这个技能

        # 检查其他技能是否可用，鼓励使用其他技能
        available_skills = sum(
            [
                cur_status["skill_1"],
                cur_status["skill_3"],
                cur_status["skill_ts"],
                cur_status["skill_fb"],
            ]
        )
        if available_skills >= 2:  # 如果有两个或更多其他技能可用
            reward -= 30  # 降低使用STEALTH_CHARGE的倾向

        return reward

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
    ):
        reward = 0
        reward -= self.injured_index_penalty(injured, injured_cnt)

        # 计算每一步的血量优势奖励
        blood_advantage_reward = self._calculate_blood_advantage_reward(
            next_status["self_blood"], next_status["boss_blood"]
        )
        reward += blood_advantage_reward

        if done:
            done_reward, real_boss_blood = self._calculate_done_reward(survival_time)
            reward += done_reward

            # 本局结束记录
            self.reward_tracker.end_episode(real_boss_blood)
            if self.reward_tracker.episode_num % 10 == 0:
                self.reward_tracker.save_overall_data()
        else:
            # 存活时间奖励
            reward += survival_time - self.prev_survival_time

            # 处理事件
            reward += self._process_events(events)

            # 处理技能冷却
            reward += self._check_skill_cooldown(cur_status, action_name)

            # 处理特殊动作
            reward += self._handle_special_actions(
                action_name, cur_status, injured, injured_cnt
            )

            # 体能检测（加大惩罚）
            if next_status["self_energy"] < 10:
                reward -= 50  # 原来是30

            # 更新状态
            self.prev_status = next_status.copy()
            self.prev_action_name = action_name
            self.prev_survival_time = survival_time
            self.prev_injured = injured

        log.debug(
            (
                f"TIME: {survival_time:.2f} | "
                f"BOSS HP: {next_status['boss_blood']:.2f} | "
                f"SELF HP: {next_status['self_blood']:.2f} | "
                f"ACTION: {action_name} | "
                f"BLOOD ADV: {blood_advantage_reward:+.2f} | "  # 添加+号显示正负
                f"REWARD: {reward:+.2f}"  # 添加+号显示正负
            )
        )
        self.reward_tracker.add_reward(reward)
        return reward
