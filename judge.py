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
            log.info(f"injured_cnt: {injured_cnt}")
            return injured_cnt * 50
        return 0

    def judge(
        self,
        action_name,
        injured,
        b_status,  # before_action_status
        a_status,  # after_action_status
        events,
        survival_time,
        done,
        injured_cnt,
    ):

        reward = 0
        reward -= self.injured_index_penalty(injured, injured_cnt)
        if done:
            # 时间奖励 鼓励更久存活
            reward += survival_time * 8.0  # 一局目前大概在20s不到

            real_boss_blood = self.prev_status["boss_blood"]
            real_self_blood = self.prev_status["self_blood"]
            log.debug(
                f"对弈结束: boss_blood:{real_boss_blood:.2f} self_blood:{real_self_blood:.2f}"
            )

            # 奖励对boss造成伤害
            reward += (100 - real_boss_blood) * 4.0
            # 惩罚自己受到伤害
            reward -= (
                100 - real_self_blood
            ) * 8.0  # 两倍扣伤，避免一直被boss打，不会闪避

            # 本局结束 记录boss血量
            self.reward_tracker.end_episode(real_boss_blood)

            # 每10局保存reward_tracker一次
            if self.reward_tracker.episode_num % 10 == 0:
                self.reward_tracker.save_overall_data()

        else:
            # 对局中

            # 时间奖励 鼓励更久存活
            reward += survival_time - self.prev_survival_time

            # 处理事件队列

            for event in events:
                if event["event"] == "self_blood":
                    self_blood_change = event["relative_change"]
                    reward += self_blood_change
                elif event["event"] == "boss_blood":
                    boss_blood_change = event["relative_change"]
                    reward -= boss_blood_change * 4.0

            # 处理技能冷却

            if b_status["skill_1"] == False and action_name == "SKILL_1":
                reward -= 100
            elif b_status["skill_2"] == False and action_name == "SKILL_2":
                reward -= 100
            elif b_status["skill_3"] == False and action_name == "SKILL_3":
                reward -= 100
            elif b_status["skill_4"] == False and action_name == "SKILL_4":
                reward -= 100
            elif b_status["skill_ts"] == False and action_name == "TISHEN":
                reward -= 200
            elif b_status["skill_fb"] == False and action_name == "FABAO":
                reward -= 100
            elif b_status["skill_2"] == False and action_name == "STEALTH_CHARGE":
                reward -= 200
            elif b_status["skill_2"] == False and action_name == "RUN_CHARGE":
                reward += 100

            if b_status["skill_1"] == True and action_name == "DING_CHARGE":
                reward += 100
            elif b_status["skill_2"] == True and action_name == "SKILL_2":
                reward += 100
            elif b_status["skill_3"] == True and action_name == "SKILL_3":
                if self.prev_action_name == "SKILL_1":
                    reward += 100
                reward -= 400
            elif b_status["skill_4"] == True and action_name == "SKILL_4":
                reward -= 100
            elif b_status["skill_ts"] == True and action_name == "TISHEN":
                reward += 100
            elif b_status["skill_fb"] == True and action_name == "FABAO":
                reward += 100
            elif b_status["skill_2"] == True and action_name == "STEALTH_CHARGE":
                reward += 120
            elif b_status["skill_2"] == True and b_status["self_blood"] < 30 and action_name == "STEALTH_DRINK":
                reward += 150
            elif b_status["skill_2"] == True and b_status["self_blood"] > 30 and action_name == "STEALTH_DRINK":
                reward -= 200

            if self.prev_injured:
                if action_name == "DODGE_THREE" or action_name == "DODGE_TWO":
                    # 刚刚受伤了 这次优先闪避
                    reward += 60
                elif action_name == "DRINK_POTION":
                    reward += 70

            elif self.prev_action_name == "SKILL_1":
                if action_name == "STEALTH_CHARGE" or action_name == "STEALTH_ATTACK":
                    reward += 100

            if b_status["gunshi3"] == False and action_name == "STEALTH_ATTACK":
                reward += 100

            # 特殊动作规则
            if action_name == "DRINK_POTION":
                # if b_status["self_blood"] > 70:
                #     # 惩罚 满血 喝药
                #     reward -= 50
                if b_status["self_blood"] < 60:
                    # 奖励 血量低时 喝药WWW
                    reward += 150
                elif b_status["self_blood"] < 40:
                    # 奖励 血量低时 喝药
                    reward += 250
                elif b_status["hulu"] < 10:
                    # 喝光了 还在喝
                    reward -= 50
            elif action_name == "DODGE":
                if not injured:
                    # 闪避 且没挨打
                    reward += 15
                else:
                    # 闪避时间不对
                    reward -= 10
            elif action_name == "QIESHOU":
                if b_status["gunshi1"] == True:
                    # 鼓励有豆的时候使用切手 有可能打出识破
                    reward += 15
                    reward -= self.injured_index_penalty(injured, injured_cnt)
                    if not injured:
                         # 还没受伤 很有可能是因为识破了
                         reward += 20
                elif b_status["gunshi1"] == False:
                    reward -= 50
            elif action_name == "HEAVY_ATTACK":
                if b_status["gunshi1"] == False:
                    # 没豆 打什么重击
                    reward -= 50
                elif b_status["gunshi3"] == True:
                    # 鼓励下 3豆重击
                    reward += 60
                if b_status["gunshi1"] == True and self.prev_action_name == "QIESHOU":
                    reward -= self.injured_index_penalty(injured, injured_cnt)
                    if not injured:
                        # 切手 追击 还没受伤 鼓励
                        reward += 30
            elif action_name == "LIGHT_ATTACK":
                if b_status["gunshi3"] == False:
                    # 没满三豆 鼓励可以攒棍势的攻击
                    reward += 25
            elif action_name == "ATTACK_DODGE":
                if b_status["gunshi3"] == False:
                    # 没满三豆 鼓励可以攒棍势的攻击
                    reward += 65
            elif action_name == "FIVE_HIT_COMBO":
                if b_status["gunshi3"] == False:
                    # 没满三豆 鼓励可以攒棍势的攻击
                    reward += 5
            elif action_name == "RUN_CHARGE":
                if b_status["gunshi3"] == False:
                    reward += 70

            # 体能检测
            if a_status["self_energy"] < 10:
                # 不鼓励把体力用光了
                reward -= 30

            self.prev_status = a_status.copy()
            self.prev_action_name = action_name
            self.prev_survival_time = survival_time
            self.prev_injured = injured

        log.debug(
            (
                f"info: time:{survival_time:.2f} "
                f"boss_blood:{a_status['boss_blood']:.2f} "
                f"self_blood:{a_status['self_blood']:.2f} "
                f"reward:{reward:.2f}"
            )
        )
        self.reward_tracker.add_reward(reward)
        return reward
