import time
from log import log
from tracker import RewardTracker


class ActionJudge:
    def __init__(self):
        self.reward_tracker = RewardTracker(train_data_dir='train_data/data')
        self.prev_action_name = ''
        self.prev_survival_time = 0
        self.prev_status = {}
        self.prev_injured = False

    def reset(self):
        self.prev_action_name = ''
        self.prev_survival_time = 0
        self.prev_status = {}
        self.prev_status['boss_blood'] = 100
        self.prev_status['self_blood'] = 100
        self.prev_injured = False


    def judge(
        self,
        action_name,
        injured,
        b_status, #before_action_status
        a_status, #after_action_status
        events,
        survival_time,
        done):

 
        reward = 0

        if done:
            # 时间奖励 鼓励更久存活
            reward += survival_time * 4.0

            real_boss_blood = self.prev_status['boss_blood']
            real_self_blood = self.prev_status['self_blood']
            log(f"对弈结束: boss_blood:{real_boss_blood:.2f} self_blood:{real_self_blood:.2f}")


            #奖励对boss造成伤害
            reward += (100 - real_boss_blood)*4.0
            #惩罚自己受到伤害
            reward -= (100 - real_self_blood)*2.0

            # 本局结束 记录boss血量
            self.reward_tracker.end_episode(real_boss_blood)

            # 每10局保存reward_tracker一次
            if self.reward_tracker.episode_num % 10 == 0:
                self.reward_tracker.save_overall_data()

        else:
            #对局中

            # 时间奖励 鼓励更久存活
            reward += survival_time - self.prev_survival_time


            # 处理事件队列

            for event in events:
                if event['event'] == 'self_blood':
                    self_blood_change = event['relative_change']
                    reward += self_blood_change
                elif event['event'] == 'boss_blood':
                    boss_blood_change = event['relative_change']
                    reward -= boss_blood_change * 4.0


            # 处理技能冷却

            if b_status['skill_1'] == False and action_name == 'SKILL_1':
                reward -= 100
            elif b_status['skill_2'] == False and action_name == 'SKILL_2':
                reward -= 100
            elif b_status['skill_3'] == False and action_name == 'SKILL_3':
                reward -= 100
            elif b_status['skill_4'] == False and action_name == 'SKILL_4':
                reward -= 100
            elif b_status['skill_ts'] == False and action_name == 'TISHEN':
                reward -= 100
            elif b_status['skill_fb'] == False and action_name == 'FABAO':
                reward -= 100
            elif b_status['skill_2'] == False and action_name == 'STEALTH_CHARGE':
                reward -= 100

            if b_status['skill_1'] == True and action_name == 'SKILL_1':
                reward += 100
            elif b_status['skill_2'] == True and action_name == 'SKILL_2':
                reward += 100
            elif b_status['skill_3'] == True and action_name == 'SKILL_3':
                reward += 100
            elif b_status['skill_4'] == True and action_name == 'SKILL_4':
                reward += 100
            elif b_status['skill_ts'] == True and action_name == 'TISHEN':
                reward += 100
            elif b_status['skill_fb'] == True and action_name == 'FABAO':
                reward += 100
            elif b_status['skill_2'] == True and action_name == 'STEALTH_CHARGE':
                reward += 100

            # 特殊动作规则
            if action_name == 'DRINK_POTION':
                if b_status['self_blood'] > 90:
                    # 惩罚 满血 喝药
                    reward -= 100
                elif b_status['self_blood'] < 40:
                    # 奖励 血量低时 喝药
                    reward += 50
                elif b_status['hulu'] < 10:
                    # 喝光了 还在喝
                    reward -= 100
            elif action_name == 'DODGE':
                if not injured:
                    # 闪避 且没挨打
                    reward += 5
                else:
                    # 闪避时间不对
                    reward -= 5

                if self.prev_injured:
                    # 刚刚受伤了 这次优先闪避
                    reward += 10
            elif action_name == 'QIESHOU':
                if b_status['gunshi1'] == True:
                    # 鼓励有豆的时候使用切手 有可能打出识破
                    reward += 20
                    if not injured:
                        # 还没受伤 很有可能是因为识破了
                        reward += 30
            elif action_name == 'HEAVY_ATTACK':
                if b_status['gunshi1'] == False:
                    # 没豆 打什么重击
                    reward -= 20
                elif b_status['gunshi3'] == True:
                    # 鼓励下 3豆重击
                    reward += 20
                if b_status['gunshi1'] == True and self.prev_action_name == 'QIESHOU':
                    if not injured:
                        # 切手 追击 还没受伤 鼓励
                        reward += 30
            elif action_name == 'LIGHT_ATTACK':
                if b_status['gunshi3'] == False:
                    # 没满三豆 鼓励可以攒棍势的攻击
                    reward += 5
            elif action_name == 'ATTACK_DODGE':
                if b_status['gunshi3'] == False:
                    # 没满三豆 鼓励可以攒棍势的攻击
                    reward += 5
            elif action_name == 'FIVE_HIT_COMBO':
                if b_status['gunshi3'] == False:
                    # 没满三豆 鼓励可以攒棍势的攻击
                    reward += 5
            elif action_name == 'SKILL_3':
                if self.prev_action_name == 'SKILL_1':
                    # 定身后 召唤猴子 安全
                    reward += 10
                if injured:
                    # 召唤猴子的时候 挨打了 时机不对
                    reward -= 10


            # 体能检测
            if a_status['self_energy'] < 10:
                # 不鼓励把体力用光了
                reward -= 30




            self.prev_status = a_status.copy()
            self.prev_action_name = action_name
            self.prev_survival_time = survival_time
            self.prev_injured = injured


        log((
            f"info: time:{survival_time:.2f} "
            f"boss_blood:{a_status['boss_blood']:.2f} "
            f"self_blood:{a_status['self_blood']:.2f} "
            f"reward:{reward:.2f}"
        ))
        self.reward_tracker.add_reward(reward)
        return reward

