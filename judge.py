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
            reward += survival_time
            real_boss_blood = self.prev_status['boss_blood']
            real_self_blood = self.prev_status['self_blood']
            log(f"对弈结束: boss_blood:{real_boss_blood:.2f} self_blood:{real_self_blood:.2f}")
            if real_boss_blood < 5 and real_self_blood > 5:
                log(f"判断为胜利!")
                reward += 1000
            else:
                log(f"判断为失败!")
                reward -= 1000

            # 本局结束 记录boss血量
            self.reward_tracker.end_episode(real_boss_blood)

            # 每10局保存reward_tracker一次
            if self.reward_tracker.episode_num % 10 == 0:
                reward_tracker.save_overall_data()

        else:
            #对局中

            # 时间奖励 鼓励更久存活
            reward += survival_time - self.prev_survival_time









            self.prev_status = b_status.copy()
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

