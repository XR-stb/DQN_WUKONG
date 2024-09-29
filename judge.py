import time
from log import log
from tracker import RewardTracker

'''
# 初始化数据跟踪器
reward_tracker = RewardTracker(train_data_dir='train_data/data')

reward_tracker.add_reward(ctx.reward)  # 添加当前奖励
reward_tracker.end_episode(boss_remain_blood)  # 每局结束时记录 Boss 血量

# 每10局保存一次
if reward_tracker.episode_num % 10 == 0:
    reward_tracker.save_overall_data()
return ctx
'''

def action_judge(
    action_name,
    before_action_status,
    after_action_status,
    events,
    survival_time,
    done):
    
    print(survival_time)
    reward = survival_time  
    return reward

