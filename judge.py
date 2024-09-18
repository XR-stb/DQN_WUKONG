import time
from context import Context
from log import log
from tracker import RewardTracker

# 初始化数据跟踪器
reward_tracker = RewardTracker(train_data_dir='train_data/data')

def action_judge(ctx: Context):
    log(f"当前状态: 自己的血量={ctx.self_blood}, Boss的血量={ctx.boss_blood}, 自己的体力值={ctx.self_energy}, paused={ctx.paused}")
    log(f"未来状态: 自己的血量={ctx.next_self_blood}, Boss的血量={ctx.next_boss_blood}, 自己的体力值={ctx.next_self_energy}, paused={ctx.paused}")
    ctx.reward = 0
    current_time = time.time()
    if current_time - ctx.last_reward_time < 0.1:
        log(f"距离上次奖励计算不足0.1秒，跳过本次奖励计算。")
        return ctx

    # 更新上次奖励计算时间
    ctx.last_reward_time = current_time

    reward_weight = (current_time - ctx.begin_time) / 5 # 存活时间越久，奖励权重越高，
    log(f"reward_weight={reward_weight}")



    # 自己死亡的情况
    # 1.自己血量到0
    # 2.boss血量异常降低
    dead_condition1 = ctx.next_self_blood <= 1
    dead_condition2 = ctx.boss_blood - ctx.next_boss_blood > 40

    if dead_condition2 or dead_condition1:
        if dead_condition2 and not dead_condition1:
            log(f"boss血量异常降低 判断为吗喽已死亡")

        boss_remain_blood = max(ctx.next_boss_blood, ctx.boss_blood)

        dead_reward = 50000
        boss_remain_blood_award = (100 - boss_remain_blood) * (dead_reward / 100)
        ctx.reward += boss_remain_blood_award
        ctx.reward -= dead_reward
        ctx.done, ctx.stop, ctx.emergence_break = 1, 0, 100
        ctx.dodge_weight, ctx.attack_weight = 1, 1

        log(f"吗喽已死亡, 当前状态: done={ctx.done}, stop={ctx.stop}, "
            f"emergence_break={ctx.emergence_break}, dodge_weight={ctx.dodge_weight}, "
            f"attack_weight={ctx.attack_weight}")
        
        reward_tracker.add_reward(ctx.reward)  # 添加当前奖励
        reward_tracker.end_episode(boss_remain_blood)  # 每局结束时记录 Boss 血量

        # 每10局保存一次
        if reward_tracker.episode_num % 10 == 0:
            reward_tracker.save_overall_data()
        return ctx

    # 战胜boss
    if ctx.self_blood > 1 and ctx.next_boss_blood < 1:

        boss_remain_blood = ctx.boss_blood
        ctx.reward += 50000
        ctx.done, ctx.stop, ctx.emergence_break = 1, 0, 100
        ctx.dodge_weight, ctx.attack_weight = 1, 1

        log(f"boss已死亡, 当前状态: done={ctx.done}, stop={ctx.stop}, "
            f"emergence_break={ctx.emergence_break}, dodge_weight={ctx.dodge_weight}, "
            f"attack_weight={ctx.attack_weight}")
        
        reward_tracker.add_reward(ctx.reward)  # 添加当前奖励
        reward_tracker.end_episode(boss_remain_blood)  # 每局结束时记录 Boss 血量

        # 每10局保存一次
        if reward_tracker.episode_num % 10 == 0:
            reward_tracker.save_overall_data()
        return ctx


    # 自己掉血的情况
    blood_change = ctx.self_blood - ctx.next_self_blood
    if blood_change > 5:
        ctx.reward -= 100 * blood_change  * ctx.dodge_weight * reward_weight  # 每局掉血扣分要抵消掉攻击的分
        ctx.attack_weight = max(1, ctx.attack_weight - blood_change)
        ctx.dodge_weight = min(1, ctx.dodge_weight - blood_change)
        ctx.stop = 1  # 防止连续帧重复计算
        log(f"自己掉血：{blood_change}%。奖励减少 {10 * blood_change}。当前权重: attack_weight={ctx.attack_weight}, "
            f"dodge_weight={ctx.dodge_weight}, stop={ctx.stop}")
    else:
        ctx.stop = 0

    # boss掉血的情况
    blood_change = ctx.boss_blood - ctx.next_boss_blood
    if blood_change > 0 and blood_change < 20:  # 伤害不可能太高，太高就是计算出错了
        add_award = 100 * blood_change * ctx.attack_weight * reward_weight
        ctx.reward += add_award  # 鼓励攻击boss
        ctx.attack_weight = min(1, ctx.attack_weight + blood_change)
        log(f"Boss掉血：{blood_change}%。奖励增加 {add_award}。当前 attack_weight={ctx.attack_weight}")
    elif blood_change > 5:
        log(f"boss 掉血{blood_change}%过高")

    # 能量消耗情况
    energy_change = ctx.self_energy - ctx.next_self_energy
    if energy_change > 5 and energy_change < 30:
        ctx.reward -= 2 * energy_change * ctx.dodge_weight
        ctx.dodge_weight = min(1, ctx.dodge_weight + energy_change / 10)
        log(f"能量消耗：{energy_change}%。奖励减少 {5 * energy_change * ctx.dodge_weight}。当前 dodge_weight={ctx.dodge_weight}")
    elif energy_change < 5:
        log(f"能量消耗{energy_change}% 忽略")
    else:
        log(f"能量消耗{energy_change}%过高 异常")

    # 最终奖励计算
    log(f"one action final reward: {ctx.reward}")
    
    # 添加当前奖励数据到 tracker
    reward_tracker.add_reward(ctx.reward)
    
    return ctx
