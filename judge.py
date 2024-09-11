
import time
from context import Context
from log import log

def action_judge(context: Context):
    log(f"当前状态: 自己的血量={context.self_blood}, Boss的血量={context.boss_blood}, 自己的体力值={context.self_energy}")
    context.reward = 0
    # 如果上次奖励计算时间距离现在不足0.5秒，则不进行奖励计算
    current_time = time.time()
    if current_time - context.last_reward_time < 0.1:
        log(f"距离上次奖励计算不足0.1秒，跳过本次奖励计算。")
        return context

    # 更新上次奖励计算时间
    context.last_reward_time = current_time


    # 自己死亡的情况
    if context.next_self_blood <= 0:
        log("自己死亡，奖励无更改。")
        # context.reward -= 1000  # 严重惩罚
        # context.done, context.stop, context.emergence_break = 1, 0, 100
        # return context

    # boss血量骤减的情况
    if context.boss_blood - context.next_boss_blood > 5:
        log("Boss血量骤减，跳过处理。")
        # return context

    # 自己死亡, boss血条消失的要快一些
    if context.next_boss_blood <= 5 and context.next_self_blood <= 5 and int(time.time()) - context.begin_time > 2:
        context.reward -= 500
        context.done, context.stop, context.emergence_break = 0, 0, 100
        context.dodge_weight, context.attack_weight = 1, 1
        log(f"吗喽已死亡, 当前状态: done={context.done}, stop={context.stop}, "
            f"emergence_break={context.emergence_break}, dodge_weight={context.dodge_weight}, "
            f"attack_weight={context.attack_weight}")
        return context

    # 自己掉血的情况
    blood_change = context.self_blood - context.next_self_blood
    if blood_change > 5:
        context.reward -= 10 * blood_change  # 每掉1%血量扣10分
        context.attack_weight = max(1, context.attack_weight - blood_change / 10)
        context.dodge_weight = min(1, context.dodge_weight + blood_change / 10)
        context.stop = 1  # 防止连续帧重复计算
        log(f"自己掉血：{blood_change}%。奖励减少 {10 * blood_change}。当前权重: attack_weight={context.attack_weight}, "
            f"dodge_weight={context.dodge_weight}, stop={context.stop}")
    else:
        context.reward += 10  # 未掉血增加奖励
        context.dodge_weight = max(0, context.dodge_weight - 0.1)
        context.stop = 0
        log(f"未掉血。奖励增加10。当前 dodge_weight={context.dodge_weight}, stop={context.stop}")

    # boss掉血的情况
    blood_change = context.boss_blood - context.next_boss_blood
    if blood_change > 0 and blood_change < 5: # 伤害不可能太高，太高就是计算出错了
        add_award = 100 * blood_change * context.attack_weight
        context.reward += add_award  # 鼓励攻击boss
        context.attack_weight = min(1, context.attack_weight + blood_change / 10)
        log(f"Boss掉血：{blood_change}%。奖励增加 {add_award}。当前 attack_weight={context.attack_weight}")

    # 能量消耗情况
    energy_change = context.self_energy - context.next_self_energy
    if energy_change > 5:
        context.reward -= 2 * energy_change * context.dodge_weight
        context.dodge_weight = min(1, context.dodge_weight + energy_change / 10)
        log(f"能量消耗：{energy_change}%。奖励减少 {5 * energy_change * context.dodge_weight}。当前 dodge_weight={context.dodge_weight}")

    # 最终奖励计算
    log(f"one action final reward: {context.reward}")
    return context
