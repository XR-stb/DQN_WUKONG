import BloodCommon


def action_judge(
    boss_blood,
    next_boss_blood,
    self_blood,
    next_self_blood,
    self_energy,
    next_self_energy,
    stop,
    emergence_break,
    dodge_weight,
    attack_weight,
    init_medicine_nums,
):
    reward, done = 0, 0
    # emergence_break is used to break down training
    if next_self_blood < 0.05:  # self dead
        # print('鼠鼠我完了')
        # reward += -200
        # done, stop, emergence_break = 1, 0, 100
        stop = 0
        pass

    if next_boss_blood < 0.05 and next_self_blood < 0.05:  # boss dead
        # print("boss 被你干掉啦") # 实际上自己死亡的时候，boss的血条消失的比自己的快一点
        print("鼠鼠我完了")
        reward -= 4000
        done, stop, emergence_break, dodge_weight, attack_weight = 0, 0, 100, 1, 1
        print("reward:%d" % reward)
        return reward, done, stop, emergence_break, dodge_weight, attack_weight, 4

    blood_change = self_blood - next_self_blood
    if blood_change >= 5:
        print("你已掉血: %f%%" % (blood_change))
        attack_weight = min(
            1, attack_weight - blood_change / 5
        )  # 受到伤害减少攻击的得分系数，避免一直进行攻击而受到伤害，不会闪避
        dodge_weight += 1  # 受到伤害也要增加闪避扣分系数，让它合理进行闪避
        if stop == 0:
            # reward += -10 * blood_change
            reward -= 70 * blood_change  # 掉血还是要扣分
            stop = 1  # 防止连续取帧时一直计算掉血
            print("self_blood change reward:%d" % reward)
    else:
        reward += 100  # 不掉血就加分
        stop = 0
        dodge_weight = min(
            1, dodge_weight - 2
        )  # 没有受到伤害，说明闪避合理，减少闪避扣分系数
        print("blood no change add award: %d" % reward)
        pass

    blood_change = boss_blood - next_boss_blood
    if blood_change > 0:
        print(
            "boss掉血: %f%%|boss_blood:%f%%|next_boss_blood:%f%%"
            % (blood_change, boss_blood, next_boss_blood)
        )

        attack_weight += blood_change  # 攻击boss掉血增加攻击系数分，提高攻击欲望
        reward += 90 * blood_change * attack_weight  # 鼓励进攻
        print("boss_blood change reward:%d" % reward)

    blood_change = self_energy - next_self_energy
    if blood_change >= 5:
        print(
            "体力值消耗: %f%%|self_energy%f%%|next_self_energy%f%%"
            % (blood_change, self_energy, next_self_energy)
        )
        reward += -20 * dodge_weight * blood_change
        print("energy change reward:%d" % reward)

    print("one action final reward:%d" % reward)
    return (
        reward,
        done,
        stop,
        emergence_break,
        dodge_weight,
        attack_weight,
        init_medicine_nums,
    )
