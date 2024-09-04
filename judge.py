def action_judge(
    boss_blood,
    next_boss_blood,
    self_blood,
    next_self_blood,
    self_energy,
    next_self_energy,
    stop,
    emergence_break,
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
        reward -= 5000
        done, stop, emergence_break = 0, 0, 100
        print("reward:%d" % reward)
        return reward, done, stop, emergence_break
    
    blood_change = self_blood - next_self_blood
    if blood_change >= 5:
        print("你已掉血: %f%%" % (blood_change))
        if stop == 0:
            # reward += -10 * blood_change
            reward -=  60  * blood_change  # 掉血还是要扣分
            stop = 1  # 防止连续取帧时一直计算掉血
            print("self_blood change reward:%d" % reward)
    else:
        reward += 20 # 不掉血就加分
        stop = 0
        print("blood no change add award: %d" % reward)
        pass

    blood_change = boss_blood - next_boss_blood
    if blood_change > 0:
        print(
            "boss掉血: %f%%|boss_blood:%f%%|next_boss_blood:%f%%"
            % (blood_change, boss_blood, next_boss_blood)
        )
        reward += 50 * blood_change  # 鼓励进攻
        print("boss_blood change reward:%d" % reward)

    blood_change = self_energy - next_self_energy
    if blood_change >= 5:
        print(
            "体力值消耗: %f%%|self_energy%f%%|next_self_energy%f%%"
            % (blood_change, self_energy, next_self_energy)
        )
        reward += 5  * blood_change
        print("energy change reward:%d" % reward)

    print("one action final reward:%d" % reward)
    return reward, done, stop, emergence_break
