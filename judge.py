def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, self_energy, next_self_energy, stop, emergence_break):
    reward, done = 0, 0
    # emergence_break is used to break down training
    if next_self_blood < 10:     # self dead
        # print('鼠鼠我完了')
        # reward += -200
        # done, stop, emergence_break = 1, 0, 100
        pass

    if next_boss_blood < 10 and next_self_blood < 10:   #boss dead
        # print("boss 被你干掉啦") # 实际上自己死亡的时候，boss的血条消失的比自己的快一点
        print("鼠鼠我完了")
        reward -= 1000
        done, stop, emergence_break = 0, 0, 100

    blood_change = self_blood - next_self_blood
    if blood_change >= 5:
        print("你已掉血: %d" % (blood_change))
        if stop == 0:
            # reward += -10 * blood_change
            reward += 5 * blood_change # 鼓励换血
            stop = 1 # 防止连续取帧时一直计算掉血
    else:
        # print("打药了: %d" % (blood_change))
        # reward += 2 * blood_change
        # stop = 0
        pass

    blood_change = boss_blood - next_boss_blood
    if blood_change >= 20:
        print("boss掉血: %d" % (blood_change))
        reward += 10 * blood_change # 鼓励进攻

    blood_change = self_energy - next_self_energy
    if blood_change >= 20:
        print("体力值消耗: %d" % (blood_change))
        reward += 2 * blood_change

    return reward, done, stop, emergence_break