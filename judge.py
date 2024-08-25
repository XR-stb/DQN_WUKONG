def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, self_energy, next_self_energy, stop, emergence_break):
    reward, done = 0, 0
    # emergence_break is used to break down training
    if next_self_blood < 5:     # self dead
        print('鼠鼠我完了')
        if emergence_break < 2:
            reward += -300
            done = 1
            stop = 0
            emergence_break += 1
        else:
            reward += -10
            done, stop, emergence_break = 1, 0, 0

    if next_boss_blood < 5:   #boss dead
        print("boss 被你干掉啦")
        if emergence_break < 2:
            reward += 200
            done = 0
            stop = 0
            emergence_break += 1
        else:
            reward += 20
            done, stop, emergence_break = 0, 0, 0

    blood_change = self_blood - next_self_blood
    if blood_change >= 5:
        print("你已掉血: %d" % (blood_change))
        if stop == 0:
            reward += -10 * blood_change
            stop = 1 # 防止连续取帧时一直计算掉血
    else:
        print("打药了: %d" % (blood_change))
        reward += 2 * blood_change
        stop = 0

    blood_change = boss_blood - next_boss_blood
    if blood_change >= 50:
        print("boss掉血: %d" % (blood_change))
        reward += 10 * blood_change

    blood_change = self_energy - next_self_energy
    if blood_change >= 30:
        print("体力值消耗: %d" % (blood_change))
        reward += 3 * blood_change

    return reward, done, stop, emergence_break