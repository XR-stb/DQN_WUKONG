# coding=utf-8
import numpy as np
from grabscreen import grab_screen
import cv2
import time
import actions
from getkeys import key_check
import random
from DQN_tensorflow_gpu import DQN
import os
import pandas as pd
from restart import restart
import random
import tensorflow.compat.v1 as tf
import BloodCommon
import judge

DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"
WIDTH = 96
HEIGHT = 88

# station window_size
window_size = BloodCommon.get_main_screen_window()

# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing
action_size = 5

EPISODES = 3000
big_BATCH_SIZE = 16
# times that evaluate the network
UPDATE_STEP = 50

# used to save log graph
num_step = 0

# used to update target Q network
target_step = 0

# used to stop training
paused = True

if __name__ == "__main__":
    # DQN init
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)

    # paused at the begin
    paused = actions.pause_game(paused)

    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    emergence_break = 0

    for episode in range(EPISODES):
        screen_gray = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_BGR2GRAY)

        # change graph to WIDTH * HEIGHT for station input
        station = cv2.resize(screen_gray, (WIDTH, HEIGHT))

        # count init blood
        boss_blood = BloodCommon.get_boss_blood_window().blood_count()

        self_blood = BloodCommon.get_self_blood_window().blood_count()
        self_energy = BloodCommon.get_self_energy_window().blood_count()

        # used to update target Q network
        target_step = 0

        done = 0
        total_reward = 0
        stop = 0
        dodge_weight = 1
        attack_weight = 1
        init_medicine_nums = 4
        # 用于防止连续帧重复计算reward
        last_time = time.time()

        while True:
            # reshape station for tf input placeholder
            station = np.array(station).reshape(-1, HEIGHT, WIDTH, 1)[0]

            print("loop took {} seconds".format(time.time() - last_time))
            last_time = time.time()

            # get the action by state
            target_step += 1

            action = agent.Choose_Action(station)
            # take station then the station change
            dodge_weight, init_medicine_nums = actions.take_action(
                action, self_blood, dodge_weight, init_medicine_nums
            )

            screen_gray = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_BGR2GRAY)
            # collect station gray graph

            next_station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
            next_station = np.array(next_station).reshape(-1, HEIGHT, WIDTH, 1)[0]

            next_boss_blood = BloodCommon.get_boss_blood_window().blood_count()

            next_self_blood = BloodCommon.get_self_blood_window().blood_count()
            next_self_energy = BloodCommon.get_self_energy_window().blood_count()

            (
                reward,
                done,
                stop,
                emergence_break,
                dodge_weight,
                attack_weight,
                init_medicine_nums,
            ) = judge.action_judge(
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
            )
            # get action reward
            if emergence_break == 100:
                # emergence break , save model and paused
                # 遇到紧急情况，保存数据，并且暂停
                print("emergence_break")
                agent.save_model()
                paused = True
            agent.Store_Data(station, action, reward, next_station, done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                # save loss graph
                # print('train')
                agent.Train_Network(big_BATCH_SIZE, num_step)
            if target_step % UPDATE_STEP == 0:
                agent.Update_Target_Network()
                # update target Q network
            station = next_station

            self_blood = next_self_blood
            self_energy = next_self_energy

            boss_blood = next_boss_blood

            before_pause = paused
            paused = actions.pause_game(paused, emergence_break)
            if before_pause == True and paused == False:
                emergence_break = 0

            if done == 1:
                break
        if episode % 10 == 0:
            agent.save_model()
            # save model
        print(
            "episode: ",
            episode,
            "Evaluation Average Reward:",
            total_reward / target_step,
        )
