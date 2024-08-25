# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:10:06 2021

@author: pang
"""

import numpy as np
from grabscreen import grab_screen
import cv2
import time
import directkeys
from getkeys import key_check
import random
from DQN_tensorflow_gpu import DQN
import os
import pandas as pd
from restart import restart
import random
import tensorflow.compat.v1 as tf
import BloodCommon

def pause_game(paused):
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'T' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused

def take_action(action):
    if action == 0:     # n_choose
        pass
    elif action == 1:   # j
        directkeys.attack()
    elif action == 2:   # k
        # directkeys.jump()
        # directkeys.dodge()
        directkeys.attack()
    elif action == 3:   # m
        # directkeys.defense()
        # directkeys.dodge()
        pass
    elif action == 4:   # r
        directkeys.dodge()


def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, stop, emergence_break):
    reward, done = 0, 0
    # get action reward
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    if next_self_blood < 5:     # self dead
        if emergence_break < 2:
            reward = -300
            done = 1
            stop = 0
            emergence_break += 1
        else:
            reward = -10
            done = 1
            stop = 0
            print(emergence_break, 'sssssssssssss')
            emergence_break = 0
    elif next_boss_blood < 5:   #boss dead
        if emergence_break < 2:
            reward = 200
            done = 0
            stop = 0
            emergence_break += 1
        else:
            reward = 20
            done = 0
            stop = 0
            print(emergence_break, 'ddddddddddd')
            emergence_break = 0
    else:
        self_blood_reward = 0
        boss_blood_reward = 0
        # print(next_self_blood - self_blood)
        # print(next_boss_blood - boss_blood)
        if next_self_blood - self_blood < -5:
            if stop == 0:
                self_blood_reward = -6 * (next_self_blood - self_blood)
                stop = 1
                # 防止连续取帧时一直计算掉血
        else:
            stop = 0
        if next_boss_blood - boss_blood <= -5:
            boss_blood_reward = 5 * (next_boss_blood - boss_blood)

        reward = self_blood_reward + boss_blood_reward
        done = 0
        emergence_break = 0
    return reward, done, stop, emergence_break
        

DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"
WIDTH = 96
HEIGHT = 88

# station window_size
# window_size = (320,100,704,452)#384,352  192,176 96,88 48,44 24,22
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

if __name__ == '__main__':
    # DQN init
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
    
    # paused at the begin
    paused = pause_game(paused)
    
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    emergence_break = 0     
    
    for episode in range(EPISODES):
        screen_gray = cv2.cvtColor(grab_screen(window_size),cv2.COLOR_BGR2GRAY)

        # change graph to WIDTH * HEIGHT for station input
        station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
        
        # count init blood
        boss_blood = BloodCommon.get_self_blood_window().blood_count()
        self_blood = BloodCommon.get_boss_blood_window().blood_count()
        
        # used to update target Q network
        target_step = 0
        
        done = 0
        total_reward = 0
        stop = 0    
        # 用于防止连续帧重复计算reward
        last_time = time.time()
        while True:
            # reshape station for tf input placeholder
            station = np.array(station).reshape(-1,HEIGHT,WIDTH,1)[0]
            
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

            # get the action by state
            target_step += 1
            
            action = agent.Choose_Action(station)
            # take station then the station change
            take_action(action)
            

            screen_gray = cv2.cvtColor(grab_screen(window_size),cv2.COLOR_BGR2GRAY)
            # collect station gray graph

            next_station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
            next_station = np.array(next_station).reshape(-1,HEIGHT,WIDTH,1)[0]

            next_boss_blood = BloodCommon.get_self_blood_window().blood_count()
            next_self_blood = BloodCommon.get_boss_blood_window().blood_count()

            reward, done, stop, emergence_break = action_judge(boss_blood, next_boss_blood,
                                                               self_blood, next_self_blood,
                                                               stop, emergence_break)
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
            boss_blood = next_boss_blood
            total_reward += reward
            paused = pause_game(paused)
            if done == 1:
                break
        if episode % 10 == 0:
            agent.save_model()
            # save model
        print('episode: ', episode, 'Evaluation Average Reward:', total_reward/target_step)
        # restart()
        
            
            
            
            
            
        
        
    
    