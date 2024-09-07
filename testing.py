# coding=utf-8

import numpy as np
from grabscreen import grab_screen
import cv2
import time
import actions
from getkeys import key_check
from DQN_tensorflow_gpu import DQN
from restart import restart
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
action_size = 4

# used to stop training
paused = True

if __name__ == '__main__':
    # DQN init
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
    
    paused = actions.pause_game(paused)
    # paused at the begin
    screen_gray = cv2.cvtColor(grab_screen(window_size),cv2.COLOR_BGR2GRAY)

    # change graph to WIDTH * HEIGHT for station input
    station = cv2.resize(screen_gray,(WIDTH,HEIGHT))

    boss_blood = BloodCommon.get_boss_blood_window().blood_count()

    self_blood = BloodCommon.get_self_blood_window().blood_count()
    self_energy = BloodCommon.get_self_energy_window().blood_count()

    # used to update target Q network
    target_step = 0
    
    done = 0
    total_reward = 0
    stop = 0    
    # 用于防止连续帧重复计算reward
    last_time = time.time()
    emergence_break = 0  
    while True:
        # reshape station for tf input placeholder
        station = np.array(station).reshape(-1,HEIGHT,WIDTH,1)[0]
        
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        # get the action by state
        action = agent.Choose_Action(station)
        actions.take_action(action, self_blood)

        screen_gray = cv2.cvtColor(grab_screen(window_size),cv2.COLOR_BGR2GRAY)

        next_station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
        next_station = np.array(next_station).reshape(-1,HEIGHT,WIDTH,1)[0]
        station = next_station
        
        next_boss_blood = BloodCommon.get_boss_blood_window().blood_count()

        next_self_blood = BloodCommon.get_self_blood_window().blood_count()
        next_self_energy = BloodCommon.get_self_energy_window().blood_count()

        reward, done, stop, emergence_break = judge.action_judge(boss_blood, next_boss_blood,
                                                               self_blood, next_self_blood,
                                                               self_energy, next_self_energy,
                                                               stop, emergence_break)
        # get action reward
        if emergence_break == 100:
            # emergence break , save model and paused
            print("emergence_break")
            agent.save_model()
            paused = True
        keys = key_check()

        before_pause = paused
        paused = actions.pause_game(paused, emergence_break)
        if before_pause == True and paused == False:
                emergence_break = 0

        if 'G' in keys:
            print('stop testing DQN')
            break
        
        
            
            
            
            
            
        
        
    
    