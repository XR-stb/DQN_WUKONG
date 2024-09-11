# coding=utf-8
import numpy as np
from grabscreen import grab_screen
import cv2
import time
import actions
from DQN_tensorflow_gpu import DQN
import window
import judge
from context import Context
from log import log

DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"
WIDTH = 96
HEIGHT = 88

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

if __name__ == "__main__":
    # DQN init
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)

    # 初始化 Context 对象
    # 初始暂停游戏
    context = Context()
    context.paused = actions.pause_game(context)

    for episode in range(EPISODES):
        station = cv2.resize(window.get_main_screen_window().gray, (WIDTH, HEIGHT))

        # 用于防止连续帧重复计算reward
        last_time = time.time()
        target_step = 0
        total_reward = 0

        while True:
            # reshape station for tf input placeholder
            station = np.array(station).reshape(-1, HEIGHT, WIDTH, 1)[0]

            log("action cost {} seconds".format(time.time() - last_time))
            last_time = time.time()

            action = agent.Choose_Action(station)
            context = actions.take_action(action, context)

            next_station = cv2.resize(window.get_main_screen_window().gray, (WIDTH, HEIGHT))
            next_station = np.array(next_station).reshape(-1, HEIGHT, WIDTH, 1)[0]

            context.updateNextContext()
            
            # 计算奖励并更新Context
            context = judge.action_judge(context)

            # 检查是否紧急停止
            if context.emergence_break == 100:
                log("emergence_break")
                agent.save_model()
                context.paused = True

            agent.Store_Data(
                station, action, context.reward, next_station, context.done
            )
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                agent.Train_Network(big_BATCH_SIZE, num_step)

            target_step += 1
            if target_step % UPDATE_STEP == 0:
                agent.Update_Target_Network()

            station = next_station

            context.updateContext()

            # 控制暂停
            before_pause = context.paused
            context = actions.pause_game(context, context.emergence_break)
            if before_pause and not context.paused:
                context.emergence_break = 0

            if context.done == 1:
                break

        if episode % 10 == 0:
            agent.save_model()

        log(
            "episode: ",
            episode,
            "Evaluation Average Reward:",
            total_reward / target_step,
        )
