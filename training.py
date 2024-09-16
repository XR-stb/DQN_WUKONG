# coding=utf-8
import numpy as np
from grabscreen import grab_screen
import cv2
import time
import actions
from dqn import DQN
from ddqn import DDQN
import window
import judge
from context import Context
from log import log


DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"
WIDTH = 96
HEIGHT = 88

action_size = 6

EPISODES = 3000
big_BATCH_SIZE = 16
# times that evaluate the network
UPDATE_STEP = 50

# used to save log graph
num_step = 0

# used to update target Q network
target_step = 0

if __name__ == "__main__":
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
    # DDQN加速收敛
    # agent = DDQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)

    # 初始化 Context 对象
    # 初始暂停游戏
    ctx = Context()
    ctx = actions.pause_game(ctx)

    for episode in range(EPISODES):
        ctx.begin_time = time.time()
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
            ctx = actions.take_action(action, ctx)

            next_station = cv2.resize(
                window.get_main_screen_window().gray, (WIDTH, HEIGHT)
            )
            next_station = np.array(next_station).reshape(-1, HEIGHT, WIDTH, 1)[0]

            ctx.updateNextContext()

            # 计算奖励并更新Context
            ctx = judge.action_judge(ctx)

            # 检查是否紧急停止
            if ctx.emergence_break == 100:
                log("emergence_break")
                agent.Save_Model()
                ctx.paused = True

            agent.Store_Data(station, action, ctx.reward, next_station, ctx.done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                agent.Train_Network(big_BATCH_SIZE, num_step)

            target_step += 1
            if target_step % UPDATE_STEP == 0:
                agent.Update_Target_Network()

            station = next_station

            ctx.updateContext()

            # 控制暂停
            before_pause = ctx.paused
            ctx = actions.pause_game(ctx, ctx.emergence_break)
            if before_pause and not ctx.paused:
                ctx.emergence_break = 0

            if ctx.done == 1:
                break

        if episode % 10 == 0:
            agent.Save_Model()

        log(
            f"episode: {episode} Evaluation Average Reward: { total_reward / target_step}"
        )
