# coding=utf-8
import numpy as np
from grabscreen import grab_screen
import cv2
import time
import actions
from dqn import DQN
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
UPDATE_STEP = 200

# used to save log graph
num_step = 0

# used to update target Q network
target_step = 0

if __name__ == "__main__":
    
    # Initialize the Context object
    ctx = Context()
    ctx = actions.pause_game(ctx)

    # Automatically calculate context_dim based on the length of get_features() output
    context_dim = len(ctx.get_features())

    # Initialize the agent with the calculated context_dim
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path, context_dim=context_dim)


    for episode in range(EPISODES):
        ctx.begin_time = time.time()
        state_image = cv2.resize(window.get_main_screen_window().gray, (WIDTH, HEIGHT))
        context_features = ctx.get_features()
        state_image_array = np.array(state_image).reshape(1, 1, HEIGHT, WIDTH)
        state = (state_image_array, context_features)

        # Prevent reward calculations on consecutive identical frames
        last_time = time.time()
        target_step = 0
        total_reward = 0
        ctx.done = 0

        while True:
            ctx.updateContext()

            log("action cost {} seconds".format(time.time() - last_time))
            last_time = time.time()

            action = agent.Choose_Action(state)
            ctx = actions.take_action(action, ctx)

            next_state_image = cv2.resize(
                window.get_main_screen_window().gray, (WIDTH, HEIGHT)
            )
            next_context_features = ctx.get_features()
            next_state_image_array = np.array(next_state_image).reshape(1, 1, HEIGHT, WIDTH)
            next_state = (next_state_image_array, next_context_features)

            ctx = judge.action_judge(ctx)

            # Check for emergency stop
            if ctx.emergence_break == 100:
                log("emergence_break")
                agent.Save_Model()
                ctx.paused = True

            agent.Store_Data(state, action, ctx.reward, next_state, ctx.done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                agent.Train_Network(big_BATCH_SIZE, num_step)

            target_step += 1
            if target_step % UPDATE_STEP == 0:
                agent.Update_Target_Network()

            state = next_state  # Update the state

            # Control pause
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
