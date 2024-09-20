# training.py
# coding=utf-8
import numpy as np
import cv2
import time
import actions
import window
import judge
from context import Context
from log import log
import yaml
import importlib

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_type = config['model']['type']
model_config = config['model'].get(model_type, {})
training_config = config['training']
env_config = config['environment']
model_file = config['model']['model_file']

# Dynamically import the model class using importlib
model_module = f"models.{model_type.lower()}"  # e.g., "models.dqn"
Agent = getattr(importlib.import_module(model_module), model_type)

if __name__ == "__main__":
    log(f"Agent load as models.{model_type.lower()}.\n")
    # Initialize Context and Agent
    log("Initializing Context and Agent.\n")
    ctx = Context()
    context_dim = len(ctx.get_features())
    state_dim = (env_config['height'], env_config['width'])
    action_dim = env_config['action_size']

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        context_dim=context_dim,
        config=model_config,
        model_file=model_file
    )
    ctx = actions.pause_game(ctx)

    for episode in range(training_config['episodes']):
        ctx.begin_time = time.time()
        # Get initial state
        state_image = cv2.resize(window.get_main_screen_window().color[:, :, :3], state_dim[::-1])
        state_image_array = np.array(state_image).transpose(2, 0, 1)[np.newaxis, :, :, :]
        context_features = ctx.get_features()
        state = (state_image_array, context_features)

        last_time = time.time()
        target_step = 0
        total_reward = 0
        ctx.done = 0

        while True:
            ctx.updateContext()
            log("Action took {:.2f} seconds".format(time.time() - last_time))
            last_time = time.time()

            action = agent.choose_action(state)
            ctx = actions.take_action(action, ctx)

            next_state_image = cv2.resize(window.get_main_screen_window().color[:, :, :3], state_dim[::-1])
            next_state_image_array = np.array(next_state_image).transpose(2, 0, 1)[np.newaxis, :, :, :]
            next_context_features = ctx.get_features()
            next_state = (next_state_image_array, next_context_features)

            ctx = judge.action_judge(ctx)
            if ctx.emergence_break == 100:
                log("Emergency break activated.\n")
                agent.save_model()
                log("Model Saved.\n")
                ctx.paused = True

            total_reward += ctx.reward
            agent.store_data(state, action, ctx.reward, next_state, ctx.done)
            agent.train_network()

            target_step += 1
            if target_step % training_config['update_step'] == 0:
                agent.update_target_network()

            state = next_state  # Update the state

            # Control pause
            before_pause = ctx.paused
            ctx = actions.pause_game(ctx, ctx.emergence_break)
            if before_pause and not ctx.paused:
                ctx.emergence_break = 0

            if ctx.done == 1:
                break
