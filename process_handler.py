# process_handler.py

import cv2
import yaml
import time
import importlib
import threading
import traceback
import numpy as np
from pynput import keyboard
from log import log
from actions import ActionExecutor
from judge import ActionJudge

def process(context, running_event):
    context.reopen_shared_memory()
    emergency_queue = context.get_emergency_event_queue()
    normal_queue = context.get_normal_event_queue()

    # Load ActionExecutor
    executor = ActionExecutor('./config/actions_config.yaml')
    judger = ActionJudge()

    # Load configuration
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_type = config['model']['type']
    model_config = config['model'].get(model_type, {})
    training_config = config['training']
    env_config = config['environment']
    model_file = config['model']['model_file']

    # Dynamically import the model class using importlib
    log("Dynamically import the model class!")
    model_module = f"models.{model_type.lower()}"  # e.g., "models.dqn"
    Agent = getattr(importlib.import_module(model_module), model_type)

    context_dim = context.get_features_len()
    state_dim = (env_config['height'], env_config['width'])
    action_dim = executor.get_action_size()

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        context_dim=context_dim,
        config=model_config,
        model_file=model_file
    )
    log("Agent created!")

    # Event to control training mode
    training_mode = threading.Event()
    training_mode.clear()  # Start in paused mode

    def on_press(key):
        try:
            if key.char == 'g':
                if training_mode.is_set():
                    log("Pausing training mode...")
                    log("Wating for press g to start training")
                    training_mode.clear()
                else:
                    log("Starting training mode...")
                    training_mode.set()
        except AttributeError:
            pass  # Ignore special keys

    # Start the key listener in a separate thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    log("Wating for press g to start training")

    def get_current_state():
        # Get frame and status
        frame, status = context.get_frame_and_status()        
        state_image = cv2.resize(frame[:, :, :3], state_dim[::-1])
        state_image_array = np.array(state_image).transpose(2, 0, 1)[np.newaxis, :, :, :]
        context_features = context.get_features(status)
        state = (state_image_array, context_features)     

        return state, status

    def clear_event_queues():
        """Clear both event queues without processing events."""
        while not emergency_queue.empty():
            emergency_queue.get_nowait()
        while not normal_queue.empty():
            normal_queue.get_nowait()

    episode = 0
    save_step = training_config['save_step']

    while running_event.is_set():
        try:
            if training_mode.is_set():
                log("Training mode is ON")
                total_episodes = training_config['episodes']
                while episode < total_episodes and training_mode.is_set():
                    log(f"episode {episode} start")
                    # reset judge
                    judger.reset()
                    # Get initial state
                    state, status = get_current_state()
                    target_step = 0
                    done = 0
                    start_time = time.time()

                    while not done and training_mode.is_set():
                        # Choose action
                        action = agent.choose_action(state)
                        action_name = executor.get_action_name(action)
                        log(f"Agent采取 {action_name} 动作.")
                        executor.take_action(action)


                        events = []
                        injured = False
                        interrupt_success = True
                        interrupt_action_done = False
                        action_timeout = False

                        action_max_wait_time = 30  
                        action_start_time = time.time()
                        action_duration = 0
                        clear_event_queues()

                        # wait for the action to complete or check for emergency events
                        while executor.is_running():
                            time.sleep(0.001)

                            # 检查超时
                            action_duration = time.time() - action_start_time
                            if action_duration > action_max_wait_time:
                                log(f"{action_name} 动作超时，强制中断.")
                                action_timeout = True
                                break

                            while not emergency_queue.empty():
                                e_event = emergency_queue.get_nowait()
                                events.append(e_event)
                                if e_event['event'] == 'q_found' and e_event['current_value'] == 0:
                                    q_found_time = time.time()
                                    # Wait for a few time and check if the state is consistent
                                    while time.time() - q_found_time < 0.3:  # 300 ms delay
                                        time.sleep(0.01)
                                        if not emergency_queue.empty():
                                            delayed_event = emergency_queue.get_nowait()
                                            if delayed_event['event'] == 'q_found' and delayed_event['current_value'] == 1:
                                                # q_found is back to 1, ignore the previous event
                                                break
                                    else:
                                        # After delay, still no q_found, mark as done
                                        done = 1
                                        if not interrupt_action_done:
                                            interrupt_success = executor.interrupt_action()
                                            interrupt_action_done = True
                                elif e_event['event'] == 'self_blood' and e_event['relative_change'] < -1.0:
                                    injured = True
                                    if not interrupt_action_done:
                                        interrupt_success = executor.interrupt_action()
                                        interrupt_action_done = True
                            while not normal_queue.empty():
                                n_event = normal_queue.get_nowait()
                                events.append(n_event)
                            

                        if injured:
                            log(f"受伤了 {action_name} 动作提前结束 {action_duration:.2f}s.")
                        else:
                            log(f"{action_name} 动作结束 {action_duration:.2f}s.")


                        if not interrupt_success:
                            log(f"动作打断失败 结束本局.")
                            done = 1

                        if action_timeout:
                            log(f"动作超时出错 结束本局.")
                            done = 1                  



                        # Get next state
                        next_state, next_status = get_current_state()

                        # Compute reward
                        reward = judger.judge(
                            action_name,
                            injured,
                            status,
                            next_status,
                            events,
                            time.time() - start_time,
                            done) 


                        # Store transition and train
                        agent.store_data(state, action, reward, next_state, done)
                        agent.train_network()

                        target_step += 1
                        if target_step % training_config['update_step'] == 0:
                            agent.update_target_network()

                        state = next_state  # Update the state
                        status = next_status.copy() # Update the status


                    #一局结束 执行 重开动作
                    restart_action_name = training_config['restart_action']
                    ra_begin_time = time.time()
                    log(f"一局结束,执行重开动作 {restart_action_name}.")
                    executor.take_action(restart_action_name)
                    while executor.is_running():
                        time.sleep(0.03)
                        if not training_mode.is_set():
                            log("暂停训练 退出重开动作.")
                            executor.interrupt_action()
                        clear_event_queues()
                        
                    ra_time_cost = time.time() - ra_begin_time
                    log(f"重开动作完成,耗时: {ra_time_cost:.2f} s.")

                    episode += 1
                    # Save the model every 'save_step' episodes
                    if episode % save_step == 0:
                        agent.save_model()
                        log(f"Model saved at episode {episode}")

                if episode >= total_episodes:
                    log("Training completed.")
                    training_mode.clear()
            else:
                # Paused mode, consume events and discard
                clear_event_queues()
                time.sleep(0.03)

        except KeyboardInterrupt:
            log("Process: Exiting...")
            running_event.clear()
            break
        except Exception as e:
            # 使用 traceback 获取详细错误信息
            error_message = traceback.format_exc()
            log(f"An error occurred: {e}\n{error_message}")
            running_event.clear()
            break

    listener.stop()
    cv2.destroyAllWindows()

