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
                    executor.interrupt_action()
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
                    clear_event_queues()
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
                        can_interrupt = executor.is_interruptible(action_name)
                        interrupt_action_done = False


                        action_start_time = time.time()
                        action_duration = 0


                        # wait for the action to complete or check for emergency events
                        while executor.is_running():
                            time.sleep(0.001)

                            action_duration = time.time() - action_start_time

                            while not emergency_queue.empty():
                                e_event = emergency_queue.get_nowait()
                                if e_event['timestamp'] > start_time:
                                    events.append(e_event)
                                if (e_event['event'] == 'q_found' and
                                    e_event['current_value'] == 0 and
                                    e_event['timestamp'] > start_time):
                                    q_found_time = time.time()
                                    # Wait for a few time and check if the state is consistent
                                    while time.time() - q_found_time < 0.5:  # 500 ms delay
                                        time.sleep(0.01)
                                        if not emergency_queue.empty():
                                            delayed_event = emergency_queue.get_nowait()
                                            if delayed_event['event'] == 'q_found' and delayed_event['current_value'] == 1:
                                                # q_found is back to 1, ignore the previous event
                                                log(f"q_found is back to 1, ignore the previous event")
                                                break
                                    else:
                                        # After delay, still no q_found, mark as done
                                        done = 1
                                        log(f"no q_found, mark as done")
                                        if not interrupt_action_done:
                                            executor.interrupt_action()
                                            interrupt_action_done = True
                                elif (e_event['event'] == 'self_blood' and
                                      e_event['relative_change'] < -1.0 and
                                      e_event['timestamp'] > action_start_time):
                                    injured = True
                                    if not interrupt_action_done and can_interrupt:
                                        executor.interrupt_action()
                                        interrupt_action_done = True
                            while not normal_queue.empty():
                                n_event = normal_queue.get_nowait()
                                if n_event['timestamp'] > start_time:
                                    events.append(n_event)
                            

                        if injured and can_interrupt:
                            log(f"受伤了 {action_name} 动作提前结束 {action_duration:.2f}s.")
                        elif injured and not can_interrupt:
                            log(f"{action_name} 动作不可中断 耗时 {action_duration:.2f}s.")
                        else:
                            log(f"{action_name} 动作结束 {action_duration:.2f}s.")



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


                    if training_mode.is_set():
                        episode += 1
                        # Save the model every 'save_step' episodes
                        if episode % save_step == 0:
                            agent.save_model()
                            log(f"Model saved at episode {episode}")

                        log(f"current episode finished,epsilon: {agent.epsilon}")

                    #SKIP CG if have
                    if training_mode.is_set():
                        executor.take_action('SKIP_CG')
                        executor.wait_for_finish()
                        log(f"Skip CG done!")


                    # 检查初始化状态 准备重开
                    if training_mode.is_set():
                        log("Checking initialization state, preparing for restart...")
                        stable_start_time = None
                        required_stable_duration = 1.0  # 稳定时间（秒）
                        while running_event.is_set():
                            _, status = get_current_state()
                            # 判断 'self_blood' 是否连续大于 95%
                            if status['self_blood'] > 95.0:
                                if stable_start_time is None:
                                    stable_start_time = time.time()
                                elif time.time() - stable_start_time >= required_stable_duration:
                                    break
                            else:
                                stable_start_time = None  # 不符合条件则重置
                            time.sleep(0.05)
                        log("Ready to do restart action!")

                    #执行 重开动作
                    if training_mode.is_set():
                        restart_action_name = training_config['restart_action']
                        executor.take_action(restart_action_name)
                        executor.wait_for_finish()
                        log(f"重开动作 {restart_action_name} 完成.")                



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

