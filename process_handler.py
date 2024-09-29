# process_handler.py
import cv2
import yaml
import importlib
import threading
import numpy as np
from pynput import keyboard
from log import log
from actions import ActionExecutor


def process(context, running_event):
    context.reopen_shared_memory()
    emergency_queue = context.get_emergency_event_queue()
    normal_queue = context.get_normal_event_queue()

    # Load ActionExecutor
    executor = ActionExecutor('./config/actions_config.yaml')

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

        #cv2.imshow("ROI Frame", state_image)

        return state

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
                    # Get initial state
                    state = get_current_state()
                    target_step = 0
                    done = 0

                    while not done and training_mode.is_set():
                        # Choose action
                        action = agent.choose_action(state)
                        executor.take_action(action)

                        events = []

                        # wait for the action to complete or check for emergency events
                        while executor.is_running():
                            while not emergency_queue.empty():
                                e_event = emergency_queue.get_nowait()
                                events.append(e_event)
                                if e_event['event'] == 'q_found' and e_event['current_value'] == 0:
                                    log(f"死亡了 {e_event}")
                                    done = 1
                                    executor.interrupt_action()
                                elif e_event['event'] == 'self_blood' and e_event['relative_change'] < 0:
                                    log(f"受伤了 {e_event}")
                                    executor.interrupt_action()
                            while not normal_queue.empty():
                                n_event = normal_queue.get_nowait()
                                events.append(n_event)
                            cv2.waitKey(10)


                        log(f"动作结束 {action}")


                        # Get next state
                        next_state = get_current_state()

                        # Compute reward
                        reward = compute_reward(events) 


                        # Store transition and train
                        agent.store_data(state, action, reward, next_state, done)
                        agent.train_network()

                        target_step += 1
                        if target_step % training_config['update_step'] == 0:
                            agent.update_target_network()

                        state = next_state  # Update the state


                    #一局结束 执行 重开动作
                    log("一局结束 执行 重开动作.")
                    executor.take_action(training_config['restart_action'])
                    while executor.is_running():
                        if not training_mode.is_set():
                            log("暂停训练 退出重开动作.")
                            executor.interrupt_action()
                        clear_event_queues()
                        cv2.waitKey(10)

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
                cv2.waitKey(30)

        except KeyboardInterrupt:
            log("Process: Exiting...")
            break
        except Exception as e:
            log(f"An error occurred in process: {e}")
            break

    listener.stop()
    cv2.destroyAllWindows()


def compute_reward(events):
    # Implement your reward computation logic here
    reward = 5  # Placeholder
    return reward
