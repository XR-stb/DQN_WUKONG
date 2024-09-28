# process_handler.py
import cv2
import yaml
import importlib


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



    #ctx = actions.pause_game(ctx)


    while running_event.is_set():
        try:
            frame, status = context.get_frame_and_status()
            # Process emergency events
            while not emergency_queue.empty():
                e_event = emergency_queue.get_nowait()
                log(f"Emergency event: {e_event}")

            # Process normal events
            while not normal_queue.empty():
                n_event = normal_queue.get_nowait()
                if n_event.get('event') == 'skill_2':
                    log(f"Normal event: {n_event}")

            cv2.imshow("ROI Frame", frame)
            cv2.waitKey(30)

        except KeyboardInterrupt:
            log("Process: Exiting...")
            break
        except Exception as e:
            log(f"An error occurred in process: {e}")
            break
