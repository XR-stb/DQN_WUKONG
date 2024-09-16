import tensorflow as tf
import random
from collections import deque
import numpy as np
import os

# Hyperparameters
REPLAY_SIZE = 50000  # Experience replay buffer size
BATCH_SIZE = 64      # Minibatch size for training
GAMMA = 0.99         # Discount factor for future rewards
INITIAL_EPSILON = 1.0  # Initial exploration rate
FINAL_EPSILON = 0.01   # Final exploration rate
EPSILON_DECAY_STEPS = 10000  # Steps over which to decay epsilon
LEARNING_RATE = 0.001  # Learning rate for optimizer
TARGET_UPDATE_FREQ = 1000  # Frequency to update target network

class DDQN:
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file):
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        self.epsilon = INITIAL_EPSILON
        self.epsilon_decay = (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY_STEPS
        self.model_path = os.path.join(model_file, "save_model")
        self.log_file = log_file
        self.global_step = 0

        # Create Q-Network and Target Network
        self.create_Q_network()
        self.create_target_network()

        # Initialize epsilon
        self.epsilon = INITIAL_EPSILON

    def create_Q_network(self):
        """Create the Q-network for estimating Q-values."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (5, 5), strides=1, padding='same', activation='relu', input_shape=(self.state_h, self.state_w, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            tf.keras.layers.Conv2D(64, (5, 5), strides=1, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.action_dim)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')

    def create_target_network(self):
        """Create the target network, identical to the Q-network."""
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def Update_Target_Network(self):
        """Update the target network to have the same weights as the Q-network."""
        self.target_model.set_weights(self.model.get_weights())

    def Choose_Action(self, state):
        """Epsilon-greedy policy for action selection."""
        self.global_step += 1
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Exploration
        else:
            Q_value = self.model.predict(np.expand_dims(state, axis=0))[0]  # Exploitation
            return np.argmax(Q_value)

    def Store_Data(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))

    def Train_Network(self, big_BATCH_SIZE=0):
        """Train the Q-network with experiences sampled from the replay buffer."""
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # 如果 big_BATCH_SIZE 被传入，则使用它，否则使用默认的 BATCH_SIZE
        batch_size = big_BATCH_SIZE if big_BATCH_SIZE else BATCH_SIZE

        # Sample a minibatch from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)

        # Convert to numpy arrays for efficiency
        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)

        # Compute the Q-values for the next states using the online network
        Q_online_next = self.model.predict(next_state_batch)
        next_action_batch = np.argmax(Q_online_next, axis=1)

        # Compute Q-values from the target network
        Q_target_next = self.target_model.predict(next_state_batch)

        # Compute the target Q-values
        y_batch = self.model.predict(state_batch)  # 初始化为当前网络的预测

        for i in range(batch_size):
            if done_batch[i]:
                y_batch[i][np.argmax(action_batch[i])] = reward_batch[i]  # 如果结束，只用奖励
            else:
                # 更新 y_batch 中的目标动作值
                y_batch[i][np.argmax(action_batch[i])] = reward_batch[i] + GAMMA * Q_target_next[i][next_action_batch[i]]

        # Train the model
        self.model.fit(state_batch, y_batch, epochs=1, verbose=0)

        # Update epsilon
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= self.epsilon_decay

        # Update target network periodically
        if self.global_step % TARGET_UPDATE_FREQ == 0:
            self.update_target_network()


    def Save_Model(self):
        """Save the trained model to disk."""
        # 改进：使用 .h5 扩展名保存模型
        self.model.save(self.model_path + ".h5")
        print(f"Model saved to {self.model_path}.h5")

    def load_model(self):
        """Load the trained model from disk."""
        # 改进：使用 .h5 扩展名加载模型
        if os.path.exists(self.model_path + ".h5"):
            self.model = tf.keras.models.load_model(self.model_path + ".h5")
            print(f"Model loaded from {self.model_path}.h5")


    def close(self):
        """Cleanup resources."""
        tf.keras.backend.clear_session()

# Example usage:
# dqn_agent = DQN(observation_width=80, observation_height=80, action_space=4, model_file='./model', log_file='./logs')
