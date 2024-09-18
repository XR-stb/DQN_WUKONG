# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from log import log

# experiences replay buffer size
REPLAY_SIZE = 2000
# size of minibatch
small_BATCH_SIZE = 32
big_BATCH_SIZE = 128
BATCH_SIZE_door = 1000

# these are the hyper Parameters for DQN
# discount factor for target Q to calculate the TD aim value
GAMMA = 0.9
# the start value of epsilon E
INITIAL_EPSILON = 0.5
# the final value of epsilon
FINAL_EPSILON = 0.009

class QNetwork(nn.Module):
    def __init__(self, state_h, state_w, action_dim):
        super(QNetwork, self).__init__()
        self.state_h = state_h
        self.state_w = state_w
        self.action_dim = action_dim
        # Define the layers as in the original code
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        # Calculate the size of the feature maps after convolution and pooling
        def conv2d_size_out(size, kernel_size=5, stride=1, padding=2):
            return (size + 2*padding - (kernel_size - 1) -1)//stride + 1
        convw = conv2d_size_out(conv2d_size_out(state_w)) // 4
        convh = conv2d_size_out(conv2d_size_out(state_h)) // 4
        linear_input_size = convw * convh * 64
        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.head = nn.Linear(256, action_dim)
    
    def forward(self, x):
        # x shape: [batch_size, 1, state_h, state_w]
        x = F.relu(self.conv1(x))  # Conv layer 1
        x = self.pool(x)           # Max pooling 1
        x = F.relu(self.conv2(x))  # Conv layer 2
        x = self.pool(x)           # Max pooling 2
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))    # Fully connected layer 1
        x = F.dropout(x, p=0, training=self.training)  # Dropout p=0 equivalent to no dropout
        x = F.relu(self.fc2(x))    # Fully connected layer 2
        x = F.dropout(x, p=0, training=self.training)
        x = self.head(x)           # Output layer
        return x

class DQN():
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file):
        # the state is the input vector of network, in this env, it has four dimensions
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        # the action is the output vector and it has two dimensions
        self.action_dim = action_space
        # init experience replay, the deque is a list that first-in & first-out
        self.replay_buffer = deque()
        # you can create the network by the two parameters
        self.policy_net = QNetwork(self.state_h, self.state_w, self.action_dim)
        self.target_net = QNetwork(self.state_h, self.state_w, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # define optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        # set the value in choose_action
        self.epsilon = INITIAL_EPSILON
        self.model_path = model_file + "/save_model.pth"
        self.model_file = model_file
        self.log_file = log_file
        # 初始化模型
        if os.path.exists(self.model_path):
            log("model exists , load model\n")
            self.policy_net.load_state_dict(torch.load(self.model_path))
        else:
            log("model don't exists , create new one\n")
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
    
    # 选择动作
    def Choose_Action(self, state):
        state = torch.from_numpy(state).float().to(self.device)  # state shape: [1, 1, HEIGHT, WIDTH]
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            with torch.no_grad():
                Q_value = self.policy_net(state)
                return Q_value.max(1)[1].item()
    
    # 存储数据
    def Store_Data(self, state, action, reward, next_state, done):
        # 如果state的形状是 [1, 1, HEIGHT, WIDTH]，需要去除第一个维度
        state = np.squeeze(state, axis=0)  # shape: [1, HEIGHT, WIDTH]
        next_state = np.squeeze(next_state, axis=0)  # shape: [1, HEIGHT, WIDTH]
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

    # 训练网络
    def Train_Network(self, BATCH_SIZE, num_step):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = np.array([data[0] for data in minibatch])  # Shape: [BATCH_SIZE, 1, HEIGHT, WIDTH]
        action_batch = np.array([data[1] for data in minibatch])
        reward_batch = np.array([data[2] for data in minibatch])
        next_state_batch = np.array([data[3] for data in minibatch])  # Shape: [BATCH_SIZE, 1, HEIGHT, WIDTH]
        done_batch = np.array([data[4] for data in minibatch])

        # 在转换为张量之前，添加一个新的维度
        state_batch = np.expand_dims(state_batch, axis=1)  # Shape: [BATCH_SIZE, 1, HEIGHT, WIDTH]
        next_state_batch = np.expand_dims(next_state_batch, axis=1)  # Shape: [BATCH_SIZE, 1, HEIGHT, WIDTH]

        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).long().to(self.device).unsqueeze(1)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        done_batch = torch.from_numpy(done_batch).float().to(self.device)

        # 计算当前Q值
        Q_values = self.policy_net(state_batch).gather(1, action_batch)
        # 计算下一个状态的最大Q值
        with torch.no_grad():
            next_Q_values = self.target_net(next_state_batch).max(1)[0]
            target_Q_values = reward_batch + GAMMA * next_Q_values * (1 - done_batch)

        # 计算损失
        loss = self.criterion(Q_values.squeeze(), target_Q_values)

        # 更新网络参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 更新目标网络
    def Update_Target_Network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # 测试时使用
    def action(self, state):
        state = torch.from_numpy(state).float().to(self.device)  # state shape: [1, 1, HEIGHT, WIDTH]
        with torch.no_grad():
            Q_value = self.policy_net(state)
            return Q_value.max(1)[1].item()

    # 保存模型
    def Save_Model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        log(f"Save to path: {self.model_path}")
