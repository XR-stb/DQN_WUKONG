# models/ddqn_resnet.py
# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from models.base_agent import BaseAgent
from torchvision import models
from torchvision.models import ResNet18_Weights

class QNetwork(nn.Module):
    def __init__(self, action_dim, context_dim):
        super(QNetwork, self).__init__()
        # 加载预训练的 ResNet18 模型
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # 冻结所有参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 解冻最后的两层（layer4 和 fc）
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        # 修改 ResNet18 的最后一层，使其输出特征向量而不是分类结果
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        # 将 ResNet 特征与上下文特征拼接
        self.fc1 = nn.Linear(num_ftrs + context_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.head = nn.Linear(256, action_dim)

    def forward(self, x, context_features):
        x = self.resnet(x)
        x = torch.cat((x, context_features), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.head(x)
        return x

class DDQN_RESNET(BaseAgent):
    def __init__(self, state_dim, action_dim, context_dim, config, model_file):
        super(DDQN_RESNET, self).__init__(config, model_file)
        self.action_dim = action_dim
        self.context_dim = context_dim
        # 创建网络
        self.policy_net = QNetwork(self.action_dim, context_dim)
        self.target_net = QNetwork(self.action_dim, context_dim)
        self.update_target_network()
        # 定义优化器和损失函数
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.policy_net.parameters()), lr=self.lr)
        self.criterion = nn.MSELoss()
        # 加载模型（如果存在）
        self.load_model()
        # 设置设备
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def choose_action(self, state):
        state_image, context_features = state
        state_image = torch.from_numpy(state_image).float().to(self.device)
        context_features = torch.from_numpy(np.array(context_features)).float().to(self.device).unsqueeze(0)
        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_value = self.policy_net(state_image, context_features)
                action = q_value.max(1)[1].item()
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        return action, None

    def store_data(self, state, action, reward, next_state, done, log_prob=None):
        state_image, context_features = state
        next_state_image, next_context_features = next_state
        state_image = np.squeeze(state_image, axis=0)
        next_state_image = np.squeeze(next_state_image, axis=0)
        self.replay_buffer.append((state_image, context_features, action, reward, next_state_image, next_context_features, done))

    def train_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_images = np.array([data[0] for data in minibatch])
        state_contexts = np.array([data[1] for data in minibatch])
        action_batch = np.array([data[2] for data in minibatch])
        reward_batch = np.array([data[3] for data in minibatch])
        next_state_images = np.array([data[4] for data in minibatch])
        next_state_contexts = np.array([data[5] for data in minibatch])
        done_batch = np.array([data[6] for data in minibatch])

        state_images = torch.from_numpy(state_images).float().to(self.device)
        state_contexts = torch.from_numpy(state_contexts).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).long().to(self.device).unsqueeze(1)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_images = torch.from_numpy(next_state_images).float().to(self.device)
        next_state_contexts = torch.from_numpy(next_state_contexts).float().to(self.device)
        done_batch = torch.from_numpy(done_batch).float().to(self.device)

        # 计算当前 Q 值
        q_values = self.policy_net(state_images, state_contexts).gather(1, action_batch)
        # 使用 DDQN 更新计算下一步 Q 值
        with torch.no_grad():
            next_state_actions = self.policy_net(next_state_images, next_state_contexts).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_images, next_state_contexts).gather(1, next_state_actions).squeeze()
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # 计算损失
        loss = self.criterion(q_values.squeeze(), target_q_values)

        # 更新网络参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
