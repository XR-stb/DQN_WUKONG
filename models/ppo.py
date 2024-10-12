# models/ppo.py
# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.base_agent import BaseAgent


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_h, state_w, action_dim, context_dim):
        super(ActorCriticNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

        # Calculate the size of the feature maps after convolution and pooling
        def conv2d_size_out(size, kernel_size=5, stride=1, padding=2):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(state_w)) // 4
        convh = conv2d_size_out(conv2d_size_out(state_h)) // 4
        linear_input_size = convw * convh * 64 + context_dim
        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.actor_head = nn.Linear(256, action_dim)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x, context_features):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, context_features), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_logits = self.actor_head(x)
        state_value = self.critic_head(x)
        return action_logits, state_value


class PPO(BaseAgent):
    def __init__(self, state_dim, action_dim, context_dim, config, model_file):
        super(PPO, self).__init__(config, model_file)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.state_h, self.state_w = state_dim
        self.policy_net = ActorCriticNetwork(
            self.state_h, self.state_w, self.action_dim, context_dim
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.eps_clip = config.get("eps_clip", 0.2)
        self.update_epochs = config.get("update_epochs", 10)  # 从配置中获取
        self.gae_lambda = config.get("gae_lambda", 0.95)  # 从配置中获取
        self.load_model()
        self.policy_net.to(self.device)

    def choose_action(self, state):
        state_image, context_features = state
        state_image = torch.from_numpy(state_image).float().to(self.device)
        context_features = (
            torch.from_numpy(np.array(context_features))
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        with torch.no_grad():
            action_logits, _ = self.policy_net(state_image, context_features)
            action_prob = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_prob, 1).item()
            log_prob = torch.log(action_prob[0, action])  # 记录选定动作的log_prob
        return action, log_prob

    def compute_advantage(self, rewards, values, masks):
        returns = []
        gae = 0
        values = values + [0]  # 为了匹配步数，增加一个终止状态的值
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * masks[step]
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update_network(self, memory):
        for _ in range(self.update_epochs):  # 使用update_epochs参数
            states, actions, rewards, next_states, masks, old_log_probs = memory
            states_images, states_contexts = states
            next_states_images, next_states_contexts = next_states

            states_images = torch.from_numpy(states_images).float().to(self.device)
            states_contexts = torch.from_numpy(states_contexts).float().to(self.device)
            next_states_images = (
                torch.from_numpy(next_states_images).float().to(self.device)
            )
            next_states_contexts = (
                torch.from_numpy(next_states_contexts).float().to(self.device)
            )
            actions = torch.from_numpy(actions).long().to(self.device)
            old_log_probs = torch.from_numpy(old_log_probs).float().to(self.device)

            action_logits, values = self.policy_net(states_images, states_contexts)
            _, next_values = self.policy_net(next_states_images, next_states_contexts)
            new_log_probs = (
                torch.log_softmax(action_logits, dim=-1)
                .gather(1, actions.unsqueeze(1))
                .squeeze()
            )

            returns = self.compute_advantage(rewards, values.tolist(), masks)
            returns = torch.tensor(returns).float().to(self.device)

            advantages = returns - values
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)

            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def store_data(self, state, action, reward, next_state, done, log_prob):
        state_image, context_features = state
        next_state_image, next_context_features = next_state
        self.replay_buffer.append(
            (
                state_image,
                context_features,
                action,
                reward,
                next_state_image,
                next_context_features,
                done,
                log_prob,
            )
        )

    def train_network(self):
        # Add your PPO training logic here
        pass

    def update_target_network(self):
        # PPO may not need this function like DQN does, but you can define it as a placeholder
        pass
