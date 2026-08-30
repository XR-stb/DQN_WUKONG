import numpy as np
import torch
import copy
import math
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn
from models.base_agent import BaseAgent
import os
from collections import deque


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # 假设输入是RGB图像
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return self.flatten(x)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(ActorCritic, self).__init__()

        # CNN特征提取器
        self.cnn = CNNFeatureExtractor()

        # 计算CNN输出维度（需要根据实际输入尺寸调整）
        cnn_out_dim = self._get_cnn_out_dim()

        # 合并CNN特征和状态特征的网络
        self.feature_net = nn.Sequential(
            nn.Linear(cnn_out_dim + state_dim, net_width),
            nn.ReLU(),
            nn.BatchNorm1d(net_width),
            nn.Dropout(0.1),
        )

        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.BatchNorm1d(net_width),
            nn.Dropout(0.1),
            nn.Linear(net_width, action_dim),
        )

        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.BatchNorm1d(net_width),
            nn.Dropout(0.1),
            nn.Linear(net_width, 1),
        )

    def _get_cnn_out_dim(self):
        # 使用一个示例输入计算CNN输出维度
        x = torch.zeros(1, 3, 84, 84)  # 假设输入是84x84的RGB图像
        return self.cnn(x).shape[1]

    def forward(self, state_img, state_vec):
        # 提取图像特征
        img_features = self.cnn(state_img)
        # 合并特征
        combined_features = torch.cat([img_features, state_vec], dim=1)
        # 共享特征提取
        features = self.feature_net(combined_features)
        # Actor和Critic输出
        action_probs = torch.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, state):
        """
        Args:
            state: 形状为 (batch_size, state_dim) 的张量
        """
        # 确保输入是二维的
        if state.dim() == 1:
            state = state.unsqueeze(0)

        n = torch.tanh(self.l1(state))
        n = self.dropout(n)
        n = torch.tanh(self.l2(n))
        n = self.l3(n)  # 不要在这里使用激活函数
        return n

    def pi(self, state, softmax_dim=-1):
        """计算动作概率分布"""
        logits = self.forward(state)
        # 添加温度系数使概率分布更加"锐利"
        temperature = 1.0  # 可以调整这个值，越小越倾向于选择最优动作
        prob = F.softmax(logits / temperature, dim=softmax_dim)
        return prob

    def get_log_prob(self, state, action):
        # 添加获取动作对数概率的方法
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)


class Critic(nn.Module):
    def __init__(self, state_dim, net_width):
        super(Critic, self).__init__()
        self.c1 = nn.Linear(state_dim, net_width)
        self.c2 = nn.Linear(net_width, net_width)
        self.c3 = nn.Linear(net_width, 1)
        self.dropout = nn.Dropout(0.1)

        # 移除BatchNorm，因为我们处理单个样本

    def forward(self, state):
        # 确保输入形状正确
        if state.dim() == 1:
            state = state.unsqueeze(0)  # 添加batch维度

        v = torch.relu(self.c1(state))
        v = self.dropout(v)
        v = torch.relu(self.c2(v))
        v = self.c3(v)
        return v


import logging

logger = logging.getLogger(__name__)


class PPO(BaseAgent):
    def __init__(self, state_dim, action_dim, context_dim, config, model_file):
        super(PPO, self).__init__(config, model_file)
        # 初始化 PPO 的超参数
        self.gamma = config.get("gamma", 0.9)
        self.lambd = config.get("lambd", 0.95)  # GAE 参数
        self.l2_reg = config.get("l2_reg", 0.001)  # L2 正则化系数
        self.initial_epsilon = config.get("initial_epsilon", 0.5)
        self.final_epsilon = config.get("final_epsilon", 0.009)
        self.epsilon_decay = config.get("epsilon_decay", 0.0002)
        self.lr = config.get("lr", 0.0003)
        self.critic_lr = config.get("critic_lr", 0.001)  # Critic单独学习率，需要更快收敛
        self.batch_size = config.get("batch_size", 64)
        self.epsilon_clip = config.get("epsilon_clip", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.05)
        self.entropy_coef_decay = config.get("entropy_coef_decay", 0.995)
        self.min_entropy = config.get("min_entropy", 0.005)  # 最小熵系数，防止探索过度衰减
        self.T_horizon = config.get("T_horizon", 512)
        self.net_width = config.get("net_width", 256)
        self.K_epochs = config.get("K_epochs", 8)
        self.adv_normalization = config.get("adv_normalization", True)
        self.clip_rate = config.get("clip_rate", 0.2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 计算输入特征的总维度
        if isinstance(state_dim, tuple):  # 如果 state_dim 是元组（如图像的高度和宽度）
            height, width = state_dim
            image_dim = height * width
            self.state_dim = (
                image_dim + context_dim
            )  # 图像展平后的维度 + 上下文特征的维度
        else:
            self.state_dim = state_dim  # 直接使用 state_dim

        # 构建 Actor 和 Critic
        self.actor = Actor(self.state_dim, action_dim, self.net_width).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = Critic(self.state_dim, self.net_width).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # 构建轨迹存储器
        self.s_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.a_hoder = np.zeros((self.T_horizon, 1), dtype=np.int64)
        self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.logprob_a_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.dw_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)

        # 添加优先级经验回放
        self.priorities = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.max_priority = 1.0  # 初始优先级

        # 添加观察标准化
        self.obs_mean = np.zeros(self.state_dim, dtype=np.float32)
        self.obs_std = np.ones(self.state_dim, dtype=np.float32)
        self.obs_count = 1e-4

        # 添加梯度累积
        self.gradient_steps = 0
        self.gradient_accumulation = 4

        # 学习率调度器：使用余弦退火，比StepLR更平滑
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=5000, eta_min=self.lr * 0.1
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=5000, eta_min=self.critic_lr * 0.1
        )

        # 添加经验回放缓冲区
        self.replay_buffer_size = config.get("replay_buffer_size", 10000)
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

        # GAE参数
        self.gae_lambda = config.get("gae_lambda", 0.95)

        # 探索相关参数
        self.exploration_rate = config.get("exploration_rate", 1.0)
        self.exploration_decay = config.get("exploration_decay", 0.995)
        self.exploration_min = config.get("exploration_min", 0.01)

        self.action_dim = action_dim

        # 当前轨迹步数计数器
        self.trajectory_step = 0

    def _prepare_state(self, s):
        """统一处理状态数据，返回 (batch_size=1, state_dim) 的张量"""
        if isinstance(s, (list, tuple)):
            state_image, context_features = s
            state_image = np.array(state_image, dtype=np.float32).ravel()
            context_features = np.array(context_features, dtype=np.float32).ravel()
            s = np.concatenate([state_image, context_features])
        else:
            s = np.array(s, dtype=np.float32).ravel()

        # 维度对齐
        if s.shape[0] > self.state_dim:
            s = s[: self.state_dim]
        elif s.shape[0] < self.state_dim:
            s = np.pad(s, (0, self.state_dim - s.shape[0]))

        state_tensor = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        return state_tensor

    def choose_action(self, s):
        """选择动作（标准PPO：从策略分布中采样）"""
        try:
            state_tensor = self._prepare_state(s)

            with torch.no_grad():
                pi = self.actor.pi(state_tensor)
                dist = Categorical(pi)

                # 训练阶段：从分布中采样（PPO的标准做法，策略自带探索性）
                a = dist.sample()
                log_prob = dist.log_prob(a)

                return a.item(), log_prob.item()

        except Exception as e:
            logger.error(f"选择动作时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0, 0.0

    def train_network(self):
        """当轨迹缓冲区满时执行训练，否则跳过"""
        if self.trajectory_step < self.T_horizon:
            return  # 数据不够，跳过训练

        # 熵系数衰减，但不低于最小值
        self.entropy_coef = max(self.entropy_coef * self.entropy_coef_decay, self.min_entropy)

        # 只取已填充的数据
        n = self.trajectory_step
        s = torch.from_numpy(self.s_hoder[:n]).to(self.device)
        a = torch.from_numpy(self.a_hoder[:n]).to(self.device)
        r = torch.from_numpy(self.r_hoder[:n]).to(self.device)
        s_next = torch.from_numpy(self.s_next_hoder[:n]).to(self.device)
        old_prob_a = torch.from_numpy(self.logprob_a_hoder[:n]).to(self.device)
        done = torch.from_numpy(self.done_hoder[:n]).to(self.device)
        dw = torch.from_numpy(self.dw_hoder[:n]).to(self.device)

        # 计算优势和回报
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            done_float = done.float()
            dw_float = dw.float()

            # 使用GAE计算优势
            deltas = r + self.gamma * vs_ * (1 - dw_float) - vs
            advantages = torch.zeros_like(deltas)
            advantage = 0
            for t in reversed(range(len(deltas))):
                advantage = deltas[t] + self.gamma * self.lambd * advantage * (
                    1 - done_float[t]
                )
                advantages[t] = advantage

            returns = advantages + vs

            if self.adv_normalization:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

        # 多轮次小批量更新
        for _ in range(self.K_epochs):
            # 随机打乱索引以进行小批量训练
            indices = np.arange(n)
            np.random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]

                s_batch = s[batch_idx]
                a_batch = a[batch_idx]
                old_prob_batch = old_prob_a[batch_idx]
                adv_batch = advantages[batch_idx]
                ret_batch = returns[batch_idx]

                # 计算新的动作概率
                new_probs = self.actor.pi(s_batch)
                new_log_probs = torch.log(new_probs.gather(1, a_batch) + 1e-8)

                # 计算比率
                ratio = torch.exp(new_log_probs - old_prob_batch)

                # 裁剪目标
                surr1 = ratio * adv_batch
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
                    * adv_batch
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                v_pred = self.critic(s_batch)
                value_loss = F.mse_loss(v_pred, ret_batch)

                # 熵正则化
                entropy = -torch.sum(
                    new_probs * torch.log(new_probs + 1e-10), dim=-1
                ).mean()

                # 总损失
                loss = actor_loss + 0.5 * value_loss - self.entropy_coef * entropy

                # 更新网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # 更新学习率
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # 重置轨迹计数器
        self.trajectory_step = 0

    def store_data(self, s, a, r, s_next, done, logprob_a, this_epo_step):
        """存储一步数据到轨迹缓冲区"""
        idx = self.trajectory_step % self.T_horizon

        try:
            # 处理状态数据
            s_flat = self._flatten_state(s)
            s_next_flat = self._flatten_state(s_next)

            # 存储数据
            self.s_hoder[idx] = s_flat
            self.a_hoder[idx] = a
            self.r_hoder[idx] = r
            self.s_next_hoder[idx] = s_next_flat
            self.logprob_a_hoder[idx] = logprob_a
            self.done_hoder[idx] = done
            self.dw_hoder[idx] = done

            self.trajectory_step += 1

        except Exception as e:
            logger.error(f"存储数据时发生错误: {e}. 跳过此数据点。")

    def _flatten_state(self, s):
        """将状态展平并对齐维度"""
        if isinstance(s, (list, tuple)):
            state_image, context_features = s
            state_image = np.array(state_image, dtype=np.float32).ravel()
            context_features = np.array(context_features, dtype=np.float32).ravel()
            s = np.concatenate([state_image, context_features])
        else:
            s = np.array(s, dtype=np.float32).ravel()

        if s.shape[0] > self.state_dim:
            s = s[: self.state_dim]
        elif s.shape[0] < self.state_dim:
            s = np.pad(s, (0, self.state_dim - s.shape[0]))

        return s

    def save_model(self):
        save_dir = self.model_file if self.model_file else "./model_weight"
        os.makedirs(save_dir, exist_ok=True)

        critic_path = os.path.join(save_dir, "ppo_critic.pth")
        actor_path = os.path.join(save_dir, "ppo_actor.pth")
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.actor.state_dict(), actor_path)
        logger.info(f"PPO模型已保存到 {save_dir}")

    def load_model(self):
        save_dir = self.model_file if self.model_file else "./model_weight"
        critic_path = os.path.join(save_dir, "ppo_critic.pth")
        actor_path = os.path.join(save_dir, "ppo_actor.pth")

        if os.path.exists(critic_path) and os.path.exists(actor_path):
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=self.device)
            )
            self.actor.load_state_dict(
                torch.load(actor_path, map_location=self.device)
            )
            logger.info(f"PPO模型已从 {save_dir} 加载")
        else:
            logger.warning(f"未找到PPO模型文件，将从头开始训练")

    def update_target_network(self):
        pass  # 如果需要实现目标网络更新，可以在这里添加代码
