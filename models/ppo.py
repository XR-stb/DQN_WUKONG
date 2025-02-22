import numpy as np
import torch
import copy
import math
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn
from models.base_agent import BaseAgent
import os


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(
            state_dim, net_width
        )  # 输入维度为 state_dim，输出维度为 net_width
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return self.l3(n)  # 返回第三层的输出

    def pi(self, state, softmax_dim=0):
        n = self.forward(state)
        prob = F.softmax(n, dim=softmax_dim)  # 使用 softmax 计算动作概率
        return prob

    def get_log_prob(self, state, action):
        # 添加获取动作对数概率的方法
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)


class Critic(nn.Module):
    def __init__(self, state_dim, net_width):
        super(Critic, self).__init__()
        self.C1 = nn.Linear(
            state_dim, net_width
        )  # 输入维度为 state_dim，输出维度为 net_width
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)  # 最后一层不使用激活函数
        return v


import logging

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class PPO(BaseAgent):
    def __init__(self, state_dim, action_dim, context_dim, config, model_file):
        super(PPO, self).__init__(config, model_file)
        # 初始化 PPO 的超参数
        self.gamma = config.get("gamma", 0.9)
        self.lambd = config.get("lambda", 0.95)  # GAE 参数
        self.l2_reg = config.get("l2_reg", 0.001)  # L2 正则化系数
        self.initial_epsilon = config.get("initial_epsilon", 0.5)
        self.final_epsilon = config.get("final_epsilon", 0.009)
        self.epsilon_decay = config.get("epsilon_decay", 0.0002)
        self.lr = config.get("lr", 0.001)
        self.batch_size = config.get("batch_size", 32)
        self.epsilon_clip = config.get("epsilon_clip", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.entropy_coef_decay = config.get("entropy_coef_decay", 0.995)
        self.T_horizon = config.get("T_horizon", 2048)
        self.net_width = config.get("net_width", 64)
        self.K_epochs = config.get("K_epochs", 4)
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
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

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

    def choose_action(self, s):
        deterministic = False

        # 解包 s 中的两个部分
        state_image_array, context_features = s

        # 处理 state_image_array 部分
        state_image_tensor = torch.from_numpy(state_image_array).float().to(self.device)

        # 处理 context_features 部分
        context_features_tensor = torch.tensor(
            context_features, dtype=torch.float32
        ).to(self.device)

        # 展平图像部分并拼接上下文特征
        state_tensor = torch.cat(
            [state_image_tensor.flatten(), context_features_tensor]
        )

        # 检查输入张量的形状
        if state_tensor.shape[0] != self.state_dim:
            logger.warning(
                f"Input feature dimension {state_tensor.shape[0]} does not match "
                f"expected input dimension {self.state_dim}. Adjusting input dimension."
            )
            # 如果输入特征维度不匹配，截取或填充输入数据
            if state_tensor.shape[0] > self.state_dim:
                state_tensor = state_tensor[
                    : self.state_dim
                ]  # 截取前 self.state_dim 个元素
            else:
                padding = torch.zeros(
                    self.state_dim - state_tensor.shape[0], device=self.device
                )
                state_tensor = torch.cat([state_tensor, padding])  # 填充 0

        with torch.no_grad():
            # 输入 Actor 网络计算动作概率分布
            pi = self.actor.pi(state_tensor, softmax_dim=0)

            if deterministic:
                # 确定性选择动作（选择概率最大的动作）
                a = torch.argmax(pi).item()
                return a, None
            else:
                # 采样动作（基于概率分布）
                m = Categorical(pi)
                a = m.sample().item()
                pi_a = pi[a].item()
                return a, pi_a

    def train_network(self):
        self.entropy_coef *= self.entropy_coef_decay

        s = torch.from_numpy(self.s_hoder).to(self.device)
        a = torch.from_numpy(self.a_hoder).to(self.device)
        r = torch.from_numpy(self.r_hoder).to(self.device)
        s_next = torch.from_numpy(self.s_next_hoder).to(self.device)
        old_prob_a = torch.from_numpy(self.logprob_a_hoder).to(self.device)
        done = torch.from_numpy(self.done_hoder).to(self.device)
        dw = torch.from_numpy(self.dw_hoder).to(self.device)

        # 计算优势和回报
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            # 将布尔张量转换为浮点张量
            done_float = done.float()
            dw_float = dw.float()

            # 使用GAE计算优势
            deltas = r + self.gamma * vs_ * (1 - dw_float) - vs
            advantages = torch.zeros_like(deltas)
            advantage = 0
            for t in reversed(range(len(deltas))):
                # 使用浮点张量进行计算
                advantage = deltas[t] + self.gamma * self.lambd * advantage * (1 - done_float[t])
                advantages[t] = advantage
            
            returns = advantages + vs
            
            if self.adv_normalization:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            # 计算新的动作概率
            new_probs = self.actor.pi(s, softmax_dim=1)
            new_log_probs = torch.log(new_probs.gather(1, a) + 1e-8)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_prob_a)
            
            # 裁剪目标函数
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            
            # 计算actor损失
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 添加熵正则化
            dist = Categorical(new_probs)
            entropy_loss = -dist.entropy().mean()
            actor_loss += self.entropy_coef * entropy_loss
            
            # 更新actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # 更新critic
            value_pred = self.critic(s)
            value_loss = F.mse_loss(value_pred, returns)
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

    def store_data(self, s, a, r, s_next, done, logprob_a, this_epo_step):
        idx = this_epo_step

        try:
            # 处理状态数据
            if isinstance(s, (list, tuple)):
                # 如果是图像和上下文特征的组合
                state_image, context_features = s
                state_image = np.array(state_image, dtype=np.float32).ravel()
                context_features = np.array(context_features, dtype=np.float32).ravel()
                s = np.concatenate([state_image, context_features])
            else:
                s = np.array(s, dtype=np.float32).ravel()

            # 处理next_state数据
            if isinstance(s_next, (list, tuple)):
                state_image, context_features = s_next
                state_image = np.array(state_image, dtype=np.float32).ravel()
                context_features = np.array(context_features, dtype=np.float32).ravel()
                s_next = np.concatenate([state_image, context_features])
            else:
                s_next = np.array(s_next, dtype=np.float32).ravel()

            # 确保维度匹配
            if s.shape[0] > self.state_dim:
                s = s[:self.state_dim]
                s_next = s_next[:self.state_dim]
            elif s.shape[0] < self.state_dim:
                s = np.pad(s, (0, self.state_dim - s.shape[0]))
                s_next = np.pad(s_next, (0, self.state_dim - s_next.shape[0]))

            # 存储数据
            self.s_hoder[idx] = s
            self.a_hoder[idx] = a
            self.r_hoder[idx] = r
            self.s_next_hoder[idx] = s_next
            self.logprob_a_hoder[idx] = logprob_a
            self.done_hoder[idx] = done
            self.dw_hoder[idx] = done

        except Exception as e:
            logger.error(f"存储数据时发生错误: {e}. 跳过此数据点。")

    def save_model(self):
        save_dir = "./model_weight"
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        episode = 50000
        torch.save(
            self.critic.state_dict(), 
            os.path.join(save_dir, "ppo_critic{}.pth".format(episode))
        )
        torch.save(
            self.actor.state_dict(), 
            os.path.join(save_dir, "ppo_actor{}.pth".format(episode))
        )

    def load_model(self):
        episode = 50000
        self.critic.load_state_dict(
            torch.load(
                "./model_weight/ppo_critic{}.pth".format(episode),
                map_location=self.device,
            )
        )
        self.actor.load_state_dict(
            torch.load(
                "./model_weight/ppo_actor{}.pth".format(episode),
                map_location=self.device,
            )
        )

    def update_target_network(self):
        pass  # 如果需要实现目标网络更新，可以在这里添加代码
