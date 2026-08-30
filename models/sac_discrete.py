# models/sac_discrete.py
# coding=utf-8
"""
SAC-Discrete v2: 带CNN视觉特征提取的离散动作 Soft Actor-Critic

v2 改进（针对收敛失败的修复）:
1. CNN特征提取: 用轻量CNN压缩224×224图像为128维特征，替代150K维MLP
2. 奖励归一化: 在训练时对奖励做 running normalization
3. 梯度裁剪: 更保守的梯度裁剪防止训练不稳定
4. 更大replay buffer: 100K (v1是50K)，off-policy需要更多数据
5. warmup: 前1000步纯随机探索，收集多样性经验
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from models.base_agent import BaseAgent
from collections import deque
import os
import random
import logging

logger = logging.getLogger(__name__)


# ==================== CNN 特征提取器 ====================

class CNNFeatureExtractor(nn.Module):
    """
    轻量级CNN: 3×224×224 → 128维特征向量
    
    设计原则:
    - 4层卷积 + 全局平均池化 → 固定128维输出
    - BatchNorm稳定训练
    - 参数量约100K，不会太慢
    """
    
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            # 3×224×224 → 32×56×56
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32×56×56 → 64×14×14
            nn.Conv2d(32, 64, kernel_size=4, stride=4, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64×14×14 → 64×7×7
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 全局平均池化 → 64
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224) 图像张量，像素值 [0, 1]
        Returns:
            (batch, output_dim) 特征向量
        """
        features = self.conv(x)
        return self.fc(features)


# ==================== 网络定义 ====================

class SACDiscreteActor(nn.Module):
    """SAC Actor — 基于CNN特征+上下文特征输出动作概率"""
    
    def __init__(self, visual_dim, context_dim, action_dim, net_width):
        super().__init__()
        total_dim = visual_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(total_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, action_dim),
        )
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state)
    
    def get_action_probs(self, state):
        logits = self.forward(state)
        action_probs = F.softmax(logits, dim=-1)
        log_action_probs = torch.log(action_probs + 1e-8)
        return action_probs, log_action_probs


class SACDiscreteCritic(nn.Module):
    """SAC Critic — 双Q网络"""
    
    def __init__(self, visual_dim, context_dim, action_dim, net_width):
        super().__init__()
        total_dim = visual_dim + context_dim
        self.q1_net = nn.Sequential(
            nn.Linear(total_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, action_dim),
        )
        self.q2_net = nn.Sequential(
            nn.Linear(total_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, action_dim),
        )
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.q1_net(state), self.q2_net(state)


# ==================== 经验回放缓冲区 ====================

class ReplayBuffer:
    """经验回放 — 分别存储图像特征和上下文特征"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, visual_feat, context_feat, action, reward, next_visual_feat, next_context_feat, done):
        self.buffer.append((visual_feat, context_feat, action, reward,
                           next_visual_feat, next_context_feat, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        vis, ctx, act, rew, nvis, nctx, dones = zip(*batch)
        
        vis = np.array(vis, dtype=np.float32)
        ctx = np.array(ctx, dtype=np.float32)
        act = np.array(act, dtype=np.int64)
        rew = np.array(rew, dtype=np.float32)
        nvis = np.array(nvis, dtype=np.float32)
        nctx = np.array(nctx, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return vis, ctx, act, rew, nvis, nctx, dones
    
    def __len__(self):
        return len(self.buffer)


# ==================== 奖励归一化器 ====================

class RunningRewardNormalizer:
    """Running mean/std 奖励归一化，稳定训练"""
    
    def __init__(self, clip=5.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip = clip
    
    def update(self, reward):
        self.count += 1
        if self.count == 1:
            self.mean = reward
            self.var = 0.0
        else:
            old_mean = self.mean
            self.mean += (reward - old_mean) / self.count
            self.var += (reward - old_mean) * (reward - self.mean)
    
    def normalize(self, reward):
        std = max(np.sqrt(self.var / max(self.count, 1)), 1e-4)
        normalized = (reward - self.mean) / std
        return np.clip(normalized, -self.clip, self.clip)


# ==================== SAC-Discrete 主算法 ====================

class SAC_Discrete(BaseAgent):
    """
    SAC-Discrete v2 — 带CNN特征提取
    
    关键改进:
    - CNN处理视觉输入: 224×224×3 → 128维特征 (替代150K维flatten)
    - 奖励归一化: 稳定训练过程
    - Warmup探索: 前1000步纯随机，收集多样性初始经验
    """
    
    def __init__(self, state_dim, action_dim, context_dim, config, model_file):
        super(SAC_Discrete, self).__init__(config, model_file)
        
        # ---- 超参数 ----
        self.gamma = config.get("gamma", 0.98)
        self.lr = config.get("lr", 0.0003)
        self.critic_lr = config.get("critic_lr", 0.001)
        self.alpha_lr = config.get("alpha_lr", 0.0003)
        self.batch_size = config.get("batch_size", 64)
        self.tau = config.get("tau", 0.005)
        self.net_width = config.get("net_width", 256)
        self.buffer_size = config.get("replay_size", 100000)
        self.min_buffer_size = config.get("min_buffer_size", 500)
        self.target_entropy_ratio = config.get("target_entropy_ratio", 0.6)
        self.updates_per_step = config.get("updates_per_step", 2)
        self.warmup_steps = config.get("warmup_steps", 1000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.context_dim = context_dim
        
        # ---- CNN 特征提取器 ----
        self.visual_feature_dim = 128
        self.cnn = CNNFeatureExtractor(output_dim=self.visual_feature_dim).to(self.device)
        self.cnn_target = CNNFeatureExtractor(output_dim=self.visual_feature_dim).to(self.device)
        self.cnn_target.load_state_dict(self.cnn.state_dict())
        
        # ---- 计算MLP输入维度 ----
        self.mlp_input_dim = self.visual_feature_dim + context_dim
        # 保持兼容: state_dim用于replay buffer
        if isinstance(state_dim, tuple):
            self.image_shape = state_dim  # (H, W)
        else:
            self.image_shape = None
        self.state_dim = self.mlp_input_dim  # 覆盖父类
        
        # ---- 构建网络 ----
        self.actor = SACDiscreteActor(
            self.visual_feature_dim, context_dim, action_dim, self.net_width
        ).to(self.device)
        
        self.critic = SACDiscreteCritic(
            self.visual_feature_dim, context_dim, action_dim, self.net_width
        ).to(self.device)
        
        self.critic_target = SACDiscreteCritic(
            self.visual_feature_dim, context_dim, action_dim, self.net_width
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # ---- 优化器: CNN和Actor共享，Critic单独 ----
        self.actor_optimizer = optim.Adam(
            list(self.cnn.parameters()) + list(self.actor.parameters()),
            lr=self.lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.cnn.parameters()) + list(self.critic.parameters()),
            lr=self.critic_lr
        )
        
        # ---- 自动温度调节 ----
        self.target_entropy = -self.target_entropy_ratio * np.log(1.0 / action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        
        # ---- 经验回放 ----
        self.replay_buffer_store = ReplayBuffer(self.buffer_size)
        
        # ---- 奖励归一化 ----
        self.reward_normalizer = RunningRewardNormalizer()
        
        # ---- 训练计数器 ----
        self.total_steps = 0
        self.train_steps = 0
        
        logger.info(f"SAC-Discrete v2 (CNN) 初始化完成 | "
                     f"visual_dim={self.visual_feature_dim}, context_dim={context_dim}, "
                     f"mlp_input={self.mlp_input_dim}, action_dim={action_dim}, "
                     f"net_width={self.net_width}, buffer={self.buffer_size}, "
                     f"warmup={self.warmup_steps}, device={self.device}")
    
    def _extract_visual_and_context(self, s):
        """
        从原始状态中分离图像和上下文特征。
        
        输入 s 是 (image_array, context_features) 元组:
          - image_array: (1, 3, H, W) numpy 数组
          - context_features: list of 14 floats
        
        返回:
          - visual_feat: (visual_feature_dim,) numpy 数组
          - context_feat: (context_dim,) numpy 数组
        """
        if isinstance(s, (list, tuple)) and len(s) == 2:
            state_image, context_features = s
            
            # 确保 image 是 (1, 3, H, W)
            img = np.array(state_image, dtype=np.float32)
            if img.ndim == 3:
                img = img[np.newaxis, :]  # (3, H, W) → (1, 3, H, W)
            
            # 归一化到 [0, 1]
            if img.max() > 1.0:
                img = img / 255.0
            
            # 通过CNN提取特征
            with torch.no_grad():
                self.cnn.eval()
                img_tensor = torch.FloatTensor(img).to(self.device)
                visual_feat = self.cnn(img_tensor).cpu().numpy().flatten()
                self.cnn.train()
            
            context_feat = np.array(context_features, dtype=np.float32).ravel()
            
            return visual_feat, context_feat
        else:
            # 兼容：如果输入已经是flat数组
            s = np.array(s, dtype=np.float32).ravel()
            if s.shape[0] > self.mlp_input_dim:
                visual_feat = s[:self.visual_feature_dim]
                context_feat = s[self.visual_feature_dim:self.mlp_input_dim]
            else:
                visual_feat = np.zeros(self.visual_feature_dim, dtype=np.float32)
                context_feat = s[:self.context_dim] if s.shape[0] >= self.context_dim else np.pad(s, (0, self.context_dim - s.shape[0]))
            return visual_feat, context_feat
    
    def _combine_features(self, visual_feat, context_feat):
        """拼接视觉特征和上下文特征"""
        return np.concatenate([visual_feat, context_feat])
    
    def choose_action(self, s):
        """
        选择动作。
        Warmup期间纯随机探索，之后从策略分布采样。
        """
        try:
            visual_feat, context_feat = self._extract_visual_and_context(s)
            
            # Warmup期间纯随机
            if self.total_steps < self.warmup_steps:
                action = random.randint(0, self.action_dim - 1)
                return action, 0.0
            
            combined = self._combine_features(visual_feat, context_feat)
            state_tensor = torch.FloatTensor(combined).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_probs, log_action_probs = self.actor.get_action_probs(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = log_action_probs[0, action.item()].item()
            
            return action.item(), log_prob
            
        except Exception as e:
            logger.error(f"SAC选择动作时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return random.randint(0, self.action_dim - 1), 0.0
    
    def store_data(self, s, a, r, s_next, done, log_prob=None, this_epo_step=0):
        """存储经验"""
        try:
            visual_feat, context_feat = self._extract_visual_and_context(s)
            next_visual_feat, next_context_feat = self._extract_visual_and_context(s_next)
            
            # 更新奖励归一化器
            self.reward_normalizer.update(r)
            
            self.replay_buffer_store.push(
                visual_feat, context_feat, a, r,
                next_visual_feat, next_context_feat, float(done)
            )
            self.total_steps += 1
        except Exception as e:
            logger.error(f"SAC存储数据时发生错误: {e}")
    
    def train_network(self):
        """训练网络"""
        if len(self.replay_buffer_store) < self.min_buffer_size:
            return
        if self.total_steps < self.warmup_steps:
            return  # warmup期间不训练
        
        for _ in range(self.updates_per_step):
            self._update_networks()
    
    def _update_networks(self):
        """执行一次网络更新"""
        vis, ctx, actions, rewards, nvis, nctx, dones = self.replay_buffer_store.sample(self.batch_size)
        
        # 归一化奖励
        normalized_rewards = np.array([self.reward_normalizer.normalize(r) for r in rewards])
        
        vis_t = torch.FloatTensor(vis).to(self.device)
        ctx_t = torch.FloatTensor(ctx).to(self.device)
        states = torch.cat([vis_t, ctx_t], dim=1)
        
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(normalized_rewards).to(self.device).unsqueeze(1)
        
        nvis_t = torch.FloatTensor(nvis).to(self.device)
        nctx_t = torch.FloatTensor(nctx).to(self.device)
        next_states = torch.cat([nvis_t, nctx_t], dim=1)
        
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # 2. 更新 Critic
        with torch.no_grad():
            next_action_probs, next_log_probs = self.actor.get_action_probs(next_states)
            next_q1_target, next_q2_target = self.critic_target(next_states)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            next_v = (next_action_probs * (next_q_target - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + self.gamma * (1 - dones) * next_v
        
        current_q1, current_q2 = self.critic(states)
        current_q1 = current_q1.gather(1, actions)
        current_q2 = current_q2.gather(1, actions)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.cnn.parameters()) + list(self.critic.parameters()), 1.0
        )
        self.critic_optimizer.step()
        
        # 3. 更新 Actor
        action_probs, log_action_probs = self.actor.get_action_probs(states.detach())
        with torch.no_grad():
            q1, q2 = self.critic(states)
            min_q = torch.min(q1, q2)
        
        actor_loss = (action_probs * (self.alpha * log_action_probs - min_q)).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.cnn.parameters()) + list(self.actor.parameters()), 1.0
        )
        self.actor_optimizer.step()
        
        # 4. 更新温度参数
        with torch.no_grad():
            action_probs_detached, log_probs_detached = self.actor.get_action_probs(states)
            entropy = -(action_probs_detached * log_probs_detached).sum(dim=1).mean()
        
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        # 5. 软更新目标网络（包括CNN目标）
        self._soft_update_target()
        
        self.train_steps += 1
        
        if self.train_steps % 100 == 0:
            logger.info(f"SAC训练 | step={self.train_steps}, "
                        f"critic_loss={critic_loss.item():.4f}, "
                        f"actor_loss={actor_loss.item():.4f}, "
                        f"alpha={self.alpha:.4f}, "
                        f"entropy={entropy.item():.4f}, "
                        f"buffer={len(self.replay_buffer_store)}")
    
    def _soft_update_target(self):
        """软更新目标网络"""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        # 也更新CNN目标网络
        for target_param, param in zip(self.cnn_target.parameters(), self.cnn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def update_target_network(self):
        """兼容接口"""
        pass
    
    def save_model(self):
        """保存模型"""
        save_dir = self.model_file if self.model_file else "./model_weight"
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.cnn.state_dict(), os.path.join(save_dir, "sac_cnn.pth"))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "sac_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "sac_critic.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(save_dir, "sac_critic_target.pth"))
        torch.save(self.cnn_target.state_dict(), os.path.join(save_dir, "sac_cnn_target.pth"))
        torch.save({
            'log_alpha': self.log_alpha,
            'alpha': self.alpha,
            'reward_normalizer_mean': self.reward_normalizer.mean,
            'reward_normalizer_var': self.reward_normalizer.var,
            'reward_normalizer_count': self.reward_normalizer.count,
        }, os.path.join(save_dir, "sac_alpha.pth"))
        
        logger.info(f"SAC-Discrete v2 模型已保存到 {save_dir}")
    
    def load_model(self):
        """加载模型"""
        save_dir = self.model_file if self.model_file else "./model_weight"
        
        cnn_path = os.path.join(save_dir, "sac_cnn.pth")
        actor_path = os.path.join(save_dir, "sac_actor.pth")
        critic_path = os.path.join(save_dir, "sac_critic.pth")
        critic_target_path = os.path.join(save_dir, "sac_critic_target.pth")
        cnn_target_path = os.path.join(save_dir, "sac_cnn_target.pth")
        alpha_path = os.path.join(save_dir, "sac_alpha.pth")
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            if os.path.exists(cnn_path):
                self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
            
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            
            if os.path.exists(critic_target_path):
                self.critic_target.load_state_dict(
                    torch.load(critic_target_path, map_location=self.device))
            else:
                self.critic_target.load_state_dict(self.critic.state_dict())
            
            if os.path.exists(cnn_target_path):
                self.cnn_target.load_state_dict(
                    torch.load(cnn_target_path, map_location=self.device))
            else:
                self.cnn_target.load_state_dict(self.cnn.state_dict())
            
            if os.path.exists(alpha_path):
                alpha_data = torch.load(alpha_path, map_location=self.device)
                self.log_alpha = alpha_data['log_alpha'].to(self.device).requires_grad_(True)
                self.alpha = alpha_data['alpha']
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
                # 恢复奖励归一化状态
                if 'reward_normalizer_mean' in alpha_data:
                    self.reward_normalizer.mean = alpha_data['reward_normalizer_mean']
                    self.reward_normalizer.var = alpha_data['reward_normalizer_var']
                    self.reward_normalizer.count = alpha_data['reward_normalizer_count']
            
            logger.info(f"SAC-Discrete v2 模型已从 {save_dir} 加载")
        else:
            logger.warning(f"未找到SAC-Discrete模型文件，将从头开始训练")
