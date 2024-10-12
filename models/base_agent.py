# models/base_agent.py
# coding=utf-8
import torch
import os
from abc import ABC, abstractmethod
from collections import deque

class BaseAgent(ABC):
    def __init__(self, config, model_file):
        self.model_name = self.__class__.__name__
        self.model_file = model_file
        self.model_path = os.path.join(model_file, f"{self.model_name}.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = deque(maxlen=config.get('replay_size', 2000))
        self.gamma = config.get('gamma', 0.9)
        self.epsilon = config.get('initial_epsilon', 0.5)
        self.epsilon_min = config.get('final_epsilon', 0.009)
        self.epsilon_decay = config.get('epsilon_decay', 0.0002)
        self.lr = config.get('lr', 0.001)
        self.batch_size = config.get('batch_size', 32)
        # Other common initializations can go here

    @abstractmethod
    def choose_action(self, state):
        """
        Choose an action based on the state.
        Returns:
            tuple: (action, log_prob)
        """
        pass

    @abstractmethod
    def store_data(self, state, action, reward, next_state, done, log_prob=None):
        pass

    @abstractmethod
    def train_network(self):
        pass

    @abstractmethod
    def update_target_network(self):
        pass

    def save_model(self):
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.policy_net.state_dict(), self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.policy_net.load_state_dict(torch.load(self.model_path, weights_only=True))
            self.policy_net.to(self.device)
