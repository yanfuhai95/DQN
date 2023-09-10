import math
from abc import ABC, abstractmethod
import random
from typing import List
from collections import namedtuple, deque

import gymnasium as gym
from gymnasium.spaces import Space
import torch
from torch import nn
import torch.optim as optim

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class RLModel(ABC):
    @abstractmethod
    def optimize(self, state, action, reward, next_state):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


class ReplayMemory(object):
    """ReplayMemory implements the experience replay.

    Experience replay is a technique in reinforcement learning where the agent stores and samples experiences from an experience replay buffer during training.
    It aims to improve the efficiency and stability of the learning process.
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class CustomNeuralNetwork(nn.Module):
    """CustomNeuralNetwork implements a basic feedforward neural network 
    with customized layers and activation functions.
    """

    def __init__(self, n_input: int, layers: List[int]):
        """
        Constructor of :class:`CustomNeuralNetwork`.

        Args:
            n_input (int): Size of input layer.
            layers (List[int]): Sizes of middle layers.
            activation_fn (nn.Module): Activation function of middle layers.
            out_activation_fn (nn.Module): Activation function of ouput layer.
        """
        super(CustomNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        prev_layer_size = n_input

        for layer_size in layers[:-1]:
            self.layers.append(nn.Linear(prev_layer_size, layer_size))
            self.layers.append(nn.ReLU())
            prev_layer_size = layer_size

        self.layers.append(nn.Linear(prev_layer_size, layers[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DQN(RLModel):
    """Implementation of Deep Q-learning Network algorithm.
    """

    def __init__(self,
                 n_observations: int,
                 action_space: Space,
                 lr=1e-4,
                 batch_size=100,
                 tau=0.005,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=1000,
                 gamma=0.99,
                 loss_fn=nn.SmoothL1Loss,
                 optimizer=optim.AdamW,
                 replay_buffer_size=10000,
                 mid_layer_size=128,
                 mode='train'):
        self.batch_size = batch_size
        self.action_space = action_space
        self.criterion = loss_fn()
        self.mode = mode

        # Discount factor
        self.gamma = gamma

        # Update rate of target network
        self.tau = tau

        # Learning rate
        self.lr = lr

        # Experience replay
        self.replay_buffer = ReplayMemory(replay_buffer_size)

        # Parameters for epsilon-greedy policy
        self.eps_steps = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # Enable CUDA
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        output_layer_size = self.action_space.n

        # The policy_net is used to select action base on given state in training progress.
        self.policy_net = CustomNeuralNetwork(
            n_observations,
            [mid_layer_size, mid_layer_size, output_layer_size]).to(self.device)

        # The target_net is a model with the same structure as the policy_net, but its parameters are
        # not updated during the training process.
        # The target_net provides a relatively stable target value for computing the loss function
        # and updating the parameters of the policy_net.
        # The parameters of the target_net are periodically copied from the policy_net, for example,
        # every certain number of time steps, to maintain the stability of the target_net.
        self.target_net = CustomNeuralNetwork(
            n_observations,
            [mid_layer_size, mid_layer_size, output_layer_size]).to(self.device)

        # Copy parameters of policy_net to target_net at the very beginning.
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Set up optimizer
        self.optimizer = optimizer(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True)

    def optimize(self, state, action, reward, next_state):        
        self.replay_buffer.push(state, action, reward, next_state)
        
        if len(self.replay_buffer) >= self.batch_size:
            self.__optimize(state, action, reward, next_state)

        self.__update_parameters()
        
        
    def __optimize(self, state, action, reward, next_state):
        # This converts batch-array of Transitions to Transition of batch-arrays.
        # For example:
        #   Given transition definition: type_name=Transition, names=('state', 'action', 'reward', 'next_state')
        #   Original (Array of transitions):
        #      [
        #           ('s_0', 'a_0', 1, 's_1'),
        #           ('s_2', 'a_4', 8, 's_9'),
        #       ]
        #   After transposing (Transitions of array):
        #      [
        #           ('s_0', 's_2'),   state
        #           ('a_0', 'a_4'),   action
        #           (1,     8),       reward
        #           ('s_1', 's_9'),   next_state
        #      ]
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken.
        # These are the actions which would've been taken for each batch state
        # according to policy_net
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Q(s_t', a'), where s_t' is the next state, and a' is the next action
        # in state s_t'.
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_q_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1)[0]

        # Compute the expected Q values based on Bellman equation.
        expected_q_values = (next_q_values * self.gamma) + reward_batch

        loss = self.criterion(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def select_action(self, state):
        if self.mode == 'train':
            # Select action based on epsilon-greedy policy.
            # At the very beginning, the model would prefer to randomly select
            # a action from action space.
            eps_threshold = self.eps_end + \
                (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.eps_steps / self.eps_decay)
            self.eps_steps += 1

            if random.random() > eps_threshold:
                return self.__select_action(state)
            else:
                # Randomly select a action from action space.
                return torch.tensor([[self.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            return self.__select_action(state)

    def __select_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def __update_parameters(self):
        # Soft update of the target network with given update rate TAU.
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path))
