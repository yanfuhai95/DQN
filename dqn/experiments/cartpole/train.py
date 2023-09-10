import os
import time
import random
from itertools import count

import torch
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

from dqn.model import DQN

# Set up random seed with current time for randomness.
random.seed(time.time())

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

episode_durations = []
episode_rewards = []

fig, axs = plt.subplots(ncols=2, figsize=(24, 10))
    
def plot_result(show_result=False):
    # Duration
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        axs[0].set_title('Result')
    else:
        axs[0].cla()
        axs[0].set_title('Training...')
        
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Duration')
    axs[0].plot(durations_t.numpy())
    
    # Take 10 episode averages and plot them too
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        axs[0].plot(means.numpy())

    # Reward
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        axs[1].set_title('Result')
    else:
        axs[1].cla()
        axs[1].set_title('Training...')
        
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Reward')
    axs[1].plot(rewards_t.numpy())
    
    # Take 10 episode averages and plot them too
    if len(rewards_t) >= 10:
        means = rewards_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        axs[1].plot(means.numpy())

    plt.pause(0.001)   # pause a bit so that plots are updated

    if show_result:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state, info = env.reset()

    n_observations = len(state)

    dqn = DQN(
        n_observations, env.action_space, 
        batch_size=BATCH_SIZE, 
        gamma=GAMMA, 
        lr=LR, 
        tau=TAU, 
        eps_start=EPS_START, 
        eps_end=EPS_END, 
        eps_decay=EPS_DECAY,
        mid_layer_size=256,
        mode='train')

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0.

        for t in count():
            action = dqn.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            total_reward += reward
            
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device).unsqueeze(0)

            dqn.optimize(state, action, reward, next_state)
        
            # Move to next state
            state = next_state

            if done:
                episode_durations.append(t + 1)
                break

        episode_rewards.append(total_reward)
        plot_result()

    if not os.path.exists('model'):
        os.makedirs('model')
        
    dqn.save('model/cart_pole.pth')
    
    print('Complete')
    plot_result(show_result=True)
    plt.ioff()
    plt.show()
