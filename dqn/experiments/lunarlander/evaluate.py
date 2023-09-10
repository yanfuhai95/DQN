import time
from itertools import count

import torch
import gymnasium as gym

from dqn.model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human")
    state, info = env.reset()
 
    n_observations = len(state)

    dqn = DQN(n_observations, env.action_space, mid_layer_size=256, mode='evaluate')
    dqn.load('model/lunar_lander.pth')

    total_reward = 0
    episode_duration = 0
    
    for i in count():
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)    
        action = dqn.select_action(state)
        
        observation, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        
        if terminated:
            episode_duration = i
            print("Agent landed safely on moon!")
            break
        
        if truncated:
            episode_duration = i
            print("Agent is crashed!")
            break
        
        state = observation

    print("Total reward: ", total_reward)
    print("Episode duration: ", episode_duration + 1)
    env.close()
