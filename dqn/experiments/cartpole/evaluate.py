import time
from itertools import count

import torch
import gymnasium as gym

from dqn.model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    state, info = env.reset()
 
    n_observations = len(state)

    dqn = DQN(n_observations, env.action_space, mid_layer_size=256, mode='evaluate')
    dqn.load('model/cart_pole.pth')

    for i in count():
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)    
        action = dqn.select_action(state)
        
        observation, reward, terminated, truncated, info = env.step(action.item())
         
        if terminated or truncated:
            break
        state = observation
        
        time.sleep(0.1)

    env.close()
