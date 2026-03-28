import sys
import os

# allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np

from env.robot_env import RobotEnv
from models.dqn_model import DQN

# initialize environment
env = RobotEnv()

state_size = 4
action_size = 3

# create neural network
model = DQN(state_size, action_size)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# training settings
episodes = 2000
epsilon = 1.0

rewards_history = []

for episode in range(episodes):

    state = torch.tensor(env.get_state()).float()
    total_reward = 0

    for step in range(100):

        # epsilon-greedy action
        if random.random() < epsilon:
            action = random.randint(0,2)
        else:
            q_values = model(state)
            action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)

        next_state = torch.tensor(next_state).float()

        q_values = model(state)

        target = q_values.clone().detach()
        target[action] = reward

        loss = torch.mean((q_values - target) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

        if done:
            break

    # decay epsilon AFTER episode finishes
    epsilon = max(0.1, epsilon * 0.995)

    rewards_history.append(total_reward)

    print("Episode:", episode, "Reward:", total_reward)

# ------------------------------
# Smooth training graph
# ------------------------------

window = 20
avg_rewards = []

for i in range(len(rewards_history)):
    avg = np.mean(rewards_history[max(0, i-window):i+1])
    avg_rewards.append(avg)

plt.plot(avg_rewards)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Robot Training Performance")

plt.show()

# ------------------------------
# Save trained model
# ------------------------------

torch.save(model.state_dict(), "robot_navigation_model.pth")