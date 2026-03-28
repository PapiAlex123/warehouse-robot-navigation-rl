import torch
import time

from env.robot_env import RobotEnv
from models.dqn_model import DQN

env = RobotEnv()

model = DQN(4,3)
model.load_state_dict(torch.load("robot_navigation_model.pth"))
model.eval()

state = torch.tensor(env.get_state()).float()

steps = 0
max_steps = 1000

while True:

    q_values = model(state)
    action = torch.argmax(q_values).item()

    next_state, reward, done = env.step(action)

    state = torch.tensor(next_state).float()

    steps += 1

    time.sleep(0.05)

    if done:
        print("Goal reached or obstacle hit")
        print("RL path length:", steps)
        break

    if steps >= max_steps:
        print("Max steps reached")
        print("RL path length:", steps)
        break