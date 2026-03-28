
# Title
# Autonomous Warehouse Robot Navigation using Reinforcement Learning

# Project Overview

This project demonstrates reinforcement learning for autonomous robot navigation in a simulated warehouse environment using PyBullet and PyTorch.

The robot learns to navigate from a starting location to a pickup station while avoiding obstacles representing warehouse shelves.

# Features
* PyBullet robotics simulation
* Deep Q-Network (DQN) navigation
* warehouse-style obstacle environment
* reward shaping for goal-directed movement
* training performance visualization
* trained model inference

# Environment
# State representation:

* robot x position
* robot y position
* distance to goal
* distance to nearest obstacle
# Model Architecture
* Input layer: 4
* Hidden layer: 64
* Hidden layer: 64
* Output layer: 3 actions


# Actions:

* move forward
* move left
* move right
* Training

# Training parameters:

* episodes: 2000
* learning rate: 0.001
* epsilon-greedy exploration

# Reward system:

*  +100 for reaching goal
* -100 for collision
* +1 for moving closer to goal
* small step penalty

# Results

The training graph shows reward improvement as the robot learns navigation behavior.


# How to Run

* Install dependencies:

* pip install -r requirements.txt

# Train the robot:

* python training/train_agent.py

* Run the trained robot:

* python run_trained_robot.py
