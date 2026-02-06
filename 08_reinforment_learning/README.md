# Reinforcement Learning

This directory contains reinforcement learning examples implemented using neural networks. The focus is on understanding
core reinforcement learning concepts and how neural networks are used to approximate
value functions and policies.

## Deep Q-Learning (DQN)

Deep Q-Learning combines classical Q-learning with neural networks to handle larger or continuous state spaces.

These examples cover:

- Markov Decision Processes (MDPs)
- Action-value functions Q(s, a)
- ε-greedy exploration
- Temporal-Difference (TD) learning
- Neural network approximation of Q-values

## Deep Deterministic Policy Gradient (DDPG)

DDPG is an actor–critic algorithm designed for continuous action spaces. It uses separate neural networks for the
policy (actor) and value function (critic).

These examples cover:

- Continuous action spaces
- Actor–critic architecture
- Experience replay
- Target networks for stability
- Reward shaping techniques

## Projects

- [01_dqn](01_dqn)
    - Discrete maintenance decision problem with stochastic state transitions.
    - Deep Q-Network (DQN) using a small fully connected neural network.
    - One-hot encoded states and greedy policy inspection.
    - Implemented using TensorFlow.

- [02_ddpg](02_ddpg)
    - Continuous control problem of parking a car in a simulated environment.
    - Deep Deterministic Policy Gradient (DDPG) with separate actor and critic networks.
    - Reward shaping using a Gym wrapper for improved learning.
    - Implemented using TensorFlow.

Additional reinforcement learning examples may be added over time following the same learning-oriented structure.
