# Deep Deterministic Policy Gradient (DDPG)

This example demonstrates a reinforcement learning approach for an autonomous parking task in a simple 2D simulation.
The goal is to train an agent to park a car in a designated parking slot using continuous control of throttle and
steering. The agent learns through trial and error, receiving rewards based on its performance in the parking task.

The project is implemented in two layers:

- A custom physics environment (`env.py`) that simulates vehicle dynamics, collisions, and parking success
- A Gym-compatible wrapper (`gym_env.py`) that provides normalized observations, reward shaping, and episode
  handling for training deep RL agents

## Model

The learning agent is based on Deep Deterministic Policy Gradient (DDPG), an actor–critic algorithm designed for
continuous action spaces.

The model consists of:

- An **Actor network**: A neural network that maps the current state of the environment to a deterministic action  
  (throttle, steering)
- A **Critic network**: A neural network that estimates the expected return (Q-value) of a given state–action pair

### State Representation

Each observation consists of six normalized values:

- x-position of the car (normalized)
- y-position of the car (normalized)
- sin(θ) and cos(θ) for orientation
- velocity (normalized)
- steering angle (normalized)

### Action Space

The action is a 2D continuous vector:

- throttle ∈ [-1, 1]
- steering ∈ [-1, 1]

## Reward Design

The reward function combines sparse and dense components:

- **Sparse rewards**
    - Large positive reward for successfully parking
    - Large negative reward for crashing
    - Small negative step reward to encourage faster parking

- **Reward shaping (Gym wrapper only)**
    - Penalty proportional to the distance from the parking slot center (encourages centering)
    - Penalty for misalignment (encourages parallel parking)
    - Alignment penalty is only applied when the car is close to the parking slot to stabilize learning

This separation keeps the environment logic clean while allowing flexible reward tuning during training.

## Training

Training is performed using off-policy reinforcement learning with experience replay.

Key components:

- Replay buffer storing transitions (state, action, reward, next state, done)
- Target actor and critic networks for stable learning
- Soft target updates using Polyak averaging
- Gaussian exploration noise with linear decay
- Periodic evaluation without exploration noise
- Early stopping after consecutive perfect evaluation runs

All hyperparameters, reward weights, and environment constants are centralized in `config.py`.

## Reproducibility

To ensure reproducible results across runs:

- Python’s `random` module is seeded
- NumPy random number generation is seeded
- TensorFlow random number generation is seeded

The default seed used in this project is:

```
SEED = 42
```

Changing the seed may lead to slightly different learned parameters and evaluation results.

## Implementation

### Environment (`env.py`)

The custom environment implements a simple 2D parking scenario with the following features:

- Simple kinematic vehicle model
- Steering and velocity damping
- Collision detection using vehicle corner geometry
- Parking success checks based on position and orientation

No reinforcement learning libraries are used at this level.

### Gym Wrapper (`gym_env.py`)

The Gym wrapper provides a standardized interface for training RL agents and includes:

- Normalized observations
- Continuous action space definition
- Reward shaping for centering and alignment
- Episode truncation based on step limits

This separation allows the same environment to be reused for manual control and learned policies.

### Training (`train.py`)

The training pipeline includes the following components:

- DDPG agent implementation (actor–critic)
- Replay buffer
- Exploration noise
- Evaluation and model checkpointing
- Early stopping logic

### Visualization and Interaction

- `manual.py`: Manual keyboard control for exploring the environment and understanding the task

- `play.py`: Runs a trained actor model in the GUI (autopilot mode)

- `gui.py`: Tkinter-based visualization shared by both manual and learned control

## How to Run

1. Ensure you have Python installed (version 3.9 or higher recommended).
    ```bash
    python --version
    ```

2. Install the required dependencies.
    ```bash
    python -m pip install -r requirements.txt
    ```

3. Train the agent.
    ```bash
    python train.py
    ```

4. Run the trained policy.
    ```bash
    python play.py
    ```

5. Drive manually using the keyboard.
    ```bash
    python manual.py
    ```
