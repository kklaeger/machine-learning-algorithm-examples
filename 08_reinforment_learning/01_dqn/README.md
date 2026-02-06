# Deep Q-Network (DQN)

This example demonstrates a small reinforcement learning task for predictive machine maintenance. The goal is to choose
appropriate maintenance actions in order to keep a machine operational for as long as possible while minimizing
unnecessary intervention costs.

The project is intentionally minimal and focuses on clarity and interpretability rather than performance.

## Environment

The environment represents the health state of a machine:

- **OK** – machine operates normally
- **WARNING** – early signs of degradation
- **CRITICAL** – high risk of failure
- **FAILED** – terminal failure state

At each step, the agent can choose one of the following actions:

- **DO_NOTHING** – continue operation
- **INSPECT** – reduce uncertainty and risk
- **REPAIR** – restore the machine to a healthy state at a cost

State transitions are stochastic and depend on both the current state and the chosen action.

## Model

The agent is trained using Deep Q-Learning.

A neural network approximates the action-value function:

```
Q(s, a) = expected future return when taking action *a* in state *s*
```

### State Representation

States are encoded using one-hot vectors:

- OK → [1, 0, 0, 0]
- WARNING → [0, 1, 0, 0]
- CRITICAL → [0, 0, 1, 0]
- FAILED → terminal (not trained)

### Action Space

The network outputs one Q-value per action:

- DO_NOTHING
- INSPECT
- REPAIR

The greedy policy selects the action with the highest predicted Q-value.

## Reward Design

The reward function balances long-term operation against maintenance costs:

- Positive reward for continuing operation
- Small penalty for inspection
- Larger penalty for repair
- Large negative penalty for machine failure

This encourages the agent to delay repairs when safe, but act before catastrophic failure.

## Training

Training uses online Deep Q-Learning with:

- ε-greedy exploration
- Mean squared TD error loss
- Adam optimizer
- Discount factor γ for future rewards

## Reproducibility

To ensure reproducible results, random seeds are fixed for:

- NumPy
- TensorFlow
- Environment stochasticity

The default seed used is:

```
SEED = 42
```

## Implementation

### Environment (`env.py`)

Contains:

- State and action definitions
- Stochastic transition logic
- Reward assignment
- Terminal state handling

### DQN Implementation (`tensorflow_implementation.py`)

Contains:

- One-hot state encoding
- Q-network definition
- Training loop
- ε-greedy action selection
- Greedy policy evaluation

## How to Run

1. Ensure Python is installed (3.8+ recommended).
    ```bash
    python --version
    ```

2. Install dependencies.
    ```bash
    python -m pip install tensorflow numpy
    ```

3. Train the agent and print the learned policy.
    ```bash
    python tensorflow_implementation.py
    ```
