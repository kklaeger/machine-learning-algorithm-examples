import numpy as np
import tensorflow as tf

from env import MachineMaintenanceEnv, OK, WARNING, CRITICAL, FAILED, DO_NOTHING, INSPECT, REPAIR

EPISODES = 1000
GAMMA = 0.95
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
MAX_STEPS_PER_EPISODE = 50
SEED = 42

STATE_NAMES = {
    OK: "OK",
    WARNING: "WARNING",
    CRITICAL: "CRITICAL",
    FAILED: "FAILED"
}

ACTION_NAMES = {
    DO_NOTHING: "DO_NOTHING",
    INSPECT: "INSPECT",
    REPAIR: "REPAIR"
}

NUM_POSSIBLE_STATES = len(STATE_NAMES)
NUM_POSSIBLE_ACTIONS = len(ACTION_NAMES)

np.random.seed(SEED)
tf.random.set_seed(SEED)


def one_hot_state(state: int, num_possible_states: int) -> np.ndarray:
    """
    Converts a discrete state index into a one-hot encoded vector.

    E.g., if state=2 and num_possible_states=4, the output will be [0, 0, 1, 0].

    Parameters:
        state (int):                The index of the current state.
        num_possible_states (int):  The total number of possible states in the environment.
    Returns:
        one_hot_vec (np.ndarray): A one-hot encoded vector representing the state.
    """
    one_hot_vec = np.zeros(num_possible_states, dtype=np.float32)
    one_hot_vec[state] = 1.0
    return one_hot_vec


def build_q_network(num_possible_states: int, num_possible_actions: int):
    """
    Builds a simple feedforward neural network to approximate the Q-function.

    Parameters:
        num_possible_states (int):  The number of discrete states in the environment (input size).
        num_possible_actions (int): The number of discrete actions in the environment (output size).
    Returns:
        model (tf.keras.Model): A Keras Sequential model representing the Q-network.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(num_possible_states,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(num_possible_actions),
        ]
    )


def train_dqn(
        episode: int = EPISODES,
        gamma: float = GAMMA,
        lr: float = LR,
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        epsilon_decay: float = EPSILON_DECAY,
        max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
):
    """
    Trains a Deep Q-Network (DQN) on the MachineMaintenanceEnv environment.

    Parameters:
        episode (int):                  The number of episodes to train for.
        gamma (float):                  The discount factor for future rewards.
        lr (float):                     The learning rate for the optimizer.
        epsilon_start (float):          The initial epsilon value for the epsilon-greedy policy.
        epsilon_end (float):            The minimum epsilon value after decay.
        epsilon_decay (float):          The decay rate for epsilon after each episode.
        max_steps_per_episode (int):    The maximum number of steps to take in each episode before terminating.
    Returns:
        model (tf.keras.Model): The trained Q-network model.
    """
    print("Starting DQN training...")

    # Define the environment and the Q-network
    env = MachineMaintenanceEnv(seed=SEED)

    # Build the Q-network and set up the optimizer and loss function
    model = build_q_network(NUM_POSSIBLE_STATES, NUM_POSSIBLE_ACTIONS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Epsilon-greedy training loop
    epsilon = epsilon_start
    for episode in range(episode):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps_per_episode):
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                # Explore: choose a random action
                action = np.random.randint(NUM_POSSIBLE_ACTIONS)
            else:
                # Exploit: choose the action with the highest Q-value
                state_vec = one_hot_state(state, NUM_POSSIBLE_STATES)[None, :]
                q_values = model(state_vec, training=False).numpy()[0]
                action = int(np.argmax(q_values))

            # Take the action in the environment
            next_state, reward, episode_over = env.step(action)
            total_reward += reward

            # Convert the current state and next state to one-hot vectors (network input)
            state_vec = one_hot_state(state, NUM_POSSIBLE_STATES)[None, :]
            next_state_vec = one_hot_state(next_state, NUM_POSSIBLE_STATES)[None, :]

            # Compute the current Q-values Q(s, ) predicted by the network
            q_pred = model(state_vec, training=True)
            q_pred_np = q_pred.numpy()[0]

            # Compute the target Q-value for the taken action using the Bellman equation
            if episode_over:
                target_q = reward
            else:
                q_next = model(next_state_vec, training=False).numpy()[0]
                target_q = reward + gamma * float(np.max(q_next))  # Bellman update: Q(s, a) = r + γ * max_a' Q(s', a')

            # Build the target Q-value vector: update only the Q-value of the taken action
            q_target_vec = q_pred_np.copy()
            q_target_vec[action] = target_q
            q_target_vec = q_target_vec[None, :]

            # TPerform a gradient descent step to fit Q(s, ) to the Bellman target
            with tf.GradientTape() as tape:
                q_out = model(state_vec, training=True)
                loss = loss_fn(q_target_vec, q_out)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            state = next_state
            if episode_over:
                break

        # Decay epsilon after each episode
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Print training progress every 200 episodes
        if (episode + 1) % 200 == 0:
            print(f"Episode {episode + 1:4d} | total_reward={total_reward:6.2f} | epsilon={epsilon:.3f}")

    return model


def print_greedy_policy(model: tf.keras.Model):
    """
    Prints the greedy policy derived from the trained Q-network for each non-terminal state.

    Parameters:
        model (tf.keras.Model): The trained Q-network model.
    """
    print("\nLearned greedy policy (best action per state):")
    for state in [OK, WARNING, CRITICAL]:
        state_vec = one_hot_state(state, NUM_POSSIBLE_STATES)[None, :]
        q = model(state_vec, training=False).numpy()[0]
        a = int(np.argmax(q))
        print(f"  {STATE_NAMES[state]:8s} -> {ACTION_NAMES[a]:11s} | Q={np.round(q, 2)}")


def main():
    model = train_dqn()
    print_greedy_policy(model)


if __name__ == "__main__":
    main()
