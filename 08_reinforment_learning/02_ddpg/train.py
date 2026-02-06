import os
import numpy as np
import time
import random
import tensorflow as tf

from gym_env import ParkingGymEnv
from config import *

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class ReplayBuffer:
    """
    Stores the experience tuples (state, action, reward, next_state, done) for training the DDPG agent.

    max_size is the maximum number of transitions to store. When the buffer is full, it will overwrite old data in a
    circular manner.
    """

    def __init__(self, max_size, state_dim, action_dim):
        """
        Initializes the replay buffer with pre-allocated numpy arrays for states, actions, rewards, next states, and
        done flags.

        Parameters:
            max_size (int): The maximum number of experience tuples the buffer can hold before overwriting old data.
        """
        self.max_size = max_size
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.ptr = 0  # pointer to the next index to write
        self.size = 0  # current size of the buffer (up to max_size)

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience tuple to the buffer at the current pointer index, then updates the pointer and size.

        Parameters:
            state (np.ndarray):         The current state of the environment.
            action (np.ndarray):        The action taken by the agent.
            reward (float):             The reward received after taking the action.
            next_state (np.ndarray):    The next state of the environment after taking the action.
            done (bool):                Indicates whether the episode has ended after taking the action.
        """
        # Store the experience at the current pointer index
        i = self.ptr
        self.state[i] = state
        self.action[i] = action
        self.reward[i] = reward
        self.next_state[i] = next_state
        self.done[i] = float(done)

        # Move pointer forward and wrap around if we exceed max size
        self.ptr = (self.ptr + 1) % self.max_size

        # Update current size (max is max_size)
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Returns a random batch of experience tuples from the buffer as TensorFlow tensors for training.

        Parameters:
            batch_size (int): The number of experience tuples to sample for training.
        Returns:
            A tuple of tensors: (states, actions, rewards, next_states, dones) each with shape (batch_size, ...).
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            tf.convert_to_tensor(self.state[idx]),
            tf.convert_to_tensor(self.action[idx]),
            tf.convert_to_tensor(self.reward[idx]),
            tf.convert_to_tensor(self.next_state[idx]),
            tf.convert_to_tensor(self.done[idx]),
        )

    def __len__(self):
        """
        Makes the buffer compatible with len() to return the current number of stored experience tuples.

        Returns:
            len (int): The current size of the buffer (number of stored experience tuples).
        """
        return self.size


def build_actor_model(state_dim, action_dim):
    """
    Builds the actor neural network model that maps states to actions.

    The actor model learns a deterministic policy that outputs the best action for a given state.

    Parameters:
        state_dim (int):    The dimension of the input state space.
        action_dim (int):   The dimension of the output action space.
    Returns:
        model (tf.keras.Model): A Keras model representing the actor network, which takes a state as input and outputs an action.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(state_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(action_dim, activation="tanh"),
        ],
        name="actor",
    )


def build_critic_model(state_dim, action_dim):
    """
    Builds the critic neural network model that maps state-action pairs to Q-values.

    The critic model learns to estimate the expected return (Q-value) of taking a certain action in a given state.
    Means how good is this action in this state?

    Parameters:
        state_dim (int):    The dimension of the input state space.
        action_dim (int):   The dimension of the output action space.
    Returns:
        model (tf.keras.Model): A Keras model representing the critic network, which takes a state and action as input and outputs a Q-value.
    """
    state_input = tf.keras.layers.Input(shape=(state_dim,))
    action_input = tf.keras.layers.Input(shape=(action_dim,))
    x = tf.keras.layers.Concatenate()([state_input, action_input])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    q = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model([state_input, action_input], q, name="critic")


class GaussianNoise:
    """
    The GaussianNoise adds random noise to the actions to enable exploration in deterministic policies like DDPG.
    """

    def __init__(self, action_dim, sigma_start, sigma_end, decay_steps):
        """
        Initializes the Gaussian noise generator with parameters for decaying the noise over time.

        Parameters:
            action_dim (int):       The dimension of the action space, which determines the shape of the noise vector.
            sigma_start (float):    The initial standard deviation of the noise at the beginning of training.
            sigma_end (float):      The final standard deviation of the noise after decay_steps have been reached.
            decay_steps (int):      The number of training steps over which the noise will decay from sigma_start to sigma_end.
        """
        self.action_dim = action_dim
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_steps = decay_steps

    def sigma(self, step):
        """
        Computes the current standard deviation of the noise based on the training step, decaying from sigma_start to
        sigma_end over decay_steps.

        Parameters:
            step (int): The current training step, used to calculate the decayed noise level.
        Returns:
            sigma (float): The current standard deviation of the noise to be applied to the actions.
        """
        t = min(1.0, step / float(self.decay_steps))
        return (1 - t) * self.sigma_start + t * self.sigma_end

    def sample(self, step):
        """
        Generates a noise sample from a Gaussian distribution with mean 0 and standard deviation computed by the sigma method.

        Parameters:
            step (int): The current training step, used to calculate the decayed noise level for sampling.
        Returns:
            noise (np.ndarray): A noise vector sampled from the Gaussian distribution, which can be added to the action for exploration.
        """
        sigma = self.sigma(step)
        return np.random.normal(loc=0.0, scale=sigma, size=(self.action_dim,)).astype(np.float32)


class DDPGAgent:
    """
    The DDPGAgent implements DDPG using an actor–critic architecture with target networks and experience replay.
    """

    def __init__(self, state_dim, action_dim, gamma=GAMMA, tau=TAU):
        """
        Initializes the DDPG agent by creating the actor and critic networks, their target counterparts, and the optimizers.

        Parameters:
            state_dim (int):    The dimension of the state space.
            action_dim (int):   The dimension of the action space.
            gamma (float):      The discount factor for future rewards.
            tau (float):        The soft update coefficient for updating target networks.
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Create actor and critic networks along with their target networks
        self.actor = build_actor_model(state_dim, action_dim)
        self.critic = build_critic_model(state_dim, action_dim)

        # Target networks are initialized with the same weights as the main networks
        self.target_actor = build_actor_model(state_dim, action_dim)
        self.target_critic = build_critic_model(state_dim, action_dim)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Create optimizers for the actor and critic networks
        self.actor_opt = tf.keras.optimizers.Adam(LR_ACTOR)
        self.critic_opt = tf.keras.optimizers.Adam(LR_CRITIC)

    @tf.function
    def train_step(self, states, actions, rewards, next_states, done_flags):
        """
        Performs a single training step for both the critic and actor networks, and updates the target networks.

        Parameters:
            states (tf.Tensor):       A batch of states from the replay buffer.
            actions (tf.Tensor):      A batch of actions taken by the agent corresponding to the states.
            rewards (tf.Tensor):      A batch of rewards received after taking the actions in the states.
            next_states (tf.Tensor):  A batch of next states resulting from taking the actions in the states.
            done_flags (tf.Tensor):   A batch of boolean flags indicating whether each transition resulted in a terminal state.
        Returns:
            actor_loss (tf.Tensor):   The computed loss for the actor network after the training step.
            critic_loss (tf.Tensor):  The computed loss for the critic network after the training step

        """

        # Updates the critic network by minimizing the mean squared error between the predicted Q-values and the target Q-values
        with tf.GradientTape() as tape:
            next_actions = self.target_actor(next_states)
            target_q = rewards + self.gamma * (1.0 - done_flags) * self.target_critic([next_states, next_actions])
            target_q = tf.clip_by_value(target_q, TARGET_Q_CLIP_MIN, TARGET_Q_CLIP_MAX)
            q = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(q - tf.stop_gradient(target_q)))

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # Updates the actor network by maximizing the expected Q-values predicted by the critic for the actions output by the actor
        with tf.GradientTape() as tape:
            predicted_actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, predicted_actions]))

        # Compute gradients for the actor network and apply them using the optimizer
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # After updating the main networks, perform a soft update of the target networks to slowly track the learned parameters
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return actor_loss, critic_loss

    @tf.function
    def soft_update(self, network, target_network):
        """
        Performs a soft update of the target network's weights towards the main network's weights using the factor tau.

        Parameters:
            network (tf.keras.Model):        The main network whose weights are used for the update.
            target_network (tf.keras.Model): The target network to be updated.
        """
        for weight, target_network_weight in zip(network.weights, target_network.weights):
            target_network_weight.assign(self.tau * weight + (1.0 - self.tau) * target_network_weight)

    def act(self, state_np):
        """
        Returns the action for a given state by passing it through the actor network.

        Parameters:
            state_np (np.ndarray): The current state of the environment as a numpy array.
        Returns:
            action (np.ndarray):   The action predicted by the actor network for the given state.
        """
        state = tf.convert_to_tensor(state_np.reshape(1, -1), dtype=tf.float32)
        action = self.actor(state)[0].numpy()
        return action

    def save(self, path):
        """
        Saves the actor and critic models along with their target networks to the specified directory.

        Parameters:
            path (str): The directory path where the models will be saved.
        """
        os.makedirs(path, exist_ok=True)
        self.actor.save(os.path.join(path, ACTOR_MODEL_FILE))
        self.critic.save(os.path.join(path, CRITIC_MODEL_FILE))
        self.target_actor.save(os.path.join(path, TARGET_ACTOR_MODEL_FILE))
        self.target_critic.save(os.path.join(path, TARGET_CRITIC_MODEL_FILE))


def evaluate(env, agent, episodes=EVAL_EPISODES):
    """
    Evaluates the agent's performance over a specified number of episodes in the given environment.

    Parameters:
        env (gym.Env):          The environment in which to evaluate the agent.
        agent (DDPGAgent):      The DDPG agent to be evaluated.
        episodes (int):         The number of episodes to run for evaluation.
    Returns:
        avg_return (float):         The average return (total reward) over the evaluation episodes.
        number_of_successes (int):  The number of successful parking episodes.
        number_of_crashes (int):    The number of episodes that ended in a crash.
    """
    returns = []
    number_of_successes = 0
    number_of_crashes = 0

    for _ in range(episodes):
        current_observation, _ = env.reset()
        done = False
        episode_return = 0.0  # total reward for this episode

        final_info = None
        while not done:
            # Get action from the agent and step the environment
            action = agent.act(current_observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            final_info = info
            done = terminated or truncated
            episode_return += float(reward)
            current_observation = next_observation

        # Check info for success or crash at the end of the episode
        if final_info.get("parked", False):
            number_of_successes += 1
        if final_info.get("crash", False):
            number_of_crashes += 1

        returns.append(episode_return)

    return float(np.mean(returns)), number_of_successes, number_of_crashes


def select_action(agent, env, current_observation, current_step, noise):
    """
    Selects an action for the given observation using the agent's policy, adding exploration noise during training.
    During the initial warmup phase, it samples random actions from the environment's action space to encourage exploration.

    Parameters:
        agent (DDPGAgent):                  The DDPG agent used to select actions based on the current observation.
        env (gym.Env):                      The environment, used to sample random actions during the warmup phase.
        current_observation (np.ndarray):   The current state observation from the environment.
        current_step (int):                 The current training step to determine whether to use random actions.
        noise (GaussianNoise):              The noise generator used to add exploration noise to the actions.
    Returns:
        action (np.ndarray): The action selected for the current observation.
    """
    if current_step <= RANDOM_WARMUP_STEPS:
        return env.action_space.sample().astype(np.float32)
    action = agent.act(current_observation) + noise.sample(current_step)
    return np.clip(action, ACTION_LOW, ACTION_HIGH).astype(np.float32)


def train_agent(agent, buffer, step):
    """
    Trains the DDPG agent by sampling a batch of experience from the replay buffer and performing a training step.

    Parameters:
        agent (DDPGAgent):      The DDPG agent to be trained.
        buffer (ReplayBuffer):  The replay buffer containing experience tuples for training.
        step (int):             The current training step.
    """
    # Only start training after we have enough experience in the buffer and after the warmup phase
    if len(buffer) < BATCH_SIZE or step <= RANDOM_WARMUP_STEPS:
        return

    for _ in range(UPDATES_PER_STEP):
        state, action, reward, new_state, done = buffer.sample(BATCH_SIZE)
        agent.train_step(state, action, reward, new_state, done)


def evaluate_and_save(step, eval_env, agent, best_success_count, best_avg_return, success_streak):
    """
    Evaluates the agent's performance and saves the model if it has improved.

    Parameters:
        step (int):                 The current training step at which evaluation is being performed.
        eval_env (gym.Env):         The environment used for evaluation.
        agent (DDPGAgent):          The DDPG agent being evaluated.
        best_success_count (int):   The best number of successes achieved in previous evaluations.
        best_avg_return (float):    The best average return achieved in previous evaluations.
        success_streak (int):       The current count of consecutive successful evaluations.

    Returns:
        best_success_count (int):   Updated best number of successes after evaluation.
        best_avg_return (float):    Updated best average return after evaluation.
        success_streak (int):       Updated success streak count after evaluation.
    """
    avg_return, successes, crashes = evaluate(eval_env, agent)
    print(
        f"[EVAL] step={step:7d}  "
        f"avg_return={avg_return:7.2f}  "
        f"success={successes}/{EVAL_EPISODES}  "
        f"crash={crashes}/{EVAL_EPISODES}"
    )

    if successes > best_success_count or (successes == best_success_count and avg_return > best_avg_return):
        best_success_count = successes
        best_avg_return = avg_return
        agent.save(MODEL_DIR_BEST)
        print(
            f"[BEST SAVE] step={step:7d} saved best model with "
            f"success={best_success_count}/{EVAL_EPISODES}  "
            f"avg_return={best_avg_return:.2f}"
        )

    if successes == EVAL_EPISODES:
        success_streak += 1
    else:
        success_streak = 0

    return best_success_count, best_avg_return, success_streak


def main():
    # Define the environment and evaluation environment to keep them separate (no noise during evaluation)
    env = ParkingGymEnv()
    eval_env = ParkingGymEnv()

    # Get the dimensions of the state and action spaces from the environment to initialize the agent and replay buffer
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # Initialize the DDPG agent, replay buffer, and noise process for exploration
    agent = DDPGAgent(state_dim, action_dim)
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, state_dim, action_dim)
    noise = GaussianNoise(
        action_dim=action_dim,
        sigma_start=NOISE_SIGMA_START,
        sigma_end=NOISE_SIGMA_END,
        decay_steps=NOISE_DECAY_STEPS_TOTAL,
    )

    # Initialize variables to track the best performance for saving the best model
    best_success_count = -1
    best_avg_return = -1e9
    success_streak = 0

    # Pre-define observation and episode length
    current_observation, _ = env.reset()
    current_episode_length = 0

    # Set the start time for logging
    start_time = time.time()
    step_start_time = start_time

    try:
        for step in range(1, TRAIN_TOTAL_STEPS + 1):
            # Select an action for the current observation
            action = select_action(agent, env, current_observation, step, noise)

            # Do a step in the environment using the selected action and observe the new state
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store the transition in the replay buffer for training
            buffer.add(current_observation, action, reward, new_observation, terminated)
            current_observation = new_observation
            current_episode_length += 1

            # Train the agent
            if step % TRAIN_EVERY == 0:
                train_agent(agent, buffer, step)

            # If the episode is done, or we have reached the maximum episode length, reset the environment to start a new episode
            if done or current_episode_length >= env.max_steps:
                current_observation, _ = env.reset()
                current_episode_length = 0

            # Log training progress
            if step % LOG_EVERY == 0:
                dt = time.time() - step_start_time
                print(f"step={step:7d}  buffer={len(buffer):6d}  sigma={noise.sigma(step):.3f}  time={dt:.1f}s")
                step_start_time = time.time()

            # Evaluate and save the model
            if step % EVAL_EVERY == 0:
                (
                    best_success_count,
                    best_avg_return,
                    success_streak,
                ) = evaluate_and_save(
                    step,
                    eval_env,
                    agent,
                    best_success_count,
                    best_avg_return,
                    success_streak,
                )

                if success_streak >= EARLY_STOPPING_PATIENCE:
                    break

            # Save the latest model at regular intervals regardless of performance improvements
            if step % SAVE_EVERY == 0:
                agent.save(MODEL_DIR_LATEST)
                print(f"[SAVE] step={step:7d} saved latest model")

    finally:
        agent.save(MODEL_DIR_LATEST)
        print(f"[END] Training ended after {time.time() - start_time:.1f}s")
        print("[FINAL SAVE] Saved final model")
        print(f"[FINAL BEST] success={best_success_count}/{EVAL_EPISODES} avg_return={best_avg_return:.2f}")


if __name__ == "__main__":
    main()
