import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env import ParkingEnv
from config import *


class ParkingGymEnv(gym.Env):
    """
    Gym environment for the parking task.
    """

    def __init__(self):
        """
        Initialize the Gym environment.
        """
        super().__init__()
        self.env = ParkingEnv()
        self.steps = 0
        self.max_steps = EPISODE_MAX_STEPS

        self.observation_space = spaces.Box(
            low=OBSERVATION_LOW,
            high=OBSERVATION_HIGH,
            dtype=np.float32
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def get_and_normalize_observations(self):
        """
        Get and normalize the current observations from the environment.

        Returns:
            np.ndarray: Normalized observation array.
        """
        x, y, theta, v, steer = self.env.get_state()

        # Normalize observations to [0, 1] or [-1, 1]
        x_n = x / self.env.width
        y_n = y / self.env.height
        v_n = v / MAX_SPEED
        s_n = steer / MAX_STEER

        return np.array([x_n, y_n, math.sin(theta), math.cos(theta), v_n, s_n], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state and return the current observation of the environment.
        E.g. the position of the car.

        Parameters:
            seed (int):     Random seed for reproducibility.
            options (dict): Additional options for resetting the environment. Ignored in this implementation.
        Returns:
            np.ndarray: Initial normalized observation array.
        """
        super().reset(seed=seed)
        self.steps = 0
        self.env.reset(seed=seed)
        return self.get_and_normalize_observations(), {}

    def step(self, action):
        """
        Take a step in the environment using the given action.
        Parameters:
            action (np.ndarray): Action array containing acceleration and steering inputs.
        Returns:
            tuple: A tuple containing the next observation, reward, terminated flag, truncated flag, and info dictionary.
        """
        self.steps += 1

        acceleration = float(action[0])
        steering = float(action[1])

        # Core step in the underlying environment
        _, base_reward, done, info = self.env.step(acceleration, steering)

        # Get the distance from the car to the center of the parking slot
        parking_slot_center_x = (self.env.park_x0 + self.env.park_x1) / 2
        parking_slot_center_y = (self.env.park_y0 + self.env.park_y1) / 2
        distance_car_parking_slot = np.hypot(
            self.env.x - parking_slot_center_x,
            self.env.y - parking_slot_center_y
        )

        # Penalize distance from the parking slot center and misalignment with the parking slot
        center_penalty = -CENTER_DISTANCE_WEIGHT * distance_car_parking_slot

        # Penalize misalignment with the parking slot if close to it and not parked, but no penalty if already parked
        if info.get("parked", False):
            parallel_penalty = 0.0
        elif distance_car_parking_slot < 20.0:
            parallel_penalty = -PARALLEL_WEIGHT * abs(math.sin(self.env.theta))
        else:
            parallel_penalty = 0.0
        # Total reward is the base reward from the environment plus the additional penalties
        reward = base_reward + center_penalty + parallel_penalty

        observation = self.get_and_normalize_observations()
        terminated = done
        truncated = self.steps >= self.max_steps

        return observation, reward, terminated, truncated, info
