import math
import random
import numpy as np

from config import *


class ParkingEnv:
    """
    A simple 2D parking environment for reinforcement learning. The environment simulates a car that can be controlled
    with throttle and steering inputs, and the goal is to park the car in a designated parking slot without crashing.
    """

    def __init__(self, width=WORLD_WIDTH, height=WORLD_HEIGHT):
        """
        Initializes the parking environment with specified dimensions and sets up the initial state of the car.

        Parameters:
            width (int):    The width of the environment canvas in pixels. Default is WORLD_WIDTH.
            height (int):   The height of the environment canvas in pixels. Default is WORLD_HEIGHT.
        """
        # Canvas dimensions
        self.width = width
        self.height = height

        # Parking slot rectangle (target area)
        self.park_x0, self.park_x1 = PARK_SLOT_X0, PARK_SLOT_X1
        self.park_y0, self.park_y1 = PARK_SLOT_Y0, PARK_SLOT_Y1
        self.slot_margin = PARK_SLOT_MARGIN

        # Road rectangle (drivable area)
        self.road_x0, self.road_x1 = ROAD_X0, ROAD_X1
        self.road_y0, self.road_y1 = ROAD_Y0, ROAD_Y1

        # Canvas boundaries for clamping (with some margin to avoid corner clipping)
        self.min_x, self.max_x = CANVAS_MARGIN, width - CANVAS_MARGIN
        self.min_y, self.max_y = CANVAS_MARGIN, height - CANVAS_MARGIN

        # Car dimensions (for collision and rendering)
        self.length = CAR_LENGTH
        self.car_width = CAR_WIDTH

        # State variables
        self.x = START_X
        self.y = START_Y
        self.theta = START_THETA
        self.v = START_V
        self.steer = START_STEER
        self._crashed = False

    def reset(self, seed=None):
        """
        Resets the environment to the initial state. Optionally sets the random seed for reproducibility.
        Parameters:
            seed (int | None): Optional random seed for reproducibility. Default is None.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.x = START_X
        self.y = START_Y
        self.theta = START_THETA
        self.v = START_V
        self.steer = START_STEER
        self._crashed = False

    def get_state(self):
        """
        Returns the current state of the car as a tuple (x, y, theta, v, steer).

        Returns:
            state (tuple): The current state variables of the car.
        """
        return self.x, self.y, self.theta, self.v, self.steer

    def calculate_car_corners(self):
        """
        Calculates the world coordinates of the car's four corners based on its center (x, y) and orientation (theta).

        Returns:
           world (list of tuples): A list of four tuples, each representing the (x, y) coordinates of a corner of the
           car in the world frame.
        """
        x, y, theta = self.x, self.y, self.theta

        car_length = self.length
        car_width = self.car_width

        vehicle_frame = [(-car_length / 2, -car_width / 2), (car_length / 2, -car_width / 2),
                         (car_length / 2, car_width / 2), (-car_length / 2, car_width / 2)]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        world = []
        for px, py in vehicle_frame:
            wx = x + px * cos_t - py * sin_t
            wy = y + px * sin_t + py * cos_t
            world.append((wx, wy))
        return world

    def is_drivable(self):
        """
        Returns True only if all corners are inside the road or parking areas.

        Returns:
            drivable (bool): True, if all corners are inside the road or parking areas.
        """
        for (cx, cy) in self.calculate_car_corners():
            if not self._point_in_road_or_parking(cx, cy):
                return False
        return True

    def is_crashed(self):
        """
        Returns True if the car has crashed.

        Returns:
            crashed (bool): True, if the car has crashed, False otherwise.
        """
        return self._crashed

    def is_fully_parked(self):
        """
        Returns True only if all corners are inside the parking slot and the car is reasonably aligned (not too
        rotated).

        Returns:
            parked (bool): True, if all corners are inside the parking slot and the car's orientation is within the
            allowed parking angle.
        """
        for (cx, cy) in self.calculate_car_corners():
            if not (self.park_x0 <= cx <= self.park_x1 and self.park_y0 <= cy <= self.park_y1):
                return False

        return abs(math.sin(self.theta)) < math.sin(math.radians(PARK_MAX_ANGLE_DEG))

    def step(self, throttle, steering):
        """
        Updates the car's state based on throttle and steer inputs, checks for collisions and parking success, and
        returns the new state, reward, done flag, and info.

        Parameters:
            throttle (float): The throttle input value. Positive: accelerate forward, negative: accelerate backward, zero: maintain current speed.
            steering (float): The steering input value. Positive: steer right, negative: steer left, zero: maintain current steering angle.
        Returns:
            state (tuple):      The new state of the car.
            reward (float):     The reward for the current step.
            done (bool):        True, if the episode has ended (either by crashing or successfully parking), False otherwise.
            parked (bool):      True, if the car is successfully parked, False otherwise.
            crashed (bool):     True, if the car has crashed, False otherwise.
        """

        # If already crashed, no further updates - just return the crash state
        if self._crashed:
            return self.get_state(), REWARD_CRASH, True, {"crash": True}

        # Increase or decrease speed based on throttle input, with clamping to max speed
        if throttle > 0:
            self.v = min(MAX_SPEED, self.v + ACCELERATION * throttle)
        elif throttle < 0:
            self.v = max(-MAX_SPEED, self.v + ACCELERATION * throttle)

        # Update steering angle based on steering input, with clamping to max steer angle
        self.steer += steering * STEER_RATE * TIME_STEP_DURATION
        self.steer = max(-MAX_STEER, min(MAX_STEER, self.steer))

        # Update orientation (moving vs. near-stationary behavior)
        if abs(self.v) > MIN_TURN_SPEED:
            self.theta += TURN_GAIN * (self.v * self.steer) * TIME_STEP_DURATION
        else:
            self.theta += LOW_SPEED_TURN_GAIN * self.steer * TIME_STEP_DURATION

        # Move the car with velocity v and orientation theta for the time_step_duration
        self.x += math.cos(self.theta) * self.v * SPEED_SCALE * TIME_STEP_DURATION
        self.y += math.sin(self.theta) * self.v * SPEED_SCALE * TIME_STEP_DURATION

        # Damp the speed and steering to simulate friction
        self.v *= SPEED_DAMP
        self.steer *= STEER_DAMP

        # Clamp the car's position to stay within the canvas boundaries
        self.x = np.clip(self.x, self.min_x, self.max_x)
        self.y = np.clip(self.y, self.min_y, self.max_y)

        crashed = False
        parked = False
        done = False

        if not self.is_drivable():
            crashed = True
            done = True
            self._crashed = True
            reward = REWARD_CRASH

        elif self.is_fully_parked():
            parked = True
            done = True
            reward = REWARD_PARKED

        else:
            reward = REWARD_STEP

        info = {"crash": crashed, "parked": parked}

        return self.get_state(), reward, done, info

    def _point_in_road_or_parking(self, x, y):
        """
        Helper function to check if a point (x, y) is within the drivable road area or the parking slot
        area (with some margin for approach).

        Parameters:
            x (float): The x-coordinate of the point to check.
            y (float): The y-coordinate of the point to check.
        Returns:
            bool: True if the point is within the road area, parking slot area (with margin), or the approach area
            above the parking slot; False otherwise.

        """
        # Check if the point is within the road rectangle
        on_road = (self.road_x0 <= x <= self.road_x1) and (self.road_y0 <= y <= self.road_y1)

        # Check if the point is within the parking slot rectangle with some buffer margin
        m = self.slot_margin
        sx0, sx1 = self.park_x0 - m, self.park_x1 + m
        sy0, sy1 = self.park_y0 - m, self.park_y1 + m
        in_slot = (sx0 <= x <= sx1) and (sy0 <= y <= sy1)

        # Check if the point is in the approach area above the parking slot (same x range, y above the slot)
        approach_y0 = sy1
        approach_y1 = self.road_y0
        in_approach = (sx0 <= x <= sx1) and (approach_y0 <= y <= approach_y1)

        return on_road or in_slot or in_approach
