import math
import time
import tkinter as tk
import numpy as np

from config import *


class ParkingGUI:
    """Compact GUI: manual driving, optional actor playback (no gym wrapper)."""

    def __init__(self, root, env, actor_model=None):
        # Initialize the GUI with the given root, environment, and optional actor model.
        self.root = root
        self.env = env
        self.actor = actor_model

        # Define the time step duration for the simulation (in seconds)
        self.tick_dt = TIME_STEP_DURATION

        # Set the dimensions based on the environment's attributes
        self.width = env.width
        self.height = env.height

        # Set manual and autopilot control states
        self.keys = set()
        self.autopilot = actor_model is not None

        # Track whether the episode has ended (e.g., parked successfully)
        self._episode_done = False

        # Define the window title and create a canvas for drawing
        root.title("Parking Demo")
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        # IDs for drawn elements (car and status text)
        self.car_id = None
        self.status_id = None

        # Reset the environment and initialize the GUI elements
        self.env.reset()
        self.env._crashed = False

        # Draw the static elements of the environment (road, parking area) and bind keyboard inputs
        self._draw_static()
        self._bind_keys()

        # Start the main loop for updating the GUI and processing inputs
        self.last_time = time.time()
        self._tick()

    def _draw_static(self):
        """
        Draw the static elements of the environment, such as the road and parking area.
        """
        c = self.canvas
        c.delete("all")

        # Draw the outer boundary of the environment
        c.create_rectangle(0, 0, self.env.width, self.env.height, outline=COLOR_BORDER, width=BORDER_WIDTH)

        # Draw the road area
        c.create_rectangle(self.env.road_x0, self.env.road_y0, self.env.road_x1, self.env.road_y1, fill=COLOR_ROAD,
                           outline="")

        # Draw the line between the road and parking area
        c.create_line(self.env.road_x0, self.env.road_y0, self.env.road_x1, self.env.road_y0, width=CURB_WIDTH)

        left_x1 = self.env.park_x0 - PARKED_CAR_MARGIN
        right_x0 = self.env.park_x1 + PARKED_CAR_MARGIN

        # Draw the obstacles (gray areas) on the left and right of the parking area
        c.create_rectangle(LEFT_BLOCK_X0, self.env.park_y0, left_x1, self.env.park_y1, fill=COLOR_PARKED_CAR,
                           outline="")
        c.create_rectangle(right_x0, self.env.park_y0, RIGHT_BLOCK_X1, self.env.park_y1, fill=COLOR_PARKED_CAR,
                           outline="")

        # Draw the parking area (white rectangle with black border)
        c.create_rectangle(self.env.park_x0, self.env.park_y0, self.env.park_x1, self.env.park_y1,
                           outline=COLOR_PARK_SLOT, width=BORDER_WIDTH)

    def _draw_car(self):
        """
        Draw the car on the canvas based on its current position and orientation. If the car has crashed, use a different color.
        """
        car_corners = []
        for (cx, cy) in self.env.calculate_car_corners():
            car_corners.extend([cx, cy])

        if self.car_id:
            self.canvas.delete(self.car_id)
        self.car_id = self.canvas.create_polygon(car_corners, fill=COLOR_CAR, outline=COLOR_PARK_SLOT,
                                                 width=BORDER_WIDTH)

    def _bind_keys(self):
        """
        Bind keyboard events for controlling the car.
        """
        self.root.bind("<KeyPress>", self._on_keypress)
        self.root.bind("<KeyRelease>", self._on_keyrelease)
        self.canvas.focus_set()

    def _on_keypress(self, event):
        """
        Handle key press events.

        Parameters:
            event (tkinter.Event): The event object containing information about the key press.
        """
        keysym = event.keysym
        if event.keysym in KEYMAP_ARROWS:
            self.keys.add(KEYMAP_ARROWS[keysym])
            return
        if event.keysym == RESET_KEY:
            self._reset()
        if event.keysym == ESCAPE_KEY:
            self.root.quit()

    def _on_keyrelease(self, event):
        """
        Handle key release events.

        Parameters:
            event (tkinter.Event): The event object containing information about the key release.
        """
        keysym = event.keysym
        if keysym in KEYMAP_ARROWS and KEYMAP_ARROWS[keysym] in self.keys:
            self.keys.discard(KEYMAP_ARROWS[keysym])

    def _reset(self):
        """
        Reset the environment and GUI to the initial state.
        """
        self.env.reset()
        self._episode_done = False
        self.canvas.delete("info_text")
        self.env._crashed = False

    def _manual_action(self):
        """
        Compute the throttle and steering inputs based on the current keys pressed.

        Returns:
            throttle (float):   The computed throttle value, ranging from -1.0 to 1.0.
            steer (float):      The computed steering value, ranging from -1.0 to 1
        """
        throttle = (1.0 if "UP" in self.keys else 0.0) + (-1.0 if "DOWN" in self.keys else 0.0)
        throttle = max(-1.0, min(1.0, throttle))
        steer = (-1.0 if "LEFT" in self.keys else 0.0) + (1.0 if "RIGHT" in self.keys else 0.0)
        steer = max(-1.0, min(1.0, steer))
        return throttle, steer

    def _obs_for_actor(self):
        """
        Prepare the observation for the actor model by normalizing the state variables, because the actor is expected
        to take normalized inputs.

        Returns:
            obs (numpy.ndarray): A 1D array containing the normalized state variables [x, y, sin(theta), cos(theta), v, s].
        """
        x, y, t, v, s = self.env.get_state()
        return np.array([x / self.env.width, y / self.env.height, math.sin(t), math.cos(t),
                         v / max(1e-6, MAX_SPEED), s / max(1e-6, MAX_STEER)], dtype=np.float32)

    def _autopilot_action(self):
        """
        Asks the trained actor model for the next action based on the current observation.

        Returns:
            throttle (float): The throttle value predicted by the actor model, clipped to the range [-1.0, 1.0].
            steer (float):    The steering value predicted by the actor model, clipped to the range [-1.0, 1.0].
        """
        obs = self._obs_for_actor().reshape(1, -1).astype(np.float32)
        action = self.actor(obs).numpy()[0]
        action = np.clip(action, -1.0, 1.0)
        return float(action[0]), float(action[1])

    def _tick(self):
        """
        Main loop for updating the GUI and processing inputs. This method is called periodically based on the defined tick duration.
        """
        self.last_time = time.time()

        if self.env.is_crashed() or self._episode_done:
            self._draw_car()
            self.root.after(int(self.tick_dt * 1000), self._tick)
            return

        if self.autopilot and self.actor is not None:
            throttle, steer = self._autopilot_action()
        else:
            throttle, steer = self._manual_action()

        state, reward, done, info= self.env.step(throttle, steer)

        if done and info.get("parked", False):
            self._episode_done = True
            self.canvas.delete("info_text")
            self.canvas.create_text(self.width // 2, 40, text="PARKED ✓", font=("Arial", 22), fill="green",
                                    tag="info_text")

        if info.get("crash", False):
            self.canvas.delete("info_text")
            self.canvas.create_text(self.width // 2, 40, text="CRASHED ✗", font=("Arial", 22), fill="red",
                                    tag="info_text")

        self._draw_car()

        self.root.after(int(self.tick_dt * 1000), self._tick)
