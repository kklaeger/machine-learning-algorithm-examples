import tkinter as tk
from pathlib import Path

import tensorflow as tf

from gui import ParkingGUI
from gym_env import ParkingGymEnv
from config import MODEL_DIR_LATEST, MODEL_DIR_BEST, ACTOR_MODEL_FILE


def resolve_actor_model_path():
    """
    Resolves the path to the actor model file. If a "best" model exists, it returns that path; otherwise, it returns
    the path to the latest model.

    Returns:
        model_path (Path): Path to the actor model file.
    """
    best_path = Path(MODEL_DIR_BEST) / ACTOR_MODEL_FILE
    latest_path = Path(MODEL_DIR_LATEST) / ACTOR_MODEL_FILE

    if best_path.exists():
        print("Found BEST model, loading:", best_path)
        return best_path

    if latest_path.exists():
        print("No BEST model found, loading LATEST model:", latest_path)
        return latest_path

    raise FileNotFoundError(f"No model found. Checked paths: {best_path} and {latest_path}")


def main():
    # Load the model
    model_path = resolve_actor_model_path()
    actor = tf.keras.models.load_model(model_path)

    # Use the underlying ParkingEnv for the GUI
    gym_env = ParkingGymEnv()
    env = gym_env.env
    env.reset()

    # Create the GUI and start the main loop
    root = tk.Tk()
    ParkingGUI(root, env, actor_model=actor)
    label = tk.Label(root, justify="left", font=("Arial", 10))
    label.pack(side="bottom", pady=6)

    root.mainloop()


if __name__ == "__main__":
    main()
