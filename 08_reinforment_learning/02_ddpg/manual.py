import tkinter as tk

from env import ParkingEnv
from gui import ParkingGUI


def main():
    env = ParkingEnv()

    canvas_w = int(env.width)
    canvas_h = int(env.height)
    help_h = 90  # space for the help label under canvas

    root = tk.Tk()
    root.title("Manual Parking (Keyboard)")

    # Set window size
    win_w = canvas_w
    win_h = canvas_h + help_h

    # Center the window on the screen
    root.update_idletasks()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    pos_x = max(0, (screen_w - win_w) // 2)
    pos_y = max(0, (screen_h - win_h) // 2)

    # Set the geometry and disable resizing
    root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
    root.resizable(False, False)

    # GUI (should bind keys internally, as in your current refactor)
    ParkingGUI(root, env)

    help_lines = [
        "Controls:",
        "Arrow keys - drive (hold for smooth control)",
        "R - reset, Space - toggle autopilot (if supported), Esc - quit",
        "C - toggle debug markers (if supported)",
    ]
    label = tk.Label(root, text="\n".join(help_lines), justify="left", font=("Arial", 10))
    label.pack(side="bottom", pady=6)

    # Always allow ESC to close, even if GUI bindings changed
    root.bind("<Escape>", lambda _e: root.destroy())

    root.mainloop()


if __name__ == "__main__":
    main()
