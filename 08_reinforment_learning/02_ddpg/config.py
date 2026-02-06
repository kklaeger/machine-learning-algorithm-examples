import numpy as np

# =============================================================================
# Simulation timing
# =============================================================================

TIME_STEP_DURATION = 0.05

# =============================================================================
# World geometry (pixels)
# =============================================================================

WORLD_WIDTH = 900
WORLD_HEIGHT = 400

CANVAS_MARGIN = 5

# =============================================================================
# Road geometry (pixels)
# =============================================================================

ROAD_X0 = 50
ROAD_X1 = 850
ROAD_Y0 = 220
ROAD_Y1 = 330

# =============================================================================
# Parking slot geometry (pixels)
# =============================================================================

PARK_SLOT_X0 = 320
PARK_SLOT_X1 = 460
PARK_SLOT_Y0 = 170
PARK_SLOT_Y1 = 220

PARK_SLOT_MARGIN = 20

# =============================================================================
# Car geometry (pixels)
# =============================================================================

CAR_LENGTH = 70.0
CAR_WIDTH = 35.0

# =============================================================================
# Car dynamics / kinematics
# =============================================================================

MAX_SPEED = 1.8
ACCELERATION = 0.30

SPEED_DAMP = 0.97  # per-step velocity damping
SPEED_SCALE = 90.0  # pixels per second scale (model speed -> px/s)

MAX_STEER = 1.2
STEER_RATE = 1.6
STEER_DAMP = 0.88  # per-step steering damping

TURN_GAIN = 1.8  # turn sensitivity
MIN_TURN_SPEED = 0.04  # below this, use stationary steering help
LOW_SPEED_TURN_GAIN = 2.8

# =============================================================================
# Initial state
# =============================================================================

START_X = 200.0
START_Y = 260.0
START_THETA = 0.0
START_V = 0.0
START_STEER = 0.0

# =============================================================================
# Rewards
# =============================================================================

REWARD_CRASH = -100.0
REWARD_PARKED = 100.0
REWARD_STEP = -1.0

CENTER_DISTANCE_WEIGHT = 0.05
ORIENTATION_WEIGHT = 0.5

PARALLEL_WEIGHT = 0.1
PARALLEL_DISTANCE_THRESHOLD = 20.0 # px

# =============================================================================
# Parking success tolerances
# =============================================================================

PARK_MAX_ANGLE_DEG = 10.0

# =============================================================================
# GUI colors
# =============================================================================

COLOR_BG = "white"
COLOR_BORDER = "#333"
COLOR_ROAD = "#f2f2f2"
COLOR_PARKED_CAR = "#cccccc"
COLOR_CAR = "#2563eb"
COLOR_PARK_SLOT = "black"

# =============================================================================
# GUI layout
# =============================================================================

BORDER_WIDTH = 2
CURB_WIDTH = 6

PARKED_CAR_MARGIN = 20
LEFT_BLOCK_X0 = 60
RIGHT_BLOCK_X1 = 810

# =============================================================================
# Manual control (keys)
# =============================================================================

KEYMAP_ARROWS = {"Up": "UP", "Down": "DOWN", "Left": "LEFT", "Right": "RIGHT"}
RESET_KEY = "r"
ESCAPE_KEY = "Escape"

# =============================================================================
# Gym wrapper
# =============================================================================

EPISODE_MAX_STEPS = 400

OBSERVATION_LOW = np.array([0, 0, -1, -1, -1, -1], dtype=np.float32)
OBSERVATION_HIGH = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)

ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# =============================================================================
# Training configuration
# =============================================================================

SEED = 42

REPLAY_BUFFER_SIZE = 200_000

# Exploration noise
NOISE_SIGMA_START = 0.30
NOISE_SIGMA_END = 0.10
NOISE_DECAY_STEPS_TOTAL = 150_000

# DDPG hyperparameters
GAMMA = 0.99
TAU = 0.005

LR_ACTOR = 1e-3
LR_CRITIC = 1e-3

# Q target clipping (stability)
TARGET_Q_CLIP_MIN = -300.0
TARGET_Q_CLIP_MAX = 300.0

# =============================================================================
# Evaluation & training loop
# =============================================================================

EVAL_EPISODES = 5

TRAIN_TOTAL_STEPS = 350_000
RANDOM_WARMUP_STEPS = 5_000

BATCH_SIZE = 256
TRAIN_EVERY = 1
UPDATES_PER_STEP = 1

LOG_EVERY = 2_000
EVAL_EVERY = 10_000
SAVE_EVERY = 50_000

EARLY_STOPPING_PATIENCE = 3  # number of perfect evals in a row

# =============================================================================
# Model saving paths / filenames
# =============================================================================

MODEL_DIR_LATEST = "models/ddpg_parking/latest"
MODEL_DIR_BEST = "models/ddpg_parking/best"

ACTOR_MODEL_FILE = "actor.keras"
CRITIC_MODEL_FILE = "critic.keras"
TARGET_ACTOR_MODEL_FILE = "target_actor.keras"
TARGET_CRITIC_MODEL_FILE = "target_critic.keras"
