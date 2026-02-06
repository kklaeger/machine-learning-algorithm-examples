import numpy as np

# States
OK = 0
WARNING = 1
CRITICAL = 2
FAILED = 3

# Actions
DO_NOTHING = 0
INSPECT = 1
REPAIR = 2

# Rewards
DO_NOTHING_REWARD = 2
INSPECT_REWARD = -0.05
REPAIR_REWARD = -6
FAILURE_PENALTY = -10

# Transition probabilities
OK_TO_WARNING_PROB = 0.3
WARNING_TO_FAILED_PROB = 0.2
WARNING_TO_CRITICAL_PROB = 0.65
WARNING_TO_CRITICAL_PROB_IF_INSPECT = 0.05
CRITICAL_TO_FAILED_PROB = 0.5
CRITICAL_TO_FAILED_PROB_IF_INSPECT = 0.2


class MachineMaintenanceEnv:
    """
    A simple environment simulating machine maintenance with four states and three actions.
    """

    def __init__(self, seed):
        """
        Initializes the environment.

        Parameters:
            seed (int): Seed for the random number generator to ensure reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self):
        """
        Resets the environment to the initial state (OK).

        Returns:
            state (int): The initial state of the machine (OK).
        """
        self.state = OK
        return self.state

    def step(self, action):
        """
        Takes an action and updates the environment state accordingly.

        Parameters:
            action (int): The action to take (DO_NOTHING, INSPECT, REPAIR).
        Returns:
            state (int):        The new state of the machine after taking the action.
            reward (float):     The reward received after taking the action.
            terminated (bool):  Whether the episode has ended (machine failed).
        """
        reward = 0
        terminated = False

        # Reward for action taken
        if action == DO_NOTHING:
            reward = DO_NOTHING_REWARD
        elif action == INSPECT:
            reward = INSPECT_REWARD
        elif action == REPAIR:
            reward = REPAIR_REWARD

        if action == INSPECT and self.state == WARNING:
            reward += 0.2  # Bonus for inspecting when in WARNING state

        # Repair action always restores the machine to OK state
        if action == REPAIR:
            self.state = OK
        else:
            risk_sample = self.rng.random()

            # If state is OK, there is a 30% chance to degrade to WARNING
            # Action INSPECT has no effect in this case
            if self.state == OK and risk_sample < OK_TO_WARNING_PROB:
                self.state = WARNING

            # If state is WARNING and action is DO_NOTHING, there is a 60% chance to degrade to CRITICAL and 10% chance to degrade to FAILED
            # If action is INSPECT, the chance to degrade is reduced to 20%
            elif self.state == WARNING:
                if action == DO_NOTHING and risk_sample < WARNING_TO_FAILED_PROB:
                    self.state = FAILED
                elif action == DO_NOTHING and risk_sample < WARNING_TO_CRITICAL_PROB:
                    self.state = CRITICAL
                elif action == INSPECT and risk_sample < WARNING_TO_CRITICAL_PROB_IF_INSPECT:
                    self.state = CRITICAL

            # If state is CRITICAL and action is DO_NOTHING, there is a 50% chance to fail
            # If action is INSPECT, the chance to fail is reduced to 20%
            elif self.state == CRITICAL:
                if action == DO_NOTHING and risk_sample < CRITICAL_TO_FAILED_PROB:
                    self.state = FAILED
                elif action == INSPECT and risk_sample < CRITICAL_TO_FAILED_PROB_IF_INSPECT:
                    self.state = FAILED

        # Failure is a terminal state with a penalty
        if self.state == FAILED:
            reward = FAILURE_PENALTY
            terminated = True

        return self.state, reward, terminated
