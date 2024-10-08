from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import numpy as np
from enum import Enum

class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """

    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select a goal that was achieved
    # after the current step, in the same episode, with the exception of the final state
    FUTURE_EXCEPT_FINAL = 3
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    "future": GoalSelectionStrategy.FUTURE,
    "future_except_final": GoalSelectionStrategy.FUTURE_EXCEPT_FINAL,
    "final": GoalSelectionStrategy.FINAL,
    "episode": GoalSelectionStrategy.EPISODE,
}

class CustomHerReplayBuffer(HerReplayBuffer):
    def ___init___(self,
                   *args,
                   **kwargs):
        super().__init__(*args, **kwargs)

    def _sample_goals(self, batch_indices: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the environments
        :return: Sampled goals
        """
        # todo: Don't sample goals from failed states
        batch_ep_start = self.ep_start[batch_indices, env_indices]
        batch_ep_length = self.ep_length[batch_indices, env_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Replay with final state of current episode
            transition_indices_in_episode = batch_ep_length - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Replay with random state which comes from the same episode and was observed after current transition
            # Note: our implementation is inclusive: current transition can be sampled
            current_indices_in_episode = (batch_indices - batch_ep_start) % self.buffer_size
            transition_indices_in_episode = np.random.randint(current_indices_in_episode, batch_ep_length)

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE_EXCEPT_FINAL:
            current_indices_in_episode = (batch_indices - batch_ep_start) % self.buffer_size
            transition_indices_in_episode = np.random.randint(current_indices_in_episode, batch_ep_length - 1)

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Replay with random state which comes from the same episode as current transition
            transition_indices_in_episode = np.random.randint(0, batch_ep_length)

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        transition_indices = (transition_indices_in_episode + batch_ep_start) % self.buffer_size
        return self.next_observations["achieved_goal"][transition_indices, env_indices]
