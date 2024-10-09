from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import numpy as np
from enum import Enum
from goal_selection_strategy import GoalSelectionStrategy

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

    def close_env(self):
        self.env.close()

    def set_env(self, env: VecEnv) -> None:
        """
        Sets the environment.

        :param env:
        """
        # fixme: Setting an environment with model.set_env() should also set the environment for her buffer.
        #  Currently the old env remains in the buffer, which leads to BrokenPipeErrors when the environment is closed.
        # if self.env is not None:
        #    raise ValueError("Trying to set env of already initialized environment.")

        self.env = env
