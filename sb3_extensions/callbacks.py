from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

import os

import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import sync_envs_normalization

try:
    import nvidia_smi
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False
import psutil
import time


class EvalSuccessCallback(EvalCallback):
    """
    Extension for EvalCallback that logs the success rate of the evaluation episodes.
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.eval_log_name = kwargs.pop("eval_log_name", "eval")
        if self.eval_log_name is None:
            self.eval_log_name = "eval"
        super().__init__(*args, **kwargs)

        self.best_mean_success_rate = 0.0

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"{self.eval_log_name}/mean_reward", float(mean_reward))
            self.logger.record(f"{self.eval_log_name}/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record(f"{self.eval_log_name}/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if success_rate > self.best_mean_success_rate:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_success_rate = success_rate
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class StopTrainingOnSuccessRateCallback(BaseCallback):
    def __init__(self, check_freq: int, success_threshold: float, window_size: int = 100, verbose=0):
        super(StopTrainingOnSuccessRateCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.success_threshold = success_threshold

    def _on_step(self) -> bool:
        if self.check_freq > 0 and self.n_calls % self.check_freq == 0:
            # Compute the success rate
            success_rate = self.compute_success_rate()

            if self.verbose > 0:
                print(f"Rollout {self.n_calls}: Success Rate: {success_rate}")

            # Stop training if the success rate is above the threshold
            if success_rate >= self.success_threshold:
                print(f"Success threshold reached: {success_rate}")
                return False

        return True

    def compute_success_rate(self):
        ep_success_buffer = self.locals['self'].ep_success_buffer
        success_rate = safe_mean(ep_success_buffer)

        return success_rate


class StopTrainingOnSuccessThreshold(BaseCallback):
    """
    Stop the training once a threshold in success rate
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalSuccessCallback``.

    :param success_threshold:  Minimum expected success in evaluation to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
        threshold reached
    """

    parent: EvalSuccessCallback

    def __init__(self, success_threshold: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.success_threshold = success_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used " "with an ``EvalCallback``"
        # Convert np.bool_ to bool, otherwise callback() is False won't work
        continue_training = bool(self.parent.best_mean_success_rate < self.success_threshold)
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because the success rate {self.parent.best_mean_success_rate:.2f} "
                f" is above the threshold {self.success_threshold}"
            )
        return continue_training


class RecordCustomMetricsCallback(BaseCallback):
    def __init__(self, metrics, verbose=0):
        super(RecordCustomMetricsCallback, self).__init__(verbose)
        self.metrics = metrics

    def _on_step(self) -> bool:
        for name, value in self.metrics.items():
            self.logger.record(name, value)

        return True


class ResourceMonitorCallback(BaseCallback):
    """
    Custom callback for monitoring CPU and GPU usage during training.
    """

    def __init__(self, print_freq=1000, verbose=1):
        """
        Parameters:
        -----------
        print_freq: int
            Number of timesteps between prints
        verbose: int
            Verbosity level
        """
        super().__init__(verbose)
        self.print_freq = print_freq

        # Initialize nvidia-smi if available
        if NVIDIA_SMI_AVAILABLE:
            nvidia_smi.nvmlInit()
            self.gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # GPU 0

        # Store history for plotting later if needed
        self.cpu_usages = []
        self.gpu_usages = []
        self.gpu_memories = []
        self.timestamps = []
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):

        # Get CPU usage
        cpu_percent = psutil.cpu_percent()

        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Get GPU stats if available
        if NVIDIA_SMI_AVAILABLE:
            gpu_info = nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            gpu_memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            gpu_util = gpu_info.gpu
            gpu_mem_used = gpu_memory.used / gpu_memory.total * 100
        else:
            gpu_util = None
            gpu_mem_used = None

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Store metrics
        self.cpu_usages.append(cpu_percent)
        self.gpu_usages.append(gpu_util if gpu_util is not None else 0)
        self.gpu_memories.append(gpu_mem_used if gpu_mem_used is not None else 0)
        self.timestamps.append(elapsed_time)


        # log metrics
        self.logger.record("cpu/usage", cpu_percent)
        self.logger.record("memory/usage", memory_percent)
        if NVIDIA_SMI_AVAILABLE:
            self.logger.record("gpu/usage", gpu_util)
            self.logger.record("gpu/memory_usage", gpu_mem_used)
        return True

    def _on_training_end(self):
        if NVIDIA_SMI_AVAILABLE:
            nvidia_smi.nvmlShutdown()

        if self.verbose >= 1:
            # Print summary statistics
            print("\n=== Training Summary ===")
            print(f"Average CPU Usage: {np.mean(self.cpu_usages):.1f}%")
            if NVIDIA_SMI_AVAILABLE:
                print(f"Average GPU Usage: {np.mean(self.gpu_usages):.1f}%")
                print(f"Average GPU Memory Usage: {np.mean(self.gpu_memories):.1f}%")
            print("=====================")


class PeriodicSaveCallback(BaseCallback):
    """
    Callback for saving the model periodically during training.

    :param save_freq: Number of timesteps between saves
    :param save_path: Path to save the models
    :param name_prefix: Prefix for the saved model files
    """

    def __init__(self, save_freq: int = 10000, save_path: str = "./models",
                 name_prefix: str = "model", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        # Create save directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def _init_callback(self) -> None:
        """
        Initialize callback variables
        """
        # Create save path if it doesn't exist
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Save the model if the number of timesteps is divisible by save_freq

        :return: True if the callback should continue
        """
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.n_calls}_steps.zip"
            )
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {path}")
        return True