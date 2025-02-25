
import torch, logging
import torch as th
import numpy as np
import multiprocessing as mp

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm  # Required for `init_callback`
from stable_baselines3.common.vec_env import DummyVecEnv

# ------------------------------------------------------------------------------
# A. Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)


class DummyVecEnvWithReset(DummyVecEnv):
    def reset(self, **kwargs):
        # Call reset on each underlying environment with the kwargs.
        results = [env.reset(**kwargs) for env in self.envs]
        # results is a list of tuples: (obs, info)
        obs, infos = zip(*results)
        # Return only the stacked observations.
        return np.stack(obs)


def get_device():
    """Returns the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        print('Found CUDA: using GPU')
        cur_proc_identity = mp.current_process()._identity
        if cur_proc_identity:
            return (cur_proc_identity[0] - 1) % torch.cuda.device_count()
        else:
            return 0
    else:
        print('CUDA not found: using CPU')
        return 'cpu'


class IntrinsicReward(BaseCallback):
    """
    A more efficient callback for logging intrinsic rewards in RL training.
    """

    def __init__(self, irs, verbose=False, log_interval=100):
        super(IntrinsicReward, self).__init__(verbose)
        self.irs = irs
        self.buffer = None
        self.rewards_internal = []  # Store intrinsic rewards for logging
        self.log_interval = log_interval
        self.last_computed_intrinsic_rewards = None  # Store for logging
        self.verbose = verbose

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        Instead of computing at each step, log previously computed intrinsic rewards.
        """
        if self.last_computed_intrinsic_rewards is not None:
            # Get last intrinsic reward from the rollout buffer
            intrinsic_reward = self.last_computed_intrinsic_rewards[-1]
            self.rewards_internal.append(intrinsic_reward)

        # ✅ Print intrinsic reward stats every `log_interval` steps
        if self.n_calls % self.log_interval == 0 and self.rewards_internal and self.verbose:
            mean_intrinsic = np.mean(self.rewards_internal[-self.log_interval:])
            min_intrinsic = np.min(self.rewards_internal[-self.log_interval:])
            max_intrinsic = np.max(self.rewards_internal[-self.log_interval:])
            main_eqn = self.locals["infos"][0]['main_eqn']
            print(f"{main_eqn}: Step {self.num_timesteps}: "
                  f"(min, mean, max)_reward_internal = ({min_intrinsic:.3f}, {mean_intrinsic:.3f}, {max_intrinsic:.3f})\n")

        return True

    def _on_rollout_end(self) -> None:
        """
        Efficiently compute intrinsic rewards once per rollout and store them.
        """
        obs = th.as_tensor(self.buffer.observations).float()
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"]).float()
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)

        # ✅ Compute **intrinsic rewards for the entire rollout** at once
        intrinsic_rewards = self.irs.compute(
            samples=dict(observations=obs, actions=actions,
                         rewards=rewards, terminateds=dones,
                         truncateds=dones, next_observations=new_obs),
            sync=True
        ).cpu().numpy()

        # ✅ Store them so `_on_step()` can access them
        self.last_computed_intrinsic_rewards = intrinsic_rewards

        # ✅ Add intrinsic rewards to the rollout buffer
        self.buffer.advantages += intrinsic_rewards
        self.buffer.returns += intrinsic_rewards



# ------------------------------------------------------------------------------
# B. Custom Callback for Accuracy Logging with Early Stopping
# ------------------------------------------------------------------------------
class AccuracyLoggingCallback(BaseCallback):
    """
    Logs train and test accuracy at given intervals and stores values for ensemble plotting.
    Implements early stopping if test_acc == 1.0.
    """
    def __init__(self, env, total_steps, log_interval=1000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.log_interval = log_interval
        self.total_steps = total_steps
        self.max_logs = total_steps // log_interval
        self.train_accs = []
        self.test_accs = []
        self.steps = []
        self.rewards_ext = []  # Initialize the rewards_ext attribute
        self.early_stop_triggered = False  # Flag to track early stopping

    def _on_step(self) -> bool:

        # Get the latest external reward from the environment
        # reward_ext = self.locals["rewards"][0]
        # self.rewards_ext.append(reward_ext)
        # info = self.locals["infos"][0]

        # if info.get('is_solved', False):
        #     main_eqn, lhs, rhs = info.get('main_eqn'), info.get('lhs'), info.get('rhs')
        #     print(Fore.GREEN + f'\nSolved {main_eqn} = 0  ==>  {lhs} = {rhs} at Nstep = {self.num_timesteps}' + Style.RESET_ALL)
        #     self.T_solve = self.num_timesteps

        if self.n_calls % self.log_interval == 0:
            train_acc, test_acc = self.compute_accuracy()
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            self.steps.append(self.n_calls)

            logging.info(f"Step {self.n_calls}: Train Acc = {train_acc:.2f}, Test Acc = {test_acc:.2f}")

            # Early stopping if test accuracy reaches 1.0
            if test_acc == 1.0 and not self.early_stop_triggered:
                self.early_stop_triggered = True
                logging.info(f"Early stopping triggered at step {self.n_calls}. Filling in missing data...")
                self.fill_missing_data()
                return False  # Stop training

        return True

    def compute_accuracy(self, num_eval_episodes=100):
        """
        Computes accuracy as the fraction of equations solved within max_steps.
        """
        train_successes, test_successes = [], []

        # Train Accuracy
        for _ in range(num_eval_episodes):
            #obs, _ = self.env.reset(options='train')
            #obs, _ = self.env.reset(options={'mode': 'train'})
            obs  = self.env.reset(options={'mode': 'train'})
            done, steps = False, 0
            while not done and steps < self.env.envs[0].max_steps:
                action = self.model.predict(obs, deterministic=True)[0]
                # obs, _, done, _, info = self.env.step(action)
                obs, _, done, info = self.env.step(action)
                info = info[0]
                steps += 1
            train_successes.append(int(info["is_solved"]))

        # Test Accuracy
        for _ in range(num_eval_episodes):
            #obs, _ = self.env.reset(options='test')
            obs = self.env.reset(options={'mode': 'test'})
            done, steps = False, 0
            while not done and steps < self.env.envs[0].max_steps:
                action = self.model.predict(obs, deterministic=True)[0]
                obs, _, done, info = self.env.step(action)
                info = info[0]
                steps += 1
            test_successes.append(int(info["is_solved"]))

        return np.mean(train_successes), np.mean(test_successes)


    def fill_missing_data(self):
        """
        Ensures that the recorded data is uniform by filling missing values after early stopping.
        Pads train_accs, test_accs, and steps to the expected length.
        """
        total_length = self.total_steps // self.log_interval  # Expected number of log entries

        # Get last recorded values
        last_train_acc = self.train_accs[-1]
        last_test_acc = self.test_accs[-1]
        last_step = self.steps[-1]

        # Pad lists to the correct length
        self.steps = self.pad_list_to_length(self.steps, total_length, last_step + self.log_interval)
        self.train_accs = self.pad_list_to_length(self.train_accs, total_length, last_train_acc)
        self.test_accs = self.pad_list_to_length(self.test_accs, total_length, last_test_acc)

        logging.info(f"Final step count: {len(self.steps)}, Expected: {total_length}")
        logging.info(f"Final train_accs length: {len(self.train_accs)}, test_accs length: {len(self.test_accs)}")

    def pad_list_to_length(self, lst, target_length, pad_value):
        """
        Pads a list to a specified length using a given pad value.
        """
        while len(lst) < target_length:
            lst.append(pad_value)
        return lst