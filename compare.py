import os
import sys
import numpy as np
import torch
import logging
import multiprocessing as mp
import matplotlib.pyplot as plt
from colorama import Fore, Style

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.env_multi_eqn import multiEqn

from utils.utils_custom_policies import HarmonicPolicy
from utils.utils_callbacks import IntrinsicReward, get_device, DummyVecEnvWithReset
from rllte.xplore.reward import ICM



# ------------------------------------------------------------------------------
# A. Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

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

# ------------------------------------------------------------------------------
# C. Train a Single Run for MaskablePPO
# ------------------------------------------------------------------------------
def train_single_run(agent_type, total_timesteps=50000, seed=None, log_interval=1000):
    """
    Trains one run of MaskablePPO on multiEqn.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = multiEqn(level=8)
    env = ActionMasker(env, lambda env: env.action_mask)
    env = DummyVecEnvWithReset([lambda: env])
    #env = DummyVecEnv([lambda: env])

    callback = AccuracyLoggingCallback(env, total_timesteps, log_interval=log_interval)
    callback_ir = IntrinsicReward(ICM(env, device=get_device()))  # internal curiosity


    logging.info(f"Starting training for {agent_type} with seed={seed}...")

    policy_kwargs = dict(net_arch=[256, 256, 256])  # Three layers of 256 neurons each

    if agent_type == 'ppo_harmonic':
        policy_kwargs.update(
            distance_norm="L2",
            harmonic_exponent=128,
            weight_reg=0.01
        )
        model = MaskablePPO(
            policy=HarmonicPolicy,
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed
        )
    elif agent_type == 'ppo':
        model = MaskablePPO('MlpPolicy', env=env, policy_kwargs=policy_kwargs, verbose=0, seed=seed)
    else:
        raise ValueError("Invalid agent type")

    model.learn(total_timesteps=total_timesteps, callback=[callback,callback_ir])

    model.save(f'models/compare/{agent_type}')

    return callback.steps, callback.train_accs, callback.test_accs

# ------------------------------------------------------------------------------
# D. Run Ensemble Training
# ------------------------------------------------------------------------------
def run_ensemble(agent_type, total_timesteps=50000, log_interval=1000, ensemble_size=5, num_workers=5):
    """
    Runs multiple training runs in parallel.
    """
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        args_list = [(agent_type, total_timesteps, 100 * i, log_interval) for i in range(ensemble_size)]
        results = pool.starmap(train_single_run, args_list)

    return results  # List of (steps, train_accs, test_accs)

# ------------------------------------------------------------------------------
# E. Plot Results for Both Agents
# ------------------------------------------------------------------------------
def plot_ensemble(agent_type, ensemble_data, ax):
    """
    Plots mean train/test accuracy with min-max shading for an ensemble.
    """
    steps, train_accs, test_accs = zip(*ensemble_data)  # Unpack

    steps = np.array(steps[0])  # All runs have the same step sizes
    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)

    train_mean, train_min, train_max = np.mean(train_accs, axis=0), np.min(train_accs, axis=0), np.max(train_accs, axis=0)
    test_mean, test_min, test_max = np.mean(test_accs, axis=0), np.min(test_accs, axis=0), np.max(test_accs, axis=0)

    ax.fill_between(steps, train_min, train_max, color="blue", alpha=0.2)
    ax.fill_between(steps, test_min, test_max, color="red", alpha=0.2)
    ax.plot(steps, train_mean, label="Train Accuracy", color="blue")
    ax.plot(steps, test_mean, label="Test Accuracy", color="red")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Ensemble (size=5) - {agent_type}")
    ax.legend()
    ax.set_ylim([0,1.1])
    ax.grid(True)

# ------------------------------------------------------------------------------
# F. Run and Compare MaskablePPO vs. MaskablePPO-Harmonic
# ------------------------------------------------------------------------------
def main():
    total_timesteps = 5*10**6
    log_interval = int(0.01 * total_timesteps)
    ensemble_size, num_workers = 3, 3

    print("Running ensemble for Regular PPO...")
    ensemble_reg = run_ensemble("ppo", total_timesteps, log_interval, ensemble_size, num_workers)
    print("Running ensemble for Harmonic PPO...")
    ensemble_har = run_ensemble("ppo_harmonic", total_timesteps, log_interval, ensemble_size, num_workers)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    plot_ensemble("Regular PPO", ensemble_reg, axes[0])
    plot_ensemble("Harmonic PPO", ensemble_har, axes[1])

    plt.suptitle(f"Train & Test Accuracy: Regular PPO vs Harmonic PPO")
    plt.tight_layout()
    plt.savefig("data/ppo_vs_harmonic.png")
    plt.show()

if __name__ == "__main__":
    main()
