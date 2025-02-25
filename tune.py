import torch
import logging
import numpy as np
import multiprocessing as mp
import gymnasium as gym
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

# Import custom Harmonic Policy
from utils.utils_custom_policies import HarmonicPolicy
from utils.utils_callbacks import IntrinsicReward, get_device, DummyVecEnvWithReset
from rllte.xplore.reward import ICM

# Import environments
from envs.env_gridworld import gridWorld
from envs.env_single_eqn import singleEqn
from envs.env_multi_eqn import multiEqn


# ------------------------------------------------------------------------------
# A. Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

# ------------------------------------------------------------------------------
# B. Custom Accuracy Logging Callback
# ------------------------------------------------------------------------------
class AccuracyLoggingCallback(BaseCallback):
    """Tracks accuracy over training and implements early stopping if test accuracy reaches 1.0."""
    
    def __init__(self, env, log_interval=10**5, total_steps=50000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.log_interval = log_interval
        self.total_steps = total_steps
        self.train_accs, self.test_accs, self.steps = [], [], []
        self.early_stop_triggered = False

    def _on_step(self) -> bool:
        """Logs accuracy and checks for early stopping."""
        if self.n_calls % self.log_interval == 0:
            train_acc, test_acc = self.compute_accuracy()
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            self.steps.append(self.n_calls)

            logging.info(f"Step {self.n_calls}: Train Acc = {train_acc:.2f}, Test Acc = {test_acc:.2f}")

            if test_acc == 1.0 and not self.early_stop_triggered:
                logging.info(f"Early stopping triggered at step {self.n_calls}.")
                self.early_stop_triggered = True
                return False  # Stop training

        return True

    def compute_accuracy(self, num_eval_episodes=100):
        """Computes accuracy as the fraction of solved equations."""
        def eval_env(mode):
            successes = []
            for _ in range(num_eval_episodes):
                obs = self.env.reset(options=mode)
                done, steps = False, 0
                while not done and steps < self.env.envs[0].max_steps:
                    action = self.model.predict(obs, deterministic=True)[0]
                    obs, _, done, info = self.env.step(action)
                    info = info[0]
                    steps += 1
                successes.append(int(info["is_solved"]))
            return np.mean(successes)

        return eval_env('train'), eval_env('test')


# ------------------------------------------------------------------------------
# C. Training Function
# ------------------------------------------------------------------------------
def train_harmonic(env_name, distance_norm, exponent, weight_reg, total_timesteps=30000, seed=None):
    """Trains MaskablePPO + HarmonicPolicy and logs accuracy."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Load environment inside the function
    env_map = {
        'grid_world': lambda: gridWorld(difficulty='easy'),
        'single_eqn': lambda: singleEqn(main_eqn='a/x+b'),
        'multi_eqn': lambda: multiEqn(level=8)
    }
    env = env_map.get(env_name, lambda: gym.make(env_name))()

    env = ActionMasker(env, lambda env: env.action_mask)
    env = DummyVecEnvWithReset([lambda: env])

    # Setup model
    model = MaskablePPO(
        policy=HarmonicPolicy,
        env=env,
        policy_kwargs=dict(
            distance_norm=distance_norm,
            harmonic_exponent=exponent,
            weight_reg=weight_reg,
            net_arch=[256, 256, 256]  # Three layers of 256 neurons each
        ),
        verbose=0,
        seed=seed
    )

    # Train with accuracy logging
    callback = AccuracyLoggingCallback(env, total_steps=total_timesteps, log_interval=int(0.1 * total_timesteps))
    callback_ir = IntrinsicReward(ICM(env, device=get_device()))  # internal curiosity
    model.learn(total_timesteps=total_timesteps, callback=[callback, callback_ir])

    final_test_acc = round(np.mean(callback.test_accs[-10:]), 2)  # Average of last 10 test accs
    logging.info(f"Final test accuracy for exponent={exponent}: {final_test_acc}")

    return callback.steps, callback.train_accs, callback.test_accs


# ------------------------------------------------------------------------------
# D. Parallel Execution Setup
# ------------------------------------------------------------------------------
def harmonic_worker(args):
    """Worker function for parallel execution."""
    run_idx, env_name, dist_norm, exponent, weight_reg, total_timesteps, seed_offset = args
    seed = 100 * run_idx + seed_offset

    logging.info(f"Starting worker {run_idx} for exponent={exponent}")

    try:
        return train_harmonic(env_name, dist_norm, exponent, weight_reg, total_timesteps, seed)
    except Exception as e:
        logging.error(f"Worker {run_idx} crashed with error: {e}")
        return None


def run_parallel(distance_norm, exponent, weight_reg, env_name, total_timesteps, ensemble_size, seed_offset=0):
    """Runs parallel training for a single hyperparameter combo."""
    ctx = mp.get_context("spawn")  # Ensure cross-platform safety
    with ctx.Pool(processes=ensemble_size, maxtasksperchild=1) as pool:
        args_list = [(i, env_name, distance_norm, exponent, weight_reg, total_timesteps, seed_offset)
                     for i in range(ensemble_size)]
        return pool.map(harmonic_worker, args_list)


# ------------------------------------------------------------------------------
# E. Main Tuning Script
# ------------------------------------------------------------------------------
def main():
    env_name = 'multi_eqn'
    total_timesteps = 10**6
    ensemble_size = 5

    distance_norms = ['L2']
    #exponents = [8, 32, 64, 128, 256]
    exponents = [96, 112, 128, 144, 160, 192]
    weight_regs = [0]

    param_results = {}

    for dist_norm in distance_norms:
        for exp in exponents:
            for wreg in weight_regs:
                logging.info(f"Running: dist={dist_norm}, exponent={exp}, wreg={wreg}")
                param_results[(dist_norm, exp, wreg)] = run_parallel(
                    dist_norm, exp, wreg, env_name, total_timesteps, ensemble_size
                )

    # Summarize results
    summary = [
        (dist_norm, exp, wreg, np.mean([runs[2][-1] for runs in data_runs if runs]))  # Test accuracy
        for (dist_norm, exp, wreg), data_runs in param_results.items()
    ]
    summary.sort(key=lambda x: x[3], reverse=True)

    print("\n=== TUNING SUMMARY ===")
    print("dist_norm | exponent | weight_reg | final_test_acc")
    for row in summary:
        print(f"{row[0]:<8}  {row[1]:<8}  {row[2]:<8}  {row[3]:.2f}")

    # Plot train and test accuracy with min-max shading
    fig, axes = plt.subplots(nrows=len(summary), ncols=1, figsize=(6, 3*len(summary)), sharex=False)
    if len(summary) == 1:
        axes = [axes]

    for ((dist_norm, exp, wreg), data_runs), ax in zip(param_results.items(), axes):
        steps = data_runs[0][0]
        train_accs = np.array([run[1] for run in data_runs])
        test_accs = np.array([run[2] for run in data_runs])

        train_mean, train_min, train_max = np.mean(train_accs, axis=0), np.min(train_accs, axis=0), np.max(train_accs, axis=0)
        test_mean, test_min, test_max = np.mean(test_accs, axis=0), np.min(test_accs, axis=0), np.max(test_accs, axis=0)

        ax.fill_between(steps, train_min, train_max, color="blue", alpha=0.2)
        ax.fill_between(steps, test_min, test_max, color="red", alpha=0.2)
        ax.plot(steps, train_mean, label=f"Train Acc - exp={exp}", color="blue")
        ax.plot(steps, test_mean, label=f"Test Acc - exp={exp}", color="red")

        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1.1])
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Training Steps")
    plt.suptitle(f"Harmonic Tuning on {env_name}")
    plt.tight_layout()
    plt.savefig('data/tuning_results_acc.png')
    plt.show()


if __name__ == "__main__":
    main()
