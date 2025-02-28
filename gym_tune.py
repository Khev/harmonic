import os
import gymnasium as gym
import numpy as np
import torch
import logging
import pickle
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Pool
import argparse

from utils.utils_custom_policies import HarmonicPolicy  # Import your custom policy if needed

GREEN = "\033[32m"
RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format=f"{GREEN}%(asctime)s{RESET} - %(message)s",
    datefmt="%H:%M:%S"
)

# -------------------------------------------------------------------
# Single callback: logs total episode returns for ANY environment
# -------------------------------------------------------------------
class EpisodeReturnLoggingCallback(BaseCallback):
    """
    Logs the total return (sum of rewards) for each episode.
    Also logs intermediate progress every `log_interval` timesteps.
    """
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.log_interval = max(1, int(0.1 * total_timesteps))
        self.episode_returns = []
        self.episode_ends = []
        self.current_return = 0.0
        self.current_length = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        self.current_return += np.sum(rewards)
        self.current_length += 1

        for done in dones:
            if done:
                self.episode_returns.append(self.current_return)
                self.episode_ends.append(self.num_timesteps)
                self.current_return = 0.0
                self.current_length = 0

        if self.num_timesteps % self.log_interval == 0:
            if len(self.episode_returns) > 0:
                recent = self.episode_returns[-10:]
                mean_return = np.mean(recent)
            else:
                mean_return = 0.0
            logging.info(f"Step={self.num_timesteps}: MeanReturn={mean_return:.2f}")
        return True

# -------------------------------------------------------------------
# Training function (tuning harmonic_exponent)
# -------------------------------------------------------------------
def train_agent(args):
    """
    Trains a single PPO-Harmonic agent on the given environment.
    args is a tuple: (env_name, harmonic_exponent, seed, total_timesteps).
    """
    env_name, harmonic_exponent, seed, total_timesteps = args
    logging.info(f"Starting training: env={env_name}, harmonic_exponent={harmonic_exponent}, seed={seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set up policy_kwargs with the tuned harmonic_exponent value
    policy_kwargs = dict(net_arch=[128, 128, 128])
    policy_kwargs.update(
        distance_norm="L2",
        harmonic_exponent=harmonic_exponent,
        weight_reg=0.01
    )
    policy = HarmonicPolicy

    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    model = PPO(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        ent_coef=0.1,
        verbose=0,
        seed=seed
    )

    callback = EpisodeReturnLoggingCallback(total_timesteps=total_timesteps, verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    logging.info(f"Finished training: env={env_name}, harmonic_exponent={harmonic_exponent}, seed={seed}")

    model_dir = f"models/{env_name}/harmonic_{harmonic_exponent}_{seed}"
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    model.save(model_dir)

    return (harmonic_exponent,
            np.array(callback.episode_ends),
            np.array(callback.episode_returns),
            None)

# -------------------------------------------------------------------
# Main function for tuning harmonic_exponent
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tune PPO-Harmonic on an environment")
    parser.add_argument("--env_name", type=str, default="LunarLander-v2",
                        help="Gym environment name (default: LunarLander-v2)")
    parser.add_argument("--total_timesteps", type=int, default=5000000,
                        help="Total timesteps per trial (default: 100000)")
    args = parser.parse_args()

    print("\n" + "-" * 25)
    print(f"Running tuning on {args.env_name}")
    print("-" * 25 + "\n")

    # Define harmonic exponents to try (for brevity, here only a subset)
    harmonic_exponents = [1, 2, 4, 8, 16, 32, 64, 128]
    harmonic_exponents = [128, 256, 512, 1024]
    trials_per_exponent = 5  # 3 trials per value

    jobs = [
        (args.env_name, he, seed, args.total_timesteps)
        for he in harmonic_exponents
        for seed in range(trials_per_exponent)
    ]

    n_workers = min(len(jobs), 6)
    with Pool(processes=n_workers) as pool:
        results_list = pool.map(train_agent, jobs)

    # Organize results by harmonic_exponent value
    results = {he: [] for he in harmonic_exponents}
    for (he, ends_arr, returns_arr, _) in results_list:
        results[he].append((ends_arr, returns_arr))

    # Save results to disk
    data_path = f"models/{args.env_name}/tuning_learning_data.pkl"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(results, f)
    logging.info(f"Saved tuning results to {data_path}")

    # Plot aggregated episode returns per harmonic_exponent
    plt.figure(figsize=(12, 8))
    # Dictionary to store average return over the last 10% of timesteps for ranking
    final_avg_returns = {}
    for he, curves in results.items():
        max_timestep = 0
        for (episode_ends, episode_returns) in curves:
            if len(episode_ends) > 0:
                max_timestep = max(max_timestep, episode_ends[-1])
        run_histories = []
        for (episode_ends, episode_returns) in curves:
            hist = np.zeros(max_timestep + 1, dtype=np.float32)
            last_return = 0.0
            prev_end = 0
            for end_step, ret in zip(episode_ends, episode_returns):
                hist[prev_end:end_step] = last_return
                last_return = ret
                prev_end = end_step
            hist[prev_end:] = last_return
            run_histories.append(hist)
        run_histories = np.array(run_histories, dtype=np.float32)
        mean_hist = run_histories.mean(axis=0)
        min_hist = run_histories.min(axis=0)
        max_hist = run_histories.max(axis=0)
        timesteps = np.arange(max_timestep + 1)
        plt.plot(timesteps, mean_hist, label=f"Exponent {he}", alpha=0.9)
        plt.fill_between(timesteps, min_hist, max_hist, alpha=0.2)
        
        # Compute the average of the last 10% of timesteps for ranking
        last_10_percent = int(0.1 * (max_timestep + 1))
        avg_last10 = run_histories[:, -last_10_percent:].mean(axis=1).mean()
        final_avg_returns[he] = avg_last10

    # Rank order the harmonic_exponents by the average of the last 10% returns
    ranked = sorted(final_avg_returns.items(), key=lambda x: x[1], reverse=True)
    print("Ranked Harmonic Exponents (best to worst based on last 10% average return):")
    for he, avg_return in ranked:
        print(f"Exponent {he}: Average Last 10% Return = {avg_return:.2f}")

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Return")
    plt.title(f"Episode Returns for {args.env_name} (Tuning Harmonic Exponent)")
    plt.legend()
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{args.env_name}_tuning_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
