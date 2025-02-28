import os
import gymnasium as gym
from gymnasium import spaces
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

from utils.utils_custom_policies import HarmonicPolicy  # Import if you have a custom policy

GREEN = "\033[32m"
RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format=f"{GREEN}%(asctime)s{RESET} - %(message)s",
    datefmt="%H:%M:%S"
)

class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=1, shape=(500,), dtype=np.float32)

    def observation(self, obs):
        one_hot = np.zeros(500, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


class EpisodeReturnLoggingCallback(BaseCallback):
    """
    Logs total episode returns for any env.
    Also logs intermediate progress every `log_interval` timesteps.
    """
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        # We'll compute 10% of total timesteps for the log interval
        self.log_interval = max(1, int(0.1 * total_timesteps))
        self.episode_returns = []
        self.episode_ends = []
        self.current_return = 0.0
        self.current_length = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        # Accumulate total reward in this ongoing episode
        self.current_return += np.sum(rewards)
        self.current_length += 1

        # If any environment is done, record the total return
        for done in dones:
            if done:
                self.episode_returns.append(self.current_return)
                self.episode_ends.append(self.num_timesteps)
                self.current_return = 0.0
                self.current_length = 0

        # --- Logging intermediate progress ---
        # Check if we've hit the log interval
        if self.num_timesteps % self.log_interval == 0:
            # Mean return over the last 10 episodes (or fewer if <10 have finished)
            if len(self.episode_returns) > 0:
                recent_episodes = self.episode_returns[-10:]
                mean_return = np.mean(recent_episodes)
            else:
                mean_return = 0.0
            logging.info(f"Step={self.num_timesteps}: <return> = {mean_return:.2f}")
        return True


def train_agent(args):
    """
    Trains a single PPO or PPO-Harmonic agent on the given environment.
    args is a tuple: (env_name, agent_type, seed, total_timesteps).
    """
    env_name, agent_type, seed, total_timesteps = args
    logging.info(f"Starting training: env={env_name}, agent={agent_type}, seed={seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    policy_kwargs = dict(net_arch=[128, 128, 128])

    if agent_type == "ppo-harmonic":
        policy_kwargs.update(
            distance_norm="L2",
            # harmonic_exponent=128,  # general
            # harmonic_exponent=64,  # acrobot
            # harmonic_exponent=128,  # taxi
            harmonic_exponent=256,  # MC
            weight_reg=0.01
        )
        policy = HarmonicPolicy
    else:
        policy = "MlpPolicy"

    env = gym.make(env_name)
    if env_name in ['Taxi-v3', 'FrozenLake-v1']:
        env = OneHotWrapper(env)
    env = DummyVecEnv([lambda: env])

    model = PPO(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        verbose=0,
        ent_coef = 0.1,
        seed=seed
    )

    # Create our callback with logging
    callback = EpisodeReturnLoggingCallback(total_timesteps=total_timesteps, verbose=0)

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback)

    logging.info(f"Finished training: env={env_name}, agent={agent_type}, seed={seed}")

    model_dir = f"models/{env_name}/{agent_type}_{seed}"
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    model.save(model_dir)

    return (
        agent_type,
        np.array(callback.episode_ends),
        np.array(callback.episode_returns),
        None
    )


def main():
    parser = argparse.ArgumentParser(description="Train PPO/PPO-Harmonic on an environment")
    parser.add_argument("--env_name", type=str, default="CartPole-v1",
                        help="Gym environment name (default: CartPole-v1)")
    parser.add_argument("--total_timesteps", type=int, default=10000,
                        help="Total timesteps (default: 10000)")
    args = parser.parse_args()

    print("\n" + "-" * 25)
    print(f"Running {args.env_name}")
    print("-" * 25 + "\n")

    ensemble_size = 3
    n_workers = 6
    agent_types = ["ppo", "ppo-harmonic"]

    jobs = [
        (args.env_name, agent_type, seed, args.total_timesteps)
        for agent_type in agent_types
        for seed in range(ensemble_size)
    ]

    with Pool(processes=n_workers) as pool:
        results_list = pool.map(train_agent, jobs)

    # Organize results by agent type
    results = {agent_type: [] for agent_type in agent_types}
    for (agent_type, ends_arr, returns_arr, _) in results_list:
        results[agent_type].append((ends_arr, returns_arr))

    # Save results
    data_path = f"models/{args.env_name}/learning_data.pkl"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(results, f)
    logging.info(f"Saved results to {data_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    for agent_type, curves in results.items():
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
        plt.plot(timesteps, mean_hist, label=agent_type.upper(), alpha=0.9)
        plt.fill_between(timesteps, min_hist, max_hist, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Return")
    plt.title(f"Episode Returns: {args.env_name} (Mean ± Min/Max)")
    plt.legend()
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{args.env_name}_learning_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
