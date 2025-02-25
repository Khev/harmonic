import os
import gymnasium as gym
import minigrid
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TransformObservation
from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Pool

from utils.utils_custom_policies import HarmonicPolicy
from utils.utils_callbacks import AccuracyLoggingCallback
from rllte.xplore.reward import ICM

# --- Custom Observation Wrapper ---
class CustomFlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # We assume we only care about the "image" component of the dict observation
        image_space = env.observation_space.spaces["image"]
        self.observation_space = gym.spaces.Box(
            low=image_space.low.flatten(),
            high=image_space.high.flatten(),
            dtype=image_space.dtype
        )

    def observation(self, obs):
        return obs["image"].flatten()

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class StepRewardLoggingCallback(BaseCallback):
    """
    A callback that logs the immediate (per-step) reward at each timestep.

    In a vectorized environment, `self.locals["rewards"]` is an array of rewards
    (one per environment). We can record their mean, sum, or store them individually.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # "rewards" is a [n_envs] array of immediate rewards in a vectorized env
        rewards = self.locals.get("rewards", [])
        if len(rewards) > 0:
            # You can store mean, sum, or each reward individually
            mean_reward = float(np.mean(rewards))
            self.step_rewards.append(mean_reward)
            self.timesteps.append(self.num_timesteps)

            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: reward = {mean_reward:.2f}")

        return True



# --- Training Function ---
def train_agent(args):
    # args is a tuple: (agent_type, seed, total_timesteps)
    agent_type, seed, total_timesteps = args
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy_kwargs = dict(net_arch=[128, 128, 128])
    if agent_type == "ppo-harmonic":
        policy_kwargs.update(
            distance_norm="L2",
            harmonic_exponent=128,
            weight_reg=0.01
        )
        policy = HarmonicPolicy
    else:
        policy = "MlpPolicy"

    # Create environment and wrap it
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    env = CustomFlattenObservation(env)
    env = DummyVecEnv([lambda: env])

    # Initialize PPO model
    model = PPO(policy, env, policy_kwargs=policy_kwargs,
                learning_rate=1e-4, verbose=0, seed=seed)
    callback = StepRewardLoggingCallback(verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return agent_type, callback.timesteps, callback.step_rewards

# --- Main function to run ensemble ---
def main():
    ensemble_size = 3
    n_workers = 6
    total_timesteps = 10**5
    agent_types = ["ppo", "ppo-harmonic"]

    # Create jobs for both agent types (2 x 3 = 6 runs)
    jobs = [(agent_type, seed, total_timesteps) 
            for agent_type in agent_types 
            for seed in range(ensemble_size)]

    # Run all jobs in parallel
    with Pool(processes=n_workers) as pool:
        results_list = pool.map(train_agent, jobs)

    # Organize results by agent type
    results = {agent_type: [] for agent_type in agent_types}
    for agent_type, timesteps, rewards in results_list:
        results[agent_type].append((timesteps, rewards))

    # --- Plot Learning Curves ---
    plt.figure(figsize=(10, 6))
    for agent_type, curves in results.items():
        # Assume each run logs at similar timesteps
        all_timesteps = np.array([curve[0] for curve in curves])
        all_rewards = np.array([curve[1] for curve in curves])
        mean_rewards = np.mean(all_rewards, axis=0)
        min_rewards = np.min(all_rewards, axis=0)
        max_rewards = np.max(all_rewards, axis=0)
        # Use timesteps from the first run
        x = curves[0][0]
        plt.plot(x, mean_rewards, label=agent_type.upper())
        plt.fill_between(x, min_rewards, max_rewards, alpha=0.3)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Learning Curves: PPO vs PPO-Harmonic")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
