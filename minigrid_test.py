import os
import gymnasium as gym
import minigrid
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Pool

from utils.utils_custom_policies import HarmonicPolicy

# Set up logging with timestamps
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")

# --- Custom Observation Wrapper ---
class CustomFlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        image_space = env.observation_space.spaces["image"]
        self.observation_space = gym.spaces.Box(
            low=image_space.low.flatten(),
            high=image_space.high.flatten(),
            dtype=image_space.dtype
        )

    def observation(self, obs):
        return obs["image"].flatten()

# --- Environment Functions ---
TRAIN_SEEDS = [0, 1, 2, 3, 4]
TEST_SEEDS  = [10, 11, 12, 13, 14]
GRID_SIZE = 6

def make_train_env():
    """Create a DoorKey environment with a specific train seed."""
    env = gym.make(f"MiniGrid-DoorKey-{GRID_SIZE}x{GRID_SIZE}-v0")
    # Instead of choosing randomly, we expect the evaluator to supply a seed.
    # For training, you may still allow some randomness; here we set one seed per reset.
    seed = int(np.random.choice(TRAIN_SEEDS))
    env.reset(seed=seed)
    env = CustomFlattenObservation(env)
    return env

def make_test_env():
    """Create a DoorKey environment with a specific test seed."""
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    seed = int(np.random.choice(TEST_SEEDS))
    env.reset(seed=seed)
    env = CustomFlattenObservation(env)
    return env

# --- Evaluation Functions ---
def evaluate_on_train(model, num_episodes_per_seed=1):
    """
    Evaluate the model systematically on all TRAIN_SEEDS.
    Runs num_episodes_per_seed episodes for each seed.
    Returns the overall success rate.
    """
    successes = 0
    total_episodes = 0
    for s in TRAIN_SEEDS:
        for _ in range(num_episodes_per_seed):
            env = gym.make("MiniGrid-DoorKey-5x5-v0")
            env.reset(seed=int(s))
            env = CustomFlattenObservation(env)
            env = DummyVecEnv([lambda: env])
            obs = env.reset()
            done, total_reward = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = env.step(action)
                total_reward += rewards[0]
                done = dones[0]
            if total_reward > 0.0:
                successes += 1
            total_episodes += 1
    return successes / total_episodes

def evaluate_on_test(model, num_episodes_per_seed=1):
    """
    Evaluate the model systematically on all TEST_SEEDS.
    Runs num_episodes_per_seed episodes for each seed.
    Returns the overall success rate.
    """
    successes = 0
    total_episodes = 0
    for s in TEST_SEEDS:
        for _ in range(num_episodes_per_seed):
            env = gym.make("MiniGrid-DoorKey-5x5-v0")
            env.reset(seed=int(s))
            env = CustomFlattenObservation(env)
            env = DummyVecEnv([lambda: env])
            obs = env.reset()
            done, total_reward = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = env.step(action)
                total_reward += rewards[0]
                done = dones[0]
            if total_reward > 0.0:
                successes += 1
            total_episodes += 1
    return successes / total_episodes

# --- Custom Callback for Periodic Evaluation ---
class PeriodicEvaluationCallback(BaseCallback):
    """
    A callback that evaluates the current model every log_interval timesteps
    and logs train/test success rates. Implements early stopping.
    """
    def __init__(self, log_interval, num_train_episodes=1, num_test_episodes=1, 
                 early_stopping=False, patience=5, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.num_train_episodes = num_train_episodes
        self.num_test_episodes = num_test_episodes
        self.early_stopping = early_stopping
        self.patience = patience
        self.eval_timesteps = []
        self.train_success = []
        self.test_success = []
        self.stagnation_counter = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval < self.training_env.num_envs:
            train_acc = evaluate_on_train(self.model, self.num_train_episodes)
            test_acc = evaluate_on_test(self.model, self.num_test_episodes)
            self.eval_timesteps.append(self.num_timesteps)
            self.train_success.append(train_acc)
            self.test_success.append(test_acc)
            if self.verbose > 0:
                logging.info(f"Timestep {self.num_timesteps}: Train Success = {train_acc:.2f}, Test Success = {test_acc:.2f}")
            if self.early_stopping:
                if test_acc == 1.0:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
                if self.stagnation_counter >= self.patience:
                    logging.info(f"Early stopping triggered at timestep {self.num_timesteps}")
                    return False
        return True

    def get_padded_results(self, full_length):
        """Pads evaluation data to match the expected full length."""
        pad_length = (full_length // self.log_interval) - len(self.eval_timesteps)
        if pad_length > 0:
            last_timestep = self.eval_timesteps[-1]
            last_train_success = self.train_success[-1]
            last_test_success = self.test_success[-1]
            self.eval_timesteps += [last_timestep] * pad_length
            self.train_success += [last_train_success] * pad_length
            self.test_success += [last_test_success] * pad_length

# --- Training Function ---
def train_and_test_agent(args):
    """
    args: (agent_type, seed, total_timesteps, log_interval)
    Trains the model and returns: (agent_type, eval_timesteps, test_success_series)
    """
    agent_type, seed, total_timesteps, log_interval = args
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
    env = DummyVecEnv([make_train_env])
    callback = PeriodicEvaluationCallback(log_interval=log_interval, num_train_episodes=1, num_test_episodes=1, verbose=1)
    model = PPO(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        verbose=0,
        seed=seed
    )
    logging.info("Starting training")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    full_length = total_timesteps // log_interval
    callback.get_padded_results(full_length)
    return agent_type, callback.eval_timesteps, callback.test_success

# --- Main function ---
def main():
    ensemble_size = 3
    n_workers = 6
    total_timesteps = 5 * 10**5
    log_interval = int(0.2 * total_timesteps)
    agent_types = ["ppo", "ppo-harmonic"]
    jobs = [(agent_type, seed, total_timesteps, log_interval)
            for agent_type in agent_types
            for seed in range(ensemble_size)]
    logging.info('Starting ensemble training')
    with Pool(processes=n_workers) as pool:
        results_list = pool.map(train_and_test_agent, jobs)
    results = {agent_type: [] for agent_type in agent_types}
    for agent_type, timesteps, test_success in results_list:
        results[agent_type].append((timesteps, test_success))
    plt.figure(figsize=(10, 6))
    for agent_type, curves in results.items():
        all_test_success = np.array([curve[1] for curve in curves])
        mean_test = np.mean(all_test_success, axis=0)
        min_test = np.min(all_test_success, axis=0)
        max_test = np.max(all_test_success, axis=0)
        x = curves[0][0]
        plt.plot(x, mean_test, label=agent_type.upper())
        plt.fill_between(x, min_test, max_test, alpha=0.3)
    plt.xlabel("Timesteps")
    plt.ylabel("Test Success Rate")
    plt.title("Generalization Learning Curve: PPO vs PPO-Harmonic")
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])
    plt.savefig(f'data/doorkey{GRID_SIZE}_learning_curves.png')
    plt.show()

if __name__ == "__main__":
    main()