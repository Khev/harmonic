import os
import pickle
import time
import gymnasium as gym
import minigrid
import numpy as np
import torch
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Pool
import argparse

from utils.utils_custom_policies import HarmonicPolicy

# Set up logging with timestamps
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")

# ANSI color codes for logging
GREEN = "\033[32m"
RED = "\033[31m"  # Red color code
MAGENTA = "\033[35m"
RESET = "\033[0m"

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
TRAIN_SEEDS = list(range(12))  # 12 train seeds
TEST_SEEDS = list(range(20, 32))  # 12 test seeds

def make_env(env_name, seed=None):
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=int(seed))
    env = CustomFlattenObservation(env)
    return env

# --- Evaluation Functions ---
def evaluate_on_env(model, env_name, seeds, num_episodes_per_seed=1):
    """Evaluate the model on the given environment and return the success rate."""
    successes, total_episodes = 0, 0
    best_reward = -float('inf')
    best_model = None
    
    for s in seeds:
        for _ in range(num_episodes_per_seed):
            env = make_env(env_name, seed=s)
            env = DummyVecEnv([lambda: env])
            obs = env.reset()
            done, total_reward = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = env.step(action)
                total_reward += rewards[0]
                done = dones[0]
            if total_reward > best_reward:
                best_reward = total_reward
                best_model = model
            if total_reward > 0.0:
                successes += 1
            total_episodes += 1
    
    return successes / total_episodes, best_model

# --- Custom Callback for Periodic Evaluation ---
class PeriodicEvaluationCallback(BaseCallback):
    def __init__(self, env_name, log_interval, num_train_episodes=5, num_test_episodes=5, 
                 early_stopping=True, patience=3, verbose=1):
        super().__init__(verbose)
        self.env_name = env_name
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
            train_acc, _ = evaluate_on_env(self.model, self.env_name, TRAIN_SEEDS, self.num_train_episodes)
            test_acc, _ = evaluate_on_env(self.model, self.env_name, TEST_SEEDS, self.num_test_episodes)
            self.eval_timesteps.append(self.num_timesteps)
            self.train_success.append(train_acc)
            self.test_success.append(test_acc)
            logging.info(f"Timestep {self.num_timesteps}: Train Success = {train_acc:.2f}, Test Success = {test_acc:.2f}")

            if self.early_stopping and test_acc >= 0.95:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.patience:
                    GREEN = "\033[32m"
                    RESET = "\033[0m"
                    logging.info(f"{GREEN}Early stopping triggered at timestep {self.num_timesteps}{RESET}")
                    return False
        return True

# --- Training Function ---
def train_and_test_agent(args):
    (
        env_name, 
        agent_type, 
        seed, 
        total_timesteps, 
        log_interval, 
        harmonic_exponent,
        distance_norm  # <-- Include distance_norm in function signature
    ) = args
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    policy_kwargs = dict(net_arch=[128, 128, 128])
    
    if agent_type == "ppo-harmonic":
        # Include distance_norm in the policy kwargs
        policy_kwargs.update(
            distance_norm=distance_norm,           # <-- main addition
            harmonic_exponent=harmonic_exponent,
            weight_reg=0.01
        )
        policy = HarmonicPolicy
    else:
        policy = "MlpPolicy"
    
    env = DummyVecEnv([lambda: make_env(env_name)])
    callback = PeriodicEvaluationCallback(env_name, log_interval=log_interval, 
                                          num_train_episodes=5, 
                                          num_test_episodes=5, 
                                          verbose=1)

    model = PPO(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        verbose=0,
        ent_coef=0.1,
        seed=seed
    )

    logging.info(f"Starting training for {agent_type} on {env_name} with seed {seed} (distance_norm={distance_norm})")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    model_dir = f"models/{env_name}/{agent_type}_{seed}"
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    model.save(model_dir)
    
    return env_name, agent_type, callback.eval_timesteps, callback.train_success, callback.test_success

# --- Main Multi-Environment Function ---
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PPO/PPO-Harmonic on multiple BabyAI/MiniGrid envs.")
    parser.add_argument('--ensemble_size', type=int, default=4, help='Number of ensemble seeds per agent type (default: 4)')
    parser.add_argument('--total_timesteps', type=int, default=5*10**7, help='Total training timesteps per agent (default: 1e6)')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of parallel workers (default: 6)')
    parser.add_argument('--harmonic_exponent', type=int, default=128, help='Harmonic exponent for PPO-Harmonic (default: 128)')
    # Add distance_norm argument with default="L2"
    parser.add_argument('--distance_norm', type=str, default='L2', help='Distance norm for PPO-Harmonic (default: L2)')

    args = parser.parse_args()

    # envs = [
    #     "MiniGrid-LavaCrossingS9N1-v0",
    #     "MiniGrid-LavaCrossingS9N2-v0",
    #     "MiniGrid-LavaCrossingS9N3-v0",
    #     "MiniGri[d-LavaCrossingS11N5-v0"
    # ][1:]


    envs = [
        # 'BabyAI-KeyCorridorS3R3-v0',
        #'BabyAI-KeyCorridorS4R3-v0',  # no success
        'MiniGrid-DoorKey-8x8-v0',  # Corrected from "MiniGri[d..." and aligned with prior suggestion
        #'BabyAI-UnlockPickup-v0'     # Added as the fourth similar environment
    ]


    # Store hyperparams, including distance_norm
    hyperparams = {
        "ensemble_size": args.ensemble_size,
        "total_timesteps": args.total_timesteps,
        "n_workers": args.n_workers,
        "harmonic_exponent": args.harmonic_exponent,
        "distance_norm": args.distance_norm,  # <-- store the new arg
    }

    agent_types = ["ppo", "ppo-harmonic"]
    log_interval = int(0.01 * args.total_timesteps)
    #log_interval = int(0.5 * args.total_timesteps)

    for env_name in envs:
        jobs = [
            (
                env_name, 
                agent_type, 
                seed, 
                args.total_timesteps, 
                log_interval, 
                args.harmonic_exponent,
                args.distance_norm  # <-- pass distance_norm through to train_and_test_agent
            )
            for agent_type in agent_types
            for seed in range(args.ensemble_size)
        ]

        logging.info(f"{MAGENTA}\n\n=== Starting ensemble training on {env_name} ==={RESET}")
        start_time = time.time()

        with Pool(processes=args.n_workers) as pool:
            results_list = pool.map(train_and_test_agent, jobs)

        # Save results in correct format and compute mean +- std for each trial
        all_results = {agt: {'train_acc': [], 'test_acc': [], 'time_steps':[]} for agt in agent_types}
        for e_name, agt, timesteps, train_acc, test_acc in results_list:
            #all_results[agt]['time_steps'].append(timesteps)
            all_results[agt]['train_acc'].append(train_acc)
            all_results[agt]['test_acc'].append(test_acc)

        # Save the full time series of train and test accuracies
        results_path = f'models/{env_name}/learning_data.pkl'
        hyperparams_path = f'models/{env_name}/hyperparams.pkl'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)
        with open(hyperparams_path, 'wb') as f:
            pickle.dump(hyperparams, f)

        # Calculate mean and std for train and test accuracies
        for agt in agent_types:
            train_acc_mean = np.mean(all_results[agt]['train_acc'], axis=0)
            train_acc_std = np.std(all_results[agt]['train_acc'], axis=0)
            test_acc_mean = np.mean(all_results[agt]['test_acc'], axis=0)
            test_acc_std = np.std(all_results[agt]['test_acc'], axis=0)

            logging.info(f"\n{RED}{env_name} - {agt.upper()}: Train Acc = {train_acc_mean[-1]:.2f} ± {train_acc_std[-1]:.2f}, Test Acc = {test_acc_mean[-1]:.2f} ± {test_acc_std[-1]:.2f}{RESET}\n")

        logging.info(f"Saved results to {results_path} and hyperparams to {hyperparams_path}")

        end_time = time.time()
        total_time = int(end_time - start_time)
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        logging.info(f"{GREEN}[{env_name}] Training time: {hours:02}:{minutes:02}:{seconds:02} {RESET}")


if __name__ == "__main__":
    main()
