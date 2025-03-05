import os, pickle, time
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
import argparse

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
TRAIN_SEEDS = list(range(12))  # 12 train seeds
TEST_SEEDS = list(range(20, 32))  # 12 test seeds

def parse_env_name(env_name):
    """
    Parse the environment name.
    If env_name is of the format 'EnvType-GridSize' (e.g., 'DoorKey-8x8'),
    returns (EnvType, GridSize).
    Otherwise, returns (env_name, None) assuming env_name is a full gym environment id.
    """
    parts = env_name.split('-')
    if len(parts) == 2 and 'x' in parts[1]:
        env_type, grid_size_str = parts
        try:
            width, height = map(int, grid_size_str.split('x'))
            if width <= 0 or height <= 0:
                raise ValueError(f"Grid dimensions must be positive integers, got {width}x{height}")
        except ValueError as e:
            raise ValueError(f"Grid size must be in 'WxH' format (e.g., '8x8'), got {grid_size_str}: {str(e)}")
        return env_type, grid_size_str
    else:
        # Assume full gym environment id
        return env_name, None

def make_env(env_name, seed=None):
    """
    Create a gym environment with a specific name and seed.
    
    If env_name follows the 'EnvType-GridSize' format (e.g., 'DoorKey-8x8'),
    it is converted to the corresponding MiniGrid environment id (e.g., 'MiniGrid-DoorKey-8x8-v0').
    Otherwise, env_name is assumed to be a full gym environment id.
    """
    env_type, grid_size_str = parse_env_name(env_name)
    if grid_size_str is not None:
        gym_env_name = f"MiniGrid-{env_type}-{grid_size_str}-v0"
    else:
        gym_env_name = env_type
    env = gym.make(gym_env_name)
    if seed is not None:
        env.reset(seed=int(seed))
    env = CustomFlattenObservation(env)
    return env

# --- Evaluation Functions ---
def evaluate_on_env(model, env_name, seeds, num_episodes_per_seed=5):
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
        self.best_train_model = None
        self.best_test_model = None
        self.best_train_reward = -float('inf')
        self.best_test_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval < self.training_env.num_envs:
            train_acc, train_model = evaluate_on_env(self.model, self.env_name, TRAIN_SEEDS, self.num_train_episodes)
            test_acc, test_model = evaluate_on_env(self.model, self.env_name, TEST_SEEDS, self.num_test_episodes)
            self.eval_timesteps.append(self.num_timesteps)
            self.train_success.append(train_acc)
            self.test_success.append(test_acc)
            logging.info(f"Timestep {self.num_timesteps}: Train Success = {train_acc:.2f}, Test Success = {test_acc:.2f}")

            if train_acc > self.best_train_reward:
                self.best_train_reward = train_acc
                self.best_train_model = train_model
            if test_acc > self.best_test_reward:
                self.best_test_reward = test_acc
                self.best_test_model = test_model

            if self.early_stopping and test_acc >= 0.95:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.patience:
                    logging.info(f"Early stopping triggered at timestep {self.num_timesteps}")
                    return False
        return True

    def save_best_models(self, env_name, agent_type):
        models_dir = os.path.join('models', env_name)
        os.makedirs(models_dir, exist_ok=True)
        if self.best_train_model:
            self.best_train_model.save(os.path.join(models_dir, f'{agent_type}_train_best.pth'))
            logging.info(f"Saved best train model for {agent_type} on {env_name}")
        if self.best_test_model:
            self.best_test_model.save(os.path.join(models_dir, f'{agent_type}_test_best.pth'))
            logging.info(f"Saved best test model for {agent_type} on {env_name}")

# --- Training Function ---
def train_and_test_agent(args):
    env_name, agent_type, seed, total_timesteps, log_interval = args
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    policy_kwargs = dict(net_arch=[128, 128, 128])
    policy = HarmonicPolicy if agent_type == "ppo-harmonic" else "MlpPolicy"
    
    env = DummyVecEnv([lambda: make_env(env_name)])
    callback = PeriodicEvaluationCallback(env_name, log_interval=log_interval, num_train_episodes=5, num_test_episodes=5, verbose=1)

    model = PPO(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        verbose=0,
        ent_coef=0.1,
        seed=seed
    )

    logging.info(f"Starting training for {agent_type} on {env_name} with seed {seed}")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    callback.save_best_models(env_name, agent_type)
    return env_name, agent_type, callback.eval_timesteps, callback.test_success

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PPO/PPO-Harmonic on any MiniGrid environment.")
    parser.add_argument('--env_name', type=str, default='BabyAI-KeyCorridorS3R3-v0')
    parser.add_argument('--ensemble_size', type=int, default=12, help='Number of ensemble seeds per agent type (default: 6)')
    parser.add_argument('--total_timesteps', type=int, default=2*10**7, help='Total training timesteps per agent (default: 10**5)')
    parser.add_argument('--log_interval', type=int, default=None, help='Evaluation interval in timesteps (default: 1000)')
    parser.add_argument('--n_workers', type=int, default=6, help='Number of parallel workers (default: 6)')
    args = parser.parse_args()

    if args.log_interval == None:
        args.log_interval = int(0.01*args.total_timesteps)

    agent_types = ["ppo", "ppo-harmonic"]
    jobs = [(args.env_name, agent_type, seed, args.total_timesteps, args.log_interval)
            for agent_type in agent_types
            for seed in range(args.ensemble_size)]
    
    logging.info(f'\nStarting ensemble training on {args.env_name} with {args.ensemble_size} seeds per agent type')

    start_time = time.time()

    with Pool(processes=args.n_workers) as pool:
        results_list = pool.map(train_and_test_agent, jobs)

    # Save results
    results_path = f'figures/{args.env_name}_results.pkl'
    os.makedirs('figures', exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results_list, f)
    logging.info(f"Saved results to {results_path}")


    # Plot results
    end_time = time.time()  # End timing
    total_time = end_time - start_time
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Total time: {hours:02}:{minutes:02}:{seconds:02}")  # Log final execution time

    results = {agent_type: [] for agent_type in agent_types}
    for env_name, agent_type, timesteps, test_success in results_list:
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
    plt.title(f"Generalization Learning Curve: {args.env_name}")
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{args.env_name}_learning_curves.png')
    plt.show()


if __name__ == "__main__":
    main()
