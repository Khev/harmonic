import os, pickle
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
TRAIN_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Increased to 12 seeds
TEST_SEEDS = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]  # Increased to 12 seeds

def make_env(env_name, seed=None, train=True):
    """Create a MiniGrid environment with a specific name and seed."""
    # Parse env_name to extract type and grid size (e.g., "DoorKey-8x8" -> "DoorKey", "8x8")
    env_type, grid_size_str = parse_env_name(env_name)
    if train:
        env_id = f"MiniGrid-{env_type}-{grid_size_str}-v0"
    else:
        # Use the same grid size for test to ensure consistency (can adjust later)
        env_id = f"MiniGrid-{env_type}-{grid_size_str}-v0"
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=int(seed))
    env = CustomFlattenObservation(env)
    return env

def parse_env_name(env_name):
    """Parse the environment name to extract type and grid size (e.g., 'DoorKey-8x8' -> ('DoorKey', '8x8'))."""
    if '-' not in env_name:
        raise ValueError(f"Invalid env_name format: {env_name}. Expected format: 'EnvType-GridSize' (e.g., 'DoorKey-8x8')")
    env_type, size_str = env_name.split('-')
    try:
        # Ensure size_str is in 'WxH' format (e.g., "8x8")
        if 'x' not in size_str:
            raise ValueError(f"Grid size must be in 'WxH' format (e.g., '8x8'), got {size_str}")
        # Validate that width and height are integers
        width, height = map(int, size_str.split('x'))
        if width <= 0 or height <= 0:
            raise ValueError(f"Grid dimensions must be positive integers, got {width}x{height}")
        grid_size = size_str  # Keep as string "8x8" for environment ID
    except ValueError as e:
        raise ValueError(f"Grid size must be in 'WxH' format (e.g., '8x8'), got {size_str}: {str(e)}")
    return env_type, grid_size

# --- Evaluation Functions ---
def evaluate_on_env(model, env_name, seeds, num_episodes_per_seed=5):
    """
    Evaluate the model systematically on the specified seeds for the given environment.
    Returns the overall success rate and the model for potential saving.
    """
    successes = 0
    total_episodes = 0
    best_reward = -float('inf')  # Track best reward for model saving
    best_model = None
    for s in seeds:
        for _ in range(num_episodes_per_seed):
            env = make_env(env_name, seed=s, train=False)
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
                best_model = model  # Save the model with the best reward
            if total_reward > 0.0:  # Assuming positive reward indicates goal reached
                successes += 1
            total_episodes += 1
    return successes / total_episodes, best_model

# --- Custom Callback for Periodic Evaluation ---
class PeriodicEvaluationCallback(BaseCallback):
    """
    A callback that evaluates the current model every log_interval timesteps
    and logs train/test success rates. Implements early stopping and tracks best models.
    """
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
            if self.verbose > 0:
                logging.info(f"Timestep {self.num_timesteps}: Train Success = {train_acc:.2f}, Test Success = {test_acc:.2f}")
            # Update best models based on rewards (assuming higher reward = better performance)
            if train_acc > self.best_train_reward:
                self.best_train_reward = train_acc
                self.best_train_model = train_model
            if test_acc > self.best_test_reward:
                self.best_test_reward = test_acc
                self.best_test_model = test_model
            if self.early_stopping:
                if test_acc >= 0.95:  # Adjusted for DoorKey's potential difficulty
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
                if self.stagnation_counter >= self.patience:
                    logging.info(f"Early stopping triggered at timestep {self.num_timesteps}")
                    return False
        return True

    def get_padded_results(self, full_length):
        """Pads evaluation data to match the expected full length."""
        pad_length = full_length - len(self.eval_timesteps)
        if pad_length > 0:
            last_timestep = self.eval_timesteps[-1]
            last_train_success = self.train_success[-1]
            last_test_success = self.test_success[-1]
            self.eval_timesteps += [last_timestep] * pad_length
            self.train_success += [last_train_success] * pad_length
            self.test_success += [last_test_success] * pad_length

    def save_best_models(self, env_name, agent_type):
        """Save the best models for train and test in the specified directory."""
        models_dir = os.path.join('models', env_name)
        os.makedirs(models_dir, exist_ok=True)
        if self.best_train_model is not None:
            self.best_train_model.save(os.path.join(models_dir, f'{agent_type}_train_best.pth'))
            logging.info(f"Saved best train model for {agent_type} on {env_name} to {models_dir}/{agent_type}_train_best.pth")
        if self.best_test_model is not None:
            self.best_test_model.save(os.path.join(models_dir, f'{agent_type}_test_best.pth'))
            logging.info(f"Saved best test model for {agent_type} on {env_name} to {models_dir}/{agent_type}_test_best.pth")

# --- Training Function ---
def train_and_test_agent(args):
    """
    args: (env_name, agent_type, seed, total_timesteps, log_interval)
    Trains the model and returns: (env_name, agent_type, eval_timesteps, test_success_series)
    """
    env_name, agent_type, seed, total_timesteps, log_interval = args
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy_kwargs = dict(net_arch=[128, 128, 128])
    if agent_type == "ppo-harmonic":
        policy_kwargs.update(
            distance_norm="L2",
            harmonic_exponent=128,  # Can be tuned
            weight_reg=0.01
        )
        policy = HarmonicPolicy
    else:
        policy = "MlpPolicy"
    env = DummyVecEnv([lambda: make_env(env_name, train=True)])
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
    full_length = total_timesteps // log_interval
    callback.get_padded_results(full_length)
    # Save the best models after training
    callback.save_best_models(env_name, agent_type)
    return env_name, agent_type, callback.eval_timesteps, callback.test_success

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PPO/PPO-Harmonic on MiniGrid environments.")
    parser.add_argument('--env_name', type=str, default='DoorKey-8x8', help='Environment name (e.g., DoorKey-5x5, DoorKey-8x8, MultiRoom-6x6)')
    parser.add_argument('--ensemble_size', type=int, default=12, help='Number of ensemble seeds per agent type (default: 24)')
    parser.add_argument('--total_timesteps', type=int, default=5000000, help='Total training timesteps per agent (default: 7.5e6)')
    parser.add_argument('--log_interval', type=int, default=None, help='Evaluation interval in timesteps (default: 0.005 * total_timesteps)')
    parser.add_argument('--n_workers', type=int, default=6, help='Number of parallel workers (default: 6)')
    args = parser.parse_args()

    if args.log_interval is None:
        args.log_interval = int(0.005 * args.total_timesteps)
        args.log_interval = 1000

    agent_types = ["ppo", "ppo-harmonic"]
    jobs = [(args.env_name, agent_type, seed, args.total_timesteps, args.log_interval)
            for agent_type in agent_types
            for seed in range(args.ensemble_size)]
    logging.info(f'Starting ensemble training on {args.env_name} with {args.ensemble_size} seeds per agent type')

    with Pool(processes=args.n_workers) as pool:
        results_list = pool.map(train_and_test_agent, jobs)

    results = {agent_type: [] for agent_type in agent_types}
    for env_name, agent_type, timesteps, test_success in results_list:
        results[agent_type].append((timesteps, test_success))

    # Save the results dictionary
    results_path = os.path.join('figures', f'{args.env_name}_results.pkl')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"Saved results data to {results_path}")

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