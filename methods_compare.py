import argparse
import numpy as np
import torch
import os, json
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool, get_context

# Envs
from envs.env_gridworld import gridWorld
from envs.env_gridworld_partial import gridWorldPartial
from envs.env_colormaze import colorMazeEnv
from envs.env_mazeworld import MazeEnv
from envs.env_mazeworld_random import MazeEnvRandom

# Agents
from agents.pg import PG
from agents.pg_cnn import PG_CNN
from agents.a2c import A2C 
from agents.ppo import PPO
from stable_baselines3 import PPO as PPO_sb3
from utils.utils_agents import HarmonicPolicy

from colorama import Fore, Style
import torch
import numpy as np
import os


def print_parameters(params):
    print(Fore.CYAN + "----------------")
    print(Fore.CYAN + "Parameters")
    print(Fore.CYAN + "----------------" + Style.RESET_ALL)
    for key, value in params.items():
        print(f"{key}: {value}")
    print("\n")


def get_agent(agent, env, **kwargs):
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['agent', 'env', 'env_name']}
    if agent == 'pg':
        return PG(env, **filtered_kwargs)
    elif agent == 'pg_cnn':
        return PG_CNN(env, **filtered_kwargs)
    elif agent == 'a2c':
        return A2C(env, **filtered_kwargs)
    elif agent == 'ppo':
        return PPO(env, **filtered_kwargs)
    elif agent == 'ppo-sb3':
        return PPO_sb3("MlpPolicy", env=env)
    else:
        raise ValueError(f"Unsupported agent type: {agent}")


def get_env(env_name, mode='easy'):
    if env_name == 'maze_world':
        return MazeEnv(mode=mode), MazeEnv(mode=mode)
    elif env_name == 'maze_world_random':
        mode = 'medium'
        env_train, env_test = MazeEnvRandom(mode=mode, train=True), MazeEnvRandom(mode=mode, train=False)
        return env_train, env_test
    else:
        raise ValueError(f"Unknown env_name: {env_name}")


class RewardCallback:
    def __init__(self, env_train, env_eval, model, log_interval=100, eval_interval=500, save_dir=".", verbose=1, early_stopping=True):
        self.env_train = env_train
        self.env_eval = env_eval
        self.model = model
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.early_stop = False
        self.episode_rewards = []
        self.train_acc = []
        self.test_acc = []
        self.current_episode = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _on_step(self, reward):
        """
        Called after every environment step.
        """
        self.episode_rewards.append(reward)
        self.current_episode += 1
        if self.current_episode % self.log_interval == 0:
            train_acc = self._compute_accuracy(self.env_train)
            self.train_acc.append(train_acc)
            test_acc = self._compute_accuracy(self.env_eval)
            self.test_acc.append(test_acc)
            print(Fore.WHITE + f"[{datetime.now().strftime('%H:%M:%S')}] Episode {self.current_episode}: acc_train, acc_test: {train_acc:.2f}%, {test_acc:.2f}%" + Style.RESET_ALL)
            if self.early_stopping and test_acc == 100.0:
                print(Fore.GREEN + f"Early stopping triggered at episode {self.current_episode} with test accuracy of 100%!" + Style.RESET_ALL)
                self.early_stop = True
                return False

    def _compute_accuracy(self, env):
        """
        Run multiple rollouts to measure success rate.
        """
        successes = 0
        total = 10
        for _ in range(total):
            state = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action = self.model.predict(state, deterministic=True)
                state, reward, done, info = env.step(action)
                if info.get('info') == "Goal Reached":
                    successes += 1
        return (successes / total) * 100

    def get_results(self):
        """
        Return reward history and logged accuracies.
        """
        return self.episode_rewards, self.train_acc, self.test_acc


def pad_to_length(arr, length):
    if len(arr) < length:
        return np.pad(arr, (0, length - len(arr)), 'edge')
    return arr


def run_training(seed, args):
    """
    Runs training for a single agent instance with a given seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_train, env_eval = get_env(args.env_name, mode=args.mode)

    # Filter arguments to only pass those accepted by PG/A2C/PPO
    agent_kwargs = {k: v for k, v in vars(args).items() if k in PG.__init__.__code__.co_varnames}

    model = get_agent(args.agent, env_train, **agent_kwargs)

    callback = RewardCallback(env_train, env_eval, model, log_interval=args.log_interval)
    model.learn(total_timesteps=args.Ntrain, callback=callback)
    _, _, train_acc = callback.get_results()
    return train_acc



def main(args):
    print("\nStarting Training with the following configuration:")
    print_parameters(vars(args))

    approach_list = [
        ("Softmax", "softmax", "L2"),
        ("Harmonic (L2)", "harmonic", "L2"),
        ("Harmonic (L1)", "harmonic", "L1")
    ]
    
    results = {}

    for label, ltype, dist in approach_list:
        print(Fore.MAGENTA + f"\n=== Running experiment: {label} ===" + Style.RESET_ALL)
        args.layer_type = ltype
        args.distance_norm = dist

        if args.parallel:
            print(Fore.YELLOW + f"Running {args.ensemble_size} parallel training runs for {label}..." + Style.RESET_ALL)
            with get_context("spawn").Pool(processes=args.ensemble_size) as pool:
                ensemble_train_acc = pool.starmap(run_training, [(seed, args) for seed in range(args.ensemble_size)])
        else:
            print(Fore.YELLOW + f"Running {args.ensemble_size} sequential training runs for {label}..." + Style.RESET_ALL)
            ensemble_train_acc = [run_training(seed, args) for seed in range(args.ensemble_size)]

        # Compute mean, min, and max across ensemble
        max_len = args.Ntrain // args.log_interval
        ensemble_train_acc = np.array([pad_to_length(arr, max_len) for arr in ensemble_train_acc])

        mean_train_acc = np.mean(ensemble_train_acc, axis=0)
        min_train_acc = np.min(ensemble_train_acc, axis=0)
        max_train_acc = np.max(ensemble_train_acc, axis=0)

        results[label] = (mean_train_acc, min_train_acc, max_train_acc)

    # Plot all three curves with shaded regions
    steps = np.arange(len(mean_train_acc)) * args.log_interval
    plt.figure(figsize=(10, 6))

    colors = {'Softmax': 'blue', 'Harmonic (L2)': 'red', 'Harmonic (L1)': 'green'}

    for label, (mean_train_acc, min_train_acc, max_train_acc) in results.items():
        plt.plot(steps, mean_train_acc, label=label, color=colors[label])
        plt.fill_between(steps, min_train_acc, max_train_acc, color=colors[label], alpha=0.2)

    plt.xlabel('Nstep')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Learning Curves for Different Methods')
    plt.legend()
    plt.ylim([0, 105])
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, "learning_curves.png"))
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PG agent on MazeEnv or gridWorld with ensemble averaging and multiprocessing.")
    parser.add_argument('--env_name', type=str, default='maze_world', help='Env name')
    parser.add_argument('--agent', type=str, default='pg', help='Agent type.')
    parser.add_argument('--Ntrain', type=int, default=5*10**4, help='Number of training timesteps per agent.')
    parser.add_argument('--layer_type', type=str, choices=['softmax', 'harmonic'], default='softmax', help='Layer type.')
    parser.add_argument('--distance_norm', type=str, choices=['L1', 'L2'], default='L2', help='Distance norm for harmonic layer.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in the PG agent.')
    parser.add_argument('--harmonic_exponent', type=int, default=4, help='Harmonic exponent.')
    parser.add_argument('--ensemble_size', type=int, default=5, help='Number of ensemble agents.')
    parser.add_argument('--save_dir', type=str, default='./data', help='Directory to save results.')
    parser.add_argument('--log_interval', type=int, default=None, help='Logging interval.')
    parser.add_argument('--mode', type=str, default='easy_medium', help='Mode for MazeEnv (e.g., "easy", "medium", "hard").')
    parser.add_argument('--parallel', action='store_true', help='Flag to enable parallel training.')

    args = parser.parse_args()

    if args.log_interval is None:
        args.log_interval = int(0.01 * args.Ntrain)
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
