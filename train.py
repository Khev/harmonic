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

import gymnasium as gym

# Agents
from agents.pg import PG
from agents.pg_cnn import PG_CNN
from agents.a2c import A2C 
from agents.ppo import PPO
from stable_baselines3 import PPO as PPO_sb3
from utils.utils_agents import HarmonicPolicy

from colorama import Fore, Style
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from utils.utils import filter_kwargs

def print_parameters(params):
    print(Fore.CYAN + "----------------")
    print(Fore.CYAN + "Parameters")
    print(Fore.CYAN + "----------------" + Style.RESET_ALL)
    for key, value in params.items():
        print(f"{key}: {value}")
    print("\n")



def get_agent(agent, env, **kwargs):
    # Filter out 'agent', 'env', and 'env_name' from kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['agent', 'env', 'env_name']}
    if agent == 'pg':
        return PG(env, **filter_kwargs(PG.__init__, filtered_kwargs))
    elif agent == 'pg_cnn':
        return PG_CNN(env, **filter_kwargs(PG_CNN.__init__, filtered_kwargs))
    elif agent == 'a2c':
        return A2C(env, **filter_kwargs(A2C.__init__, filtered_kwargs))
    elif agent == 'ppo':
        return PPO(env, **filter_kwargs(A2C.__init__, filtered_kwargs))
    elif agent == 'ppo-sb3':
        return  PPO_sb3(policy=HarmonicPolicy, env=env,
    policy_kwargs=dict(
        distance_norm="L2",
        harmonic_exponent=4,
        weight_reg=0.01
    )
)
    else:
        raise ValueError(f"Unsupported agent type: {agent}")


def get_env(env_name):

    if env_name == 'grid_world_hard':
        env_train = gridWorld(
            grid_size=(10, 10),
            possible_starts=[(0, 0), (0, 9), (9, 0), (5, 5), (2, 7)],
            possible_goals=[(9, 9), (9, 1), (1, 9), (7, 2), (4, 4)],
            obstacles={(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)},
            max_steps=50,
            randomize=True
        )
        env_test = gridWorld(
            grid_size=(10, 10),
            possible_starts=[(1, 1), (1, 8), (8, 1), (6, 6), (3, 8)],
            possible_goals=[(8, 8), (8, 2), (2, 8), (6, 3), (5, 5)],
            obstacles={(1, 7), (2, 5), (7, 3), (8, 6), (4, 8)},
            max_steps=50,
            randomize=True
        )

    elif env_name == 'grid_world':
        difficulty = 'hard'
        env_train = gridWorld(difficulty=difficulty)
        env_test = gridWorld(difficulty=difficulty)

    elif env_name == 'grid_world_partial':
        env_train = gridWorldPartial()
        env_test = gridWorldPartial()

    elif env_name == 'maze_world':
        #mode = 'easy_medium'
        mode = 'medium'
        env_train, env_test = MazeEnv(mode=mode), MazeEnv(mode=mode)

    elif env_name == 'maze_world_random':
        mode = 'medium'
        env_train, env_test = MazeEnvRandom(mode=mode, train=True), MazeEnvRandom(mode=mode, train=False)


    elif env_name == 'minigrid':
        env_train, env_test = gym.make("MiniGrid-Empty-5x5-v0"), gym.make("MiniGrid-Empty-5x5-v0")

    return env_train, env_test



class RewardCallback:
    def __init__(self, env_train, env_eval, model, log_interval=100, eval_interval=500, save_dir=".", layer_type="harmonic", distance_norm="L2", verbose=1, early_stopping=True):
        self.env_train = env_train
        self.env_eval = env_eval
        self.model = model
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.layer_type = layer_type
        self.distance_norm = distance_norm
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.early_stop = False  # Early stopping flag
        self.episode_rewards = []
        self.train_acc = []
        self.test_acc = []
        self.current_episode = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _on_step(self, reward):
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
                self.early_stop = True  # Set early stop flag
                save_name = os.path.join(self.save_dir, f"{self.layer_type}_{self.distance_norm}.pth")                
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(self.model.state_dict(), save_name)
                print(Fore.YELLOW + f"Model saved at: {save_name}" + Style.RESET_ALL)
                return False  



    def _compute_accuracy(self, env):
        successes = 0
        total = 100
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
        return self.episode_rewards, self.train_acc, self.test_acc


def pad_to_length(arr, length):
    """Pad array with last value to the target length."""
    if len(arr) < length:
        return np.pad(arr, (0, length - len(arr)), 'edge')
    return arr


def plot_ensemble_curves(ensemble_train, ensemble_test, save_dir, log_interval, Ntrain, layer_type):
    max_len = Ntrain // log_interval  # Explicitly set max length to Ntrain/log_interval
    ensemble_train = np.array([pad_to_length(arr, max_len) for arr in ensemble_train])
    ensemble_test = np.array([pad_to_length(arr, max_len) for arr in ensemble_test])

    mean_train = np.mean(ensemble_train, axis=0)
    min_train = np.min(ensemble_train, axis=0)
    max_train = np.max(ensemble_train, axis=0)
    mean_test = np.mean(ensemble_test, axis=0)
    min_test = np.min(ensemble_test, axis=0)
    max_test = np.max(ensemble_test, axis=0)

    steps = np.arange(len(mean_train)) * log_interval  # X-axis as Nstep

    plt.figure(figsize=(10, 6))
    plt.fill_between(steps, min_train, max_train, color='blue', alpha=0.2)
    plt.fill_between(steps, min_test, max_test, color='red', alpha=0.2)
    plt.plot(steps, mean_train, label='Mean Train Accuracy', color='blue')
    plt.plot(steps, mean_test, label='Mean Test Accuracy', color='red')
    plt.xlabel('Nstep')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{layer_type}')
    plt.legend()
    plt.ylim([0,105])
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'ensemble_accuracy_curves_{layer_type}_{Ntrain}.png'))  # Updated filename
    plt.show()


def save_results(results, save_dir, agent, layer_type):
    results_path = os.path.join(save_dir, f"{agent}_{layer_type}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f)


def plot_actions_and_states(model, env, layer_type='harmonic', save_dir='.', save_fig=True):
    """
    Plots both the per-state embeddings (via model.mlp) and,
    if layer_type='harmonic' or 'softmax', the action embeddings in the same PCA space.

    The difference is:
    - "harmonic": we read `model.weights` (shape [num_actions, hidden_dim])
    - "softmax":  we read `model.output_layer.weight` (shape [num_actions, hidden_dim])

    Args:
        model: PG model with:
               - model.mlp (states -> hidden_dim)
               - If 'harmonic', model.weights (action centers)
               - If 'softmax', model.output_layer.weight
        env:   A small gridWorld environment.
        layer_type: one of {"softmax", "harmonic"}.
        save_dir: directory to save the figure
        save_fig: if True, saves a PNG in `save_dir`.
    """

    # For labeling actions if we know it's a 4-action grid
    moves = {
        0: "Left",
        1: "Right",
        2: "Down",
        3: "Up"
    }

    # 1. Embed all states
    H, W = env.grid_size
    all_states = []
    coords = []
    for r in range(H):
        for c in range(W):
            idx = env.get_state_index((r, c))
            one_hot = env.to_one_hot(idx)
            all_states.append(one_hot)
            coords.append((r, c))

    all_states_t = torch.tensor(all_states, dtype=torch.float32)
    with torch.no_grad():
        # shape=[num_states, hidden_dim]
        state_features = model.mlp(all_states_t)
    state_features_np = state_features.cpu().numpy()

    # 2. Extract action embeddings, if possible
    action_features_np = None

    if layer_type == 'harmonic':
        # shape = [num_actions, hidden_dim]
        action_features_np = model.weights.detach().cpu().numpy()

    elif layer_type == 'softmax':
        # We assume your final linear layer is called `output_layer`
        # with shape = [num_actions, hidden_dim]. If it's reversed, do .T
        if hasattr(model, 'output_layer'):
            action_features_np = model.output_layer.weight.detach().cpu().numpy()
        else:
            print("Warning: no `output_layer` found—cannot plot action embeddings for softmax.")

    # 3. Combine for PCA
    if action_features_np is not None:
        combined = np.vstack([state_features_np, action_features_np])
    else:
        combined = state_features_np

    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    n_states = state_features_np.shape[0]
    states_2d = combined_2d[:n_states]
    actions_2d = None
    if action_features_np is not None:
        actions_2d = combined_2d[n_states:]

    # 4. Plot everything
    plt.figure(figsize=(7,7))
    plt.scatter(states_2d[:, 0], states_2d[:, 1],
                color='gray', alpha=0.7, label='States')

    # Label each state with (row,col)
    for i, (r, c) in enumerate(coords):
        plt.text(states_2d[i, 0], states_2d[i, 1],
                 f"({r},{c})", fontsize=6, alpha=0.7)

    # Plot action embeddings if we have them
    if actions_2d is not None:
        num_actions = actions_2d.shape[0]
        sc = plt.scatter(actions_2d[:, 0], actions_2d[:, 1],
                         c=range(num_actions), cmap='rainbow',
                         edgecolors='black', s=80, label='Actions')
        for i in range(num_actions):
            action_label = moves.get(i, f"A{i}")
            plt.text(actions_2d[i, 0], actions_2d[i, 1],
                     f"{action_label}", fontsize=9, ha='center', va='bottom')

    plt.title(f"State & Action Embeddings (PCA) - {layer_type}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)

    # Optionally save
    if save_fig:
        outpath = os.path.join(save_dir, f"state_action_embeddings_{layer_type}.png")
        plt.savefig(outpath, dpi=150)
        print(f"Saved embeddings figure to: {outpath}")

    plt.show()


def run_training(args, seed=None):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env_train, env_test = get_env(args.env_name)
    model = get_agent(args.agent, env_train, **{k: v for k, v in vars(args).items() if k not in ['agent', 'env_name']})
    callback = RewardCallback(env_train, env_test, model, 
                              log_interval=args.log_interval, 
                              eval_interval=args.log_interval, 
                              save_dir='models/', 
                              layer_type=args.layer_type, 
                              distance_norm=args.distance_norm)
    model.learn(total_timesteps=args.Ntrain, callback=callback)
    model.save(os.path.join(args.save_dir, f"{args.agent}_{args.layer_type}_model_{seed}.pth"))
    _, train_acc, test_acc = callback.get_results()
    return train_acc, test_acc



def main(args):
    print("\nStarting Training with the following configuration:")
    print_parameters(vars(args))

    if args.parallel:
        ctx = get_context('spawn')
        with ctx.Pool(args.num_workers) as pool:
            results = pool.starmap(run_training, [(args, i) for i in range(args.ensemble_size)])
    else:
        results = [run_training(args, i) for i in range(args.ensemble_size)]

    save_results(results, args.save_dir, args.agent, args.layer_type)
    ensemble_train = [res[0] for res in results]
    ensemble_test = [res[1] for res in results]
    plot_ensemble_curves(ensemble_train, ensemble_test, args.save_dir, args.log_interval, args.Ntrain, args.layer_type)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PG agent on gridWorld with ensemble and multiprocessing.")
    parser.add_argument('--env_name', type=str, default='maze_world', help='Env name')
    parser.add_argument('--agent', type=str, default='pg', help='Agent type.')
    parser.add_argument('--Ntrain', type=int, default=5*10**4, help='Number of training timesteps per agent.')
    parser.add_argument('--layer_type', type=str, choices=['softmax', 'harmonic'], default='harmonic', help='Layer type.')
    parser.add_argument('--distance_norm', type=str, choices=['L1', 'L2'], default='L2', help='Distance norm for harmonic layer.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in the PG agent.')
    parser.add_argument('--harmonic_exponent', type=int, default=4, help='Number of layers in the PG agent.')
    parser.add_argument('--ensemble_size', type=int, default=1, help='Number of ensemble agents.')
    parser.add_argument('--save_dir', type=str, default='./data', help='Directory to save results.')
    parser.add_argument('--parallel', action='store_true', help='Flag to use parallel training.')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of parallel workers.')
    parser.add_argument('--log_interval', type=int, default=None, help='Logging interval.')

    args = parser.parse_args()
    if args.log_interval is None:
        args.log_interval = int(0.1 * args.Ntrain)
    #args.save_dir = os.path.join(args.save_dir, args.agent)
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
