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

# Agents
from agents.pg import PG
from agents.pg_cnn import PG_CNN
from agents.a2c import A2C 
from agents.ppo import PPO
from stable_baselines3 import PPO as PPO_sb3
from utils.utils_agents import HarmonicPolicy

from colorama import Fore, Style
from sklearn.decomposition import PCA
import torch
import numpy as np
import os

from utils.utils import filter_kwargs


def print_parameters(params):
    print(Fore.CYAN + "----------------")
    print(Fore.CYAN + "Parameters")
    print(Fore.CYAN + "----------------" + Style.RESET_ALL)
    for key, value in params.items():
        print(f"{key}: {value}")
    print("\n")


def get_agent(agent, env, **kwargs):
    """
    Creates the agent. For layer_type 'harmonic', we configure the HarmonicPolicy
    for PPO-sb3 if using that agent, otherwise use our PG.
    """
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['agent', 'env', 'env_name']}
    layer_type = filtered_kwargs.get('layer_type', 'softmax')
    distance_norm = filtered_kwargs.get('distance_norm', 'L2')

    if agent == 'pg':
        return PG(env, **filter_kwargs(PG.__init__, filtered_kwargs))
    elif agent == 'pg_cnn':
        return PG_CNN(env, **filter_kwargs(PG_CNN.__init__, filtered_kwargs))
    elif agent == 'a2c':
        return A2C(env, **filter_kwargs(A2C.__init__, filtered_kwargs))
    elif agent == 'ppo':
        return PPO(env, **filter_kwargs(A2C.__init__, filtered_kwargs))
    elif agent == 'ppo-sb3':
        if layer_type == 'harmonic':
            return PPO_sb3(
                policy=HarmonicPolicy, 
                env=env,
                policy_kwargs=dict(
                    distance_norm=distance_norm,
                    harmonic_exponent=filtered_kwargs.get('harmonic_exponent', 4),
                    weight_reg=0.01
                )
            )
        else:
            # Normal PPO with MlpPolicy
            return PPO_sb3("MlpPolicy", env=env)
    else:
        raise ValueError(f"Unsupported agent type: {agent}")


def get_env(env_name, mode='easy'):
    """
    Returns a (train_env, test_env) tuple based on env_name and mode.
    """
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
        env_train, env_test = MazeEnv(mode=mode), MazeEnv(mode=mode)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
    return env_train, env_test


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
                    action = self.model.predict(state, deterministic=False)
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
    """
    Pads the array to the given length.
    """
    if len(arr) < length:
        return np.pad(arr, (0, length - len(arr)), 'edge')
    return arr


def plot_ensemble_curves(ensemble_train, ensemble_test, save_dir, log_interval, Ntrain, approach_label):
    max_len = Ntrain // log_interval
    ensemble_train = np.array([pad_to_length(arr, max_len) for arr in ensemble_train])
    ensemble_test = np.array([pad_to_length(arr, max_len) for arr in ensemble_test])

    mean_train = np.mean(ensemble_train, axis=0)
    min_train = np.min(ensemble_train, axis=0)
    max_train = np.max(ensemble_train, axis=0)
    mean_test = np.mean(ensemble_test, axis=0)
    min_test = np.min(ensemble_test, axis=0)
    max_test = np.max(ensemble_test, axis=0)

    steps = np.arange(len(mean_train)) * log_interval
    plt.figure(figsize=(10, 6))
    plt.fill_between(steps, min_train, max_train, color='blue', alpha=0.2)
    plt.fill_between(steps, min_test, max_test, color='red', alpha=0.2)
    plt.plot(steps, mean_train, label='Mean Train Accuracy', color='blue')
    plt.plot(steps, mean_test, label='Mean Test Accuracy', color='red')
    plt.xlabel('Nstep')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{approach_label}')
    plt.legend()
    plt.ylim([0,100])
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'ensemble_accuracy_curves_{approach_label}_{Ntrain}.png'))
    plt.show()


def save_results(results, save_dir, agent, approach_label):
    results_path = os.path.join(save_dir, f"{agent}_{approach_label}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f)


def plot_actions_and_states_ax(model, env, approach_label='softmax', ax=None):
    """
    Plots the PCA of state embeddings and (optionally) action embeddings.
    Shows the explained variance of PC1 and PC2 in the subplot title.
    """
    moves = {0: "Left", 1: "Right", 2: "Down", 3: "Up"}
    H, W = env.grid_size

    # 1. Gather all states and their (row, col) coordinates
    all_states = []
    coords = []
    for r in range(H):
        for c in range(W):
            idx = env.get_state_index((r, c))
            one_hot = env.to_one_hot(idx)
            all_states.append(one_hot)
            coords.append((r, c))

    # 2. Forward pass through the MLP to get state features
    all_states_t = torch.tensor(all_states, dtype=torch.float32)
    with torch.no_grad():
        state_features = model.mlp(all_states_t)
    state_features_np = state_features.cpu().numpy()

    # 3. Gather action embeddings if harmonic or softmax
    action_features_np = None
    if approach_label.lower().startswith('harmonic'):
        if hasattr(model, 'weights'):
            action_features_np = model.weights.detach().cpu().numpy()
    elif approach_label.lower().startswith('softmax'):
        if hasattr(model, 'output_layer'):
            action_features_np = model.output_layer.weight.detach().cpu().numpy()

    # 4. Combine states & actions for PCA
    if action_features_np is not None:
        combined = np.vstack([state_features_np, action_features_np])
    else:
        combined = state_features_np

    # 5. PCA
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    pc_variance = pca.explained_variance_ratio_  # array of length 2

    n_states = state_features_np.shape[0]
    states_2d = combined_2d[:n_states]
    actions_2d = combined_2d[n_states:] if action_features_np is not None else None

    # 6. Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,7))

    # Plot states
    ax.scatter(states_2d[:, 0], states_2d[:, 1],
               color='gray', alpha=0.7, label='States')

    # Label each state with (row,col)
    for i, (r, c) in enumerate(coords):
        ax.text(states_2d[i, 0], states_2d[i, 1],
                f"({r},{c})", fontsize=6, alpha=0.7)

    # Plot action embeddings if available
    if actions_2d is not None:
        num_actions = actions_2d.shape[0]
        sc = ax.scatter(actions_2d[:, 0], actions_2d[:, 1],
                        c=range(num_actions), cmap='rainbow',
                        edgecolors='black', s=80, label='Actions')
        for i in range(num_actions):
            action_label = moves.get(i, f"A{i}")
            ax.text(actions_2d[i, 0], actions_2d[i, 1],
                    f"{action_label}", fontsize=9, ha='center', va='bottom')

    # 7. Update title with explained variance
    pc1_var = pc_variance[0] * 100.0
    pc2_var = pc_variance[1] * 100.0
    ax.set_title(f"{approach_label} (PC1={pc1_var:.1f}%, PC2={pc2_var:.1f}%)")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True)
    return ax



def plot_env_render(env, ax, title="Environment Render"):
    """
    Improved environment rendering in matplotlib that mimics env.render():
      - Obstacles (X) in red,
      - Start (S) in cyan,
      - Goal (G) in green,
      - Agent (A) in yellow,
      - Empty cells (.) in white.
    Each cell is labeled with its (row, col) coordinates.
    """
    H, W = env.grid_size
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(np.arange(W+1))
    ax.set_yticks(np.arange(H+1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.invert_yaxis()  # so row=0 is at the top
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.grid(color='black', linestyle='-', linewidth=1)

    obstacles = env.obstacles if hasattr(env, 'obstacles') else set()
    sr, sc = env.start
    gr, gc = env.goal
    cr, cc = env.state

    for r in range(H):
        for c in range(W):
            if (r, c) in obstacles:
                cell_label = 'X'
                facecolor = 'red'
            elif (r, c) == (sr, sc):
                cell_label = 'S'
                facecolor = 'cyan'
            elif (r, c) == (gr, gc):
                cell_label = 'G'
                facecolor = 'green'
            elif (r, c) == (cr, cc):
                cell_label = 'A'
                facecolor = 'yellow'
            else:
                cell_label = '.'
                facecolor = 'white'

            rect = plt.Rectangle((c, r), 1, 1, facecolor=facecolor, edgecolor='black')
            ax.add_patch(rect)
            ax.text(c+0.5, r+0.6, cell_label,
                    ha='center', va='center', fontsize=10, color='black')
            ax.text(c+0.5, r+0.25, f"({r},{c})",
                    ha='center', va='center', fontsize=6, color='black')

    return ax


def run_training(args, seed=None):
    """
    Trains a single agent with the specified layer_type and distance_norm.
    Returns (model, env_train, train_acc, test_acc).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env_train, env_test = get_env(args.env_name, mode=args.mode)
    model = get_agent(
        args.agent, 
        env_train, 
        **{k: v for k, v in vars(args).items() if k not in ['agent', 'env_name', 'mode']}
    )
    callback = RewardCallback(env_train, env_test, model, log_interval=args.log_interval, eval_interval=args.log_interval)
    model.learn(total_timesteps=args.Ntrain, callback=callback)
    model.save(os.path.join(args.save_dir, f"{args.agent}_{args.layer_type}_model_{seed}.pth"))
    _, train_acc, test_acc = callback.get_results()
    return model, env_train, train_acc, test_acc


def main(args):
    print("\nStarting Training with the following configuration:")
    print_parameters(vars(args))

    # Define three approaches: Softmax, Harmonic (L2), Harmonic (L1)
    approach_list = [
        ("Softmax", "softmax", "L2"),       # Softmax; distance_norm not used
        ("Harmonic (L2)", "harmonic", "L2"),  # Harmonic with L2 norm
        ("Harmonic (L1)", "harmonic", "L1")   # Harmonic with L1 norm
    ]
    
    models_envs = {}
    results = {}

    for label, ltype, dist in approach_list:
        print(Fore.MAGENTA + f"\n=== Running experiment: {label} ===" + Style.RESET_ALL)
        args.layer_type = ltype
        args.distance_norm = dist
        args.ensemble_size = 1  # single model
        model, env_train, train_acc, test_acc = run_training(args, seed=0)
        models_envs[label] = (model, env_train)
        results[label] = (train_acc, test_acc)

    # Create a 2x3 grid: top row for embeddings, bottom row for environment renders.
    n_cols = len(approach_list)
    fig, axs = plt.subplots(2, n_cols, figsize=(7*n_cols, 12))

    for i, (label, _, _) in enumerate(approach_list):
        model, env_train = models_envs[label]
        plot_actions_and_states_ax(model, env_train, approach_label=label, ax=axs[0, i])
        plot_env_render(env_train, axs[1, i], title=f"Env Render ({label})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_outpath = os.path.join(args.save_dir, "comparison_2x3.png")
    plt.savefig(combined_outpath, dpi=150)
    print(f"Saved combined figure to: {combined_outpath}")
    plt.show()

    # Optionally, plot ensemble curves for each approach.
    for label in results:
        train_acc, test_acc = results[label]
        plot_ensemble_curves([train_acc], [test_acc], args.save_dir, args.log_interval, args.Ntrain, label)

    save_results(results, args.save_dir, args.agent, "comparison")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PG agent on MazeEnv or gridWorld with ensemble and multiprocessing.")
    parser.add_argument('--env_name', type=str, default='maze_world', help='Env name')
    parser.add_argument('--agent', type=str, default='pg', help='Agent type.')
    parser.add_argument('--Ntrain', type=int, default=5*10**4, help='Number of training timesteps per agent.')
    parser.add_argument('--layer_type', type=str, choices=['softmax', 'harmonic'], default='softmax', help='Layer type.')
    parser.add_argument('--distance_norm', type=str, choices=['L1', 'L2'], default='L2', help='Distance norm for harmonic layer.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in the PG agent.')
    parser.add_argument('--harmonic_exponent', type=int, default=4, help='Harmonic exponent.')
    parser.add_argument('--ensemble_size', type=int, default=1, help='Number of ensemble agents.')
    parser.add_argument('--save_dir', type=str, default='./data', help='Directory to save results.')
    parser.add_argument('--parallel', action='store_true', help='Flag to use parallel training.')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of parallel workers.')
    parser.add_argument('--log_interval', type=int, default=None, help='Logging interval.')
    parser.add_argument('--mode', type=str, default='easy_medium', help='Mode for MazeEnv (e.g., "easy", "medium", "hard").')
    args = parser.parse_args()

    if args.log_interval is None:
        args.log_interval = int(0.1 * args.Ntrain)
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
