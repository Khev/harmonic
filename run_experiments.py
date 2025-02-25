import subprocess
import matplotlib.pyplot as plt
import os
import numpy as np
import json

# Define the experiment configurations
agents = ['pg', 'a2c', 'ppo']
layers = ['softmax','harmonic']
Ntrain = 5*10**4  # or adjust as needed

results = {}

for agent in agents:
    for layer in layers:
        save_dir = f'./data/{agent}'
        os.makedirs(save_dir, exist_ok=True)
        print(f"Running experiment for Agent: {agent}, Layer: {layer}")
        subprocess.run([
            'python', 'train.py',
            '--env_name', 'grid_world',
            '--agent', agent,
            '--Ntrain', str(Ntrain),
            '--layer_type', layer,
            '--distance_norm', 'L2',
            '--n_layers', '2',
            '--ensemble_size', '5',
            '--save_dir', save_dir,
            '--parallel',  # Logging every 10% of Ntrain
            '--log_interval', str(Ntrain // 10)  # Logging every 10% of Ntrain

        ])

        # Load results from saved JSON (if implemented in train.py)
        with open(os.path.join(save_dir, f'{agent}_{layer}_results.json'), 'r') as f:
            results[f'{agent}_{layer}'] = json.load(f)
        

# 2) Load JSON results from disk into a dict
for agent in agents:
    for layer in layers:
        key = f"{agent}_{layer}"
        save_dir = f'./data/{agent}'
        result_file = os.path.join(save_dir, f'{key}_results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results[key] = json.load(f)
        else:
            print(f"Results file not found: {result_file}")

# 3) Dynamically create subplots based on #agents × #layers
n_rows = len(agents)
n_cols = len(layers)
fig, axes = plt.subplots(
    n_rows, n_cols, 
    figsize=(6 * n_cols, 4 * n_rows),  # scale figure size as desired
    squeeze=False                      # so axes is always 2D
)

# 4) Plot each (agent, layer) on the appropriate subplot
for i, agent in enumerate(agents):
    for j, layer in enumerate(layers):
        key = f"{agent}_{layer}"
        ax = axes[i][j]

        if key not in results:
            ax.set_title(f"No Results: {key}")
            ax.set_xlabel('Nstep')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True)
            continue

        # Each entry in results[key] is one run, e.g. [train_acc, test_acc]
        # We'll assume "train_acc" = run[0], "test_acc" = run[1].
        data = results[key]
        ensemble_train = [np.array(run[0]) for run in data]  # Train accuracies
        ensemble_test = [np.array(run[1]) for run in data]   # Test accuracies

        # Pad arrays so they have same length
        max_len = max(len(arr) for arr in ensemble_train)
        def pad(arr):
            return np.pad(arr, (0, max_len - len(arr)), 'edge')

        ensemble_train = np.array([pad(arr) for arr in ensemble_train])
        ensemble_test = np.array([pad(arr) for arr in ensemble_test])

        # Compute mean/min/max
        mean_train = ensemble_train.mean(axis=0)
        min_train = ensemble_train.min(axis=0)
        max_train = ensemble_train.max(axis=0)
        mean_test = ensemble_test.mean(axis=0)
        min_test = ensemble_test.min(axis=0)
        max_test = ensemble_test.max(axis=0)

        # Create a steps array to match the #points in mean_train
        # You logged every (Ntrain//10) steps or so, but let's guess a step spacing:
        # If e.g. each array is length M, then total steps might be Ntrain,
        # so spacing is roughly Ntrain/(M-1).
        step_spacing = Ntrain // (max_len - 1) if max_len > 1 else Ntrain
        steps = np.arange(max_len) * step_spacing

        # Plot
        ax.fill_between(steps, min_train, max_train, color='blue', alpha=0.2)
        ax.fill_between(steps, min_test, max_test, color='red', alpha=0.2)
        ax.plot(steps, mean_train, label='Mean Train Acc', color='blue')
        ax.plot(steps, mean_test, label='Mean Test Acc', color='red')

        ax.set_title(f"{agent} - {layer}")
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(True)

plt.tight_layout()
out_fig = './data/experiment_learning_curves_dynamic.png'
plt.savefig(out_fig, dpi=150)
plt.show()
print(f"Saved figure to {out_fig}")