#!/usr/bin/env python3
import os
import json
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
import gymnasium as gym
import torch
import numpy as np
from sympy import symbols
from utils.utils_env import *

# Keep CPU threads modest
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

# Your custom environment
from envs.env_single_eqn_fixed import singleEqn
from envs.env_multi_eqn_fixed import multiEqn

def timed_print(msg):
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{time_str}: {msg}")

# --- Env / agent factories ---
def make_env(env_name: str, gen, use_relabel_constants=False, use_memory=False, seed: int = 0):
    state_rep = 'graph_integer_1d'
    if env_name == 'single_eqn':
        env = singleEqn(main_eqn='a*x+b', use_relabel_constants=True)
    elif env_name == 'multi_eqn':
        # x,a,b,c = symbols('x a b c')
        # env = multiEqn(train_eqns=[x+b+c], gen=gen, use_relabel_constants=True)
        env = multiEqn(gen=gen, use_relabel_constants=use_relabel_constants, state_rep=state_rep, use_memory=use_memory)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    return env

def make_agent(agent: str, env, hidden_dim: int, seed: int = 0, load_path: str = None):
    """
    Create or load an agent. If `load_path` is provided, we load the model and
    attach the given env. Works for MlpPolicy models.
    """
    if load_path:
        if not os.path.isfile(load_path):
            timed_print(f"[WARN] load_path does not exist: {load_path}. Training from scratch.")
        else:
            timed_print(f"[{agent}] Loading model from: {load_path}")
            model = PPO.load(load_path, env=env, device="auto", print_system_info=False)
            return model

    policy_kwargs = dict(net_arch=[hidden_dim, hidden_dim])

    if agent == 'ppo':
        model = PPO(
            'MlpPolicy',
            env=env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0,
            n_steps=2048
        )

    elif agent == 'ppo-tree':
        pol, rep = "MultiInputPolicy", 'graph_integer_1d'
        hidden_dim = 64
        embed_dim = 32
        K = 2
        kwargs = dict(
            features_extractor_class=TreeMLPExtractor,
            features_extractor_kwargs=dict(
                max_nodes=env.observation_dim//2,
                max_edges=2*env.observation_dim//2,
                vocab_min_id=-10,   # your dict had op ids down to -4 (e.g., 'sqrt'), safe default
                pad_id=99,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                K=K,
                pooling="mean",
            ),
            net_arch=dict(pi=[128], vf=[128]),
        )

        model = PPO(
            policy=pol,
            env=env,
            policy_kwargs=kwargs,
            verbose=0,
            seed=seed,
            n_steps=2048
        )

    else:
        raise ValueError(f"Unknown agent: {agent}")

    return model

# --- Callback with logging ---
class TrainingLogger(BaseCallback):
    def __init__(self, algo_name: str, save_dir: str, num_train_eqns, verbose=1):
        super().__init__(verbose)
        self.algo_name = algo_name
        self.save_dir = save_dir
        self.eqn_solved = set([])
        self.coverage = 0.0
        self.num_train_eqns = num_train_eqns
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        timed_print(f"[{self.algo_name}] Training started")

    def _on_step(self) -> bool:
        step = self.num_timesteps
        for info in self.locals.get("infos", []):
            if info.get("is_solved"):
                eqn = info.get("main_eqn", "eqn")
                lhs = info.get("lhs")
                rhs = info.get("rhs")
                if eqn not in self.eqn_solved:
                    self.eqn_solved.add(eqn)
                    self.coverage += 1.0/self.num_train_eqns
                    print(f"\033[33m[{self.algo_name}] Solved {eqn} ==> {lhs} = {rhs} at step {step} | coverage = {self.coverage:.3f} \033[0m")
        return True

    def _on_training_end(self) -> None:
        timed_print(f"[{self.algo_name}] Training finished | Coverage={self.coverage}")

# --- Run a single trial ---
def run_trial(agent: str, env_name: str, gen, Ntrain: int, seed: int, save_dir: str,
              use_relabel_constants: bool, use_memory: bool, hidden_dim: int):
    # Build env
    train_env = make_env(env_name, gen,
                         use_relabel_constants=use_relabel_constants,
                         use_memory=use_memory,
                         seed=seed)

    # Build model
    model = make_agent(agent, train_env, hidden_dim, seed=seed)

    # Per-trial save dir
    tag = f"seed{seed}"
    run_dir = os.path.join(save_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    # Callback
    cb_solver = TrainingLogger(algo_name=agent, save_dir=run_dir, num_train_eqns=len(train_env.train_eqns))
    cb_progress = ProgressBarCallback()
    cb = [cb_solver, cb_progress]

    # Learn
    model.learn(total_timesteps=Ntrain, callback=cb)

    # Save model
    model_path = os.path.join(run_dir, f"{tag}.zip")
    model.save(model_path)
    timed_print(f"[{agent}] Saved model → {model_path}")

    coverage = cb_solver.coverage
    print(f'Coverage = {coverage:.3f}')

    # Metrics
    metrics = {
        "agent": agent,
        "env": env_name,
        "seed": seed,
        "coverage": coverage,
        "use_relabel_constants": use_relabel_constants,
        "use_memory": use_memory
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    timed_print(f"[{agent}] Saved metrics → {metrics_path}")

    # Cleanup
    train_env.close()

    return metrics, run_dir

# --- Main ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL Debug Script")
    parser.add_argument('--env_name', type=str, default='multi_eqn', help='Environment name')
    parser.add_argument('--agent', type=str, default='ppo-tree', help='Agent type')
    parser.add_argument('--Ntrain', type=int, default=5*10**3, help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=10, help='Base seed')
    parser.add_argument('--gen', type=str, default='abel_level3', help='Generator for multi_eqn')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--save_root', type=str, default="data/debug", help='Save root directory')

    # Kept for interface parity; we sweep both flags below anyway
    parser.add_argument('--use_relabel_constants', type=bool, default=False, help='Use relabel constants')
    parser.add_argument('--use_memory', type=bool, default=False, help='Use hard-recall memory')

    args = parser.parse_args()

    env_name = args.env_name
    agent = args.agent
    Ntrain = args.Ntrain
    base_use_relabel_constants = args.use_relabel_constants
    base_use_memory = args.use_memory
    seed = args.seed
    gen = args.gen
    hidden_dim = args.hidden_dim
    save_root = args.save_root

    timed_print(f"Running debug: agent={agent}, env={env_name}, Ntrain={Ntrain}")

    # Run single trial across both booleans
    sweep_use_relabel = [True, False]
    sweep_use_memory = [True, False]

    results = {}
    for use_relabel_constants in sweep_use_relabel:
        for use_memory in sweep_use_memory:
            cfg_key = (use_relabel_constants, use_memory)
            coverages = []

            # Unique save dir per combo
            subdir = f"{agent}_relabel{int(use_relabel_constants)}_mem{int(use_memory)}"
            save_dir = os.path.join(save_root, subdir)

            for trial in range(1):  # same as your original (1 trial)
                seed_i = seed + trial
                timed_print(f"\nStarting run relabel={use_relabel_constants}, memory={use_memory}")
                metrics, run_dir = run_trial(
                    agent, env_name, gen, Ntrain, seed_i, save_dir,
                    use_relabel_constants, use_memory, hidden_dim
                )
                coverages.append(metrics['coverage'])

            results[cfg_key] = coverages

    # Summary at the end
    def _stats(vals):
        return (min(vals), float(np.mean(vals)), max(vals)) if vals else (np.nan, np.nan, np.nan)

    timed_print(f"\n=== Coverage Summary ===")
    for use_relabel_constants in sweep_use_relabel:
        for use_memory in sweep_use_memory:
            vals = results.get((use_relabel_constants, use_memory), [])
            mn, mean, mx = _stats(vals)
            timed_print(f"Relabel={use_relabel_constants}, Memory={use_memory}: "
                        f"min={mn:.3f}, mean={mean:.3f}, max={mx:.3f}")

    breakpoint()

    timed_print("Debug run completed")
