#!/usr/bin/env python3
import os
import json
import datetime
import itertools
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback

# --- Your utils / envs ---
from utils.utils_env import *
from envs.env_single_eqn_fixed import singleEqn
from envs.env_multi_eqn_fixed import multiEqn

# Keep CPU threads modest
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


def timed_print(msg):
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{time_str}: {msg}")


# ---------------------------
# Env / agent factories
# ---------------------------
def make_env(env_name: str, gen, use_relabel_constants=False, use_memory=False, seed: int = 0):
    state_rep = 'graph_integer_1d'
    if env_name == 'single_eqn':
        env = singleEqn(main_eqn='a*x+b', use_relabel_constants=True)
    elif env_name == 'multi_eqn':
        env = multiEqn(gen=gen, use_relabel_constants=use_relabel_constants,
                       state_rep=state_rep, use_memory=use_memory)
    else:
        raise ValueError(f"Unknown env_name: {env_name}")
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    return env


def make_agent(agent: str, env, hidden_dim: int, seed: int = 0, load_path: str = None):
    """
    Create or load an agent. If `load_path` is provided, load and attach env.
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
        model = PPO('MlpPolicy', env=env, policy_kwargs=policy_kwargs,
                    seed=seed, verbose=0, n_steps=2048)
    elif agent == 'ppo-tree':
        pol, rep = "MultiInputPolicy", 'graph_integer_1d'
        hidden_dim_local = 64
        embed_dim = 32
        K = 2
        kwargs = dict(
            features_extractor_class=TreeMLPExtractor,
            features_extractor_kwargs=dict(
                max_nodes=env.observation_dim // 2,
                max_edges=2 * env.observation_dim // 2,
                vocab_min_id=-10,
                pad_id=99,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim_local,
                K=K,
                pooling="mean",
            ),
            net_arch=dict(pi=[128], vf=[128]),
        )
        model = PPO(policy=pol, env=env, policy_kwargs=kwargs, verbose=0, seed=seed, n_steps=2048)
    else:
        raise ValueError(f"Unknown agent: {agent}")

    return model


# ---------------------------
# Callback with logging
# ---------------------------
class TrainingLogger(BaseCallback):
    def __init__(self, algo_name: str, save_dir: str, num_train_eqns, verbose=1):
        super().__init__(verbose)
        self.algo_name = algo_name
        self.save_dir = save_dir
        self.eqn_solved = set()
        self.coverage = 0.0
        self.num_train_eqns = num_train_eqns
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        timed_print(f"[{self.algo_name}] Training started")

    def _on_step(self) -> bool:
        step = self.num_timesteps
        for info in self.locals.get("infos", []):
            if info.get("is_solved"):
                # Prefer canonical identifiers if your env emits them
                eqn = info.get("template_id", info.get("template_str", info.get("main_eqn", "eqn")))
                lhs = info.get("lhs")
                rhs = info.get("rhs")
                if eqn not in self.eqn_solved:
                    self.eqn_solved.add(eqn)
                    # Clamp to [0, 1] in case of overcounting due to augmentation
                    self.coverage = min(1.0, self.coverage + 1.0 / max(1, self.num_train_eqns))
                    print(f"\033[33m[{self.algo_name}] Solved {eqn} ==> {lhs} = {rhs} "
                          f"at step {step} | coverage = {self.coverage:.3f}\033[0m")
        return True

    def _on_training_end(self) -> None:
        timed_print(f"[{self.algo_name}] Training finished | Coverage={self.coverage:.3f}")


# ---------------------------
# Single trial (runs inside worker process)
# ---------------------------
def run_trial(agent: str, env_name: str, gen, Ntrain: int, seed: int, save_dir: str,
              use_relabel_constants: bool, use_memory: bool, hidden_dim: int, load_model_path: str = None):
    # Build env
    train_env = make_env(env_name, gen,
                         use_relabel_constants=use_relabel_constants,
                         use_memory=use_memory,
                         seed=seed)

    # Build model
    model = make_agent(agent, train_env, hidden_dim, seed=seed, load_path=load_model_path)

    # Per-trial save dir
    tag = f"seed{seed}"
    run_dir = os.path.join(save_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    # Callback
    num_train_eqns = len(getattr(train_env, "train_eqns", [])) or 1
    cb_solver = TrainingLogger(algo_name=agent, save_dir=run_dir, num_train_eqns=num_train_eqns)
    cb_progress = ProgressBarCallback()
    cb = [cb_solver, cb_progress]

    # Learn
    model.learn(total_timesteps=Ntrain, callback=cb)

    # Save model
    model_path = os.path.join(run_dir, f"{tag}.zip")
    model.save(model_path)
    timed_print(f"[{agent}] Saved model → {model_path}")

    coverage = cb_solver.coverage
    metrics = {
        "agent": agent,
        "env": env_name,
        "seed": seed,
        "coverage": coverage,
        "use_relabel_constants": use_relabel_constants,
        "use_memory": use_memory
    }

    # Save metrics
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    timed_print(f"[{agent}] Saved metrics → {metrics_path}")

    # Cleanup
    try:
        train_env.close()
    except Exception:
        pass

    return metrics, run_dir


# ---------------------------
# Wrapper for Pool.map
# ---------------------------
def _trial_wrapper(args):
    return run_trial(*args)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    import argparse
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="RL Debug Script (multiprocessing)")
    parser.add_argument('--env_name', type=str, default='multi_eqn', help='Environment name')
    parser.add_argument('--agent', type=str, default='ppo-tree', help='Agent type')
    parser.add_argument('--Ntrain', type=int, default=10**6, help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=10, help='Base seed')
    parser.add_argument('--gen', type=str, default='abel_level4', help='Generator for multi_eqn')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--save_root', type=str, default="data/debug_mp", help='Save root directory')
    parser.add_argument('--n_workers', type=int, default=2, help='Number of parallel workers')
    parser.add_argument('--trials', type=int, default=2, help='Trials per config')
    parser.add_argument('--load_model_path', type=str, default=None, help='Optional path to load a model')
    args = parser.parse_args()

    env_name = args.env_name
    agent = args.agent
    Ntrain = args.Ntrain
    base_seed = args.seed
    gen = args.gen
    hidden_dim = args.hidden_dim
    save_root = args.save_root
    n_workers = int(args.n_workers)
    trials = int(args.trials)
    load_model_path = args.load_model_path

    timed_print(f"Running debug (mp): agent={agent}, env={env_name}, Ntrain={Ntrain}, n_workers={n_workers}")

    sweep_use_relabel = [False, True]
    sweep_use_memory = [False, True]

    # Build jobs: (agent, env_name, gen, Ntrain, seed, save_dir, use_relabel, use_memory, hidden_dim, load_model_path)
    jobs = []
    for use_relabel_constants, use_memory in itertools.product(sweep_use_relabel, sweep_use_memory):
        subdir = f"{agent}_relabel{int(use_relabel_constants)}_mem{int(use_memory)}"
        save_dir_cfg = os.path.join(save_root, subdir)
        os.makedirs(save_dir_cfg, exist_ok=True)
        for t in range(trials):
            seed_i = base_seed + t
            jobs.append((
                agent, env_name, gen, Ntrain, seed_i, save_dir_cfg,
                use_relabel_constants, use_memory, hidden_dim, load_model_path
            ))

    # Run in parallel
    results = {}
    with Pool(processes=n_workers,maxtasksperchild=1) as pool:
        for (metrics, run_dir) in pool.imap_unordered(_trial_wrapper, jobs):
            key = (metrics["use_relabel_constants"], metrics["use_memory"])
            results.setdefault(key, []).append(metrics["coverage"])
            timed_print(f"✓ Finished: relabel={key[0]}, memory={key[1]}, seed={metrics['seed']} | "
                        f"coverage={metrics['coverage']:.3f}")
            

    # Summary
    def _stats(vals):
        return (min(vals), float(np.mean(vals)), max(vals)) if vals else (np.nan, np.nan, np.nan)

    timed_print("\n=== Coverage Summary ===")
    for use_relabel_constants, use_memory in itertools.product(sweep_use_relabel, sweep_use_memory):
        vals = results.get((use_relabel_constants, use_memory), [])
        mn, mean, mx = _stats(vals)
        timed_print(f"Relabel={use_relabel_constants}, Memory={use_memory}: "
                    f"min={mn:.3f}, mean={mean:.3f}, max={mx:.3f}")

