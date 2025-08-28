#!/usr/bin/env python3
import os, gc
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import datetime
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import torch
import numpy as np

# Intrinsic reward imports
from rllte.xplore.reward import E3B, ICM, NGU, RE3, RIDE, RND

# Keep CPU threads modest
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

# Your custom environment (assumed to be available)
from envs.env_single_eqn import singleEqn


class CustomProgressBarCallback(ProgressBarCallback):
    def __init__(self, total_timesteps, agent, equation, seed, **kwargs):
        super().__init__(total_timesteps=total_timesteps, **kwargs)
        self.agent = agent
        self.equation = equation
        self.seed = seed

    def _init_callback(self):
        self.model._logger.info(f"Training {self.agent} [{self.equation}, seed={self.seed}] for {self.total_timesteps} timesteps...")
        self.pbar = tqdm(total=self.total_timesteps, desc=f"{self.agent} [{self.equation}, seed={self.seed}]")

    def _on_step(self):
        if self.pbar is not None:
            self.pbar.n = self.num_timesteps
            self.pbar.refresh()
        return True

def timed_print(msg):
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{time_str}: {msg}")

def get_device():
    """Returns the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        print('Found CUDA: using GPU')
        cur_proc_identity = mp.current_process()._identity
        if cur_proc_identity:
            return (cur_proc_identity[0] - 1) % torch.cuda.device_count()
        else:
            return 0
    else:
        print('CUDA not found: using CPU')
        return 'cpu'

def get_intrinsic_reward(intrinsic_reward, vec_env):
    """Returns an intrinsic reward module from rllte.xplore."""
    device = get_device()
    if intrinsic_reward == 'ICM':
        return ICM(vec_env, device=device)
    elif intrinsic_reward == 'E3B':
        return E3B(vec_env, device=device)
    elif intrinsic_reward == 'RIDE':
        return RIDE(vec_env, device=device)
    elif intrinsic_reward == 'RND':
        return RND(vec_env, device=device)
    elif intrinsic_reward == 'RE3':
        return RE3(vec_env, device=device)
    elif intrinsic_reward == 'NGU':
        return NGU(vec_env, device=device)
    else:
        return None

class IntrinsicReward(BaseCallback):
    """
    A more efficient callback for logging intrinsic rewards in RL training.
    """

    def __init__(self, irs, verbose=0, log_interval=100):
        super(IntrinsicReward, self).__init__(verbose)
        self.irs = irs
        self.buffer = None
        self.rewards_internal = []  # Store intrinsic rewards for logging
        self.log_interval = log_interval
        self.last_computed_intrinsic_rewards = None  # Store for logging

    def init_callback(self, model) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        Call .watch() to monitor transitions (required for NGU's episodic memory).
        Then, log previously computed intrinsic rewards if available.
        """
        # Call watch only for NGU; no-op for others
        if isinstance(self.irs, NGU):
            try:
                # Convert to tensors on the correct device
                device = self.irs.device
                observations = torch.as_tensor(self.model._last_obs, device=device).float()
                actions = torch.as_tensor(self.locals["actions"], device=device).float()
                rewards = torch.as_tensor(self.locals["rewards"], device=device).float()
                dones = torch.as_tensor(self.locals["dones"], device=device).float()
                next_observations = torch.as_tensor(self.locals["new_obs"], device=device).float()
                self.irs.watch(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    terminateds=dones,
                    truncateds=dones
                )
            except Exception as e:
                # Defensive: log error but continue
                print(f"Warning: NGU.watch() failed: {e}")

        if self.last_computed_intrinsic_rewards is not None:
            # Get last intrinsic reward from the rollout buffer
            intrinsic_reward = self.last_computed_intrinsic_rewards[-1]
            self.rewards_internal.append(intrinsic_reward)

        return True

    def _on_rollout_end(self) -> None:
        """
        Efficiently compute intrinsic rewards once per rollout and store them.
        """
        device = self.irs.device
        obs = torch.as_tensor(self.buffer.observations, device=device).float()
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = torch.as_tensor(self.locals["new_obs"], device=device).float()
        actions = torch.as_tensor(self.buffer.actions, device=device)
        rewards = torch.as_tensor(self.buffer.rewards, device=device)
        dones = torch.as_tensor(self.buffer.episode_starts, device=device)

        # Compute intrinsic rewards for the entire rollout at once
        samples = dict(observations=obs, actions=actions,
                      rewards=rewards, terminateds=dones,
                      truncateds=dones, next_observations=new_obs)
        intrinsic_rewards = self.irs.compute(
            samples=samples,
            sync=True
        ).cpu().numpy()

        # Update the reward module
        self.irs.update(samples=samples)

        # Ensure shape is (n_steps, 1) for addition to buffer
        if intrinsic_rewards.ndim == 1:
            intrinsic_rewards = intrinsic_rewards[:, np.newaxis]
        elif intrinsic_rewards.ndim > 2:
            # Defensive: flatten extra dims if unexpected (e.g., for NGU errors)
            intrinsic_rewards = intrinsic_rewards.reshape(intrinsic_rewards.shape[0], -1).mean(axis=1, keepdims=True)

        # Store them so `_on_step()` can access them
        self.last_computed_intrinsic_rewards = intrinsic_rewards

        # Add intrinsic rewards to the rollout buffer
        self.buffer.advantages += intrinsic_rewards
        self.buffer.returns += intrinsic_rewards

# Simple callback to track Tsolve
class TrainingLogger(BaseCallback):
    def __init__(self, algo_name: str, save_dir: str):
        super().__init__(verbose=1)
        self.algo_name = algo_name
        self.save_dir = save_dir
        self.Tsolve = None
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        step = self.num_timesteps
        for info in self.locals.get("infos", []):
            if info.get("is_solved"):
                self.Tsolve = step
                main_eqn = info.get("main_eqn")
                print(f"\033[33m[{self.algo_name}] Solved {main_eqn} at t={step}\033[0m")
                return False  # Stop training on solve
        return True

    def _on_training_end(self):
        gc.collect()

def make_agent(base_agent: str, env, hidden_dim: int, seed: int):
    policy_kwargs = dict(net_arch=[hidden_dim, hidden_dim])
    if base_agent == 'ppo':
        return PPO(
            'MlpPolicy',
            env=env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0
        )
    elif base_agent == 'dqn':
        return DQN(
            'MlpPolicy',
            env=env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0
        )
    elif base_agent == 'a2c':
        return A2C(
            'MlpPolicy',
            env=env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown base agent: {base_agent}")

def run_trial(agent: str, env_name: str, main_eqn: str, Ntrain: int, seed: int, save_dir: str, hidden_dim: int):
    # Parse base_agent and curiosity
    if '-' in agent:
        parts = agent.split('-')
        base_agent = parts[0].lower()
        curiosity = parts[1].upper()
    else:
        base_agent = agent.lower()
        curiosity = None

    # Build env
    train_env = singleEqn(main_eqn=main_eqn)
    
    # Prepare env for model
    env_for_model = train_env
    
    # Callback for logging Tsolve
    cb = TrainingLogger(algo_name=agent, save_dir=save_dir)
    cb_progress = ProgressBarCallback()
    #cb_progress = CustomProgressBarCallback(total_timesteps=Ntrain, agent=agent, equation=main_eqn, seed=seed)
    callbacks = [cb, cb_progress]
    
    if curiosity:
        if base_agent != 'ppo':
            raise ValueError("Curiosity (intrinsic rewards) only supported for PPO-based agents.")
        vec_env = DummyVecEnv([lambda: train_env])
        env_for_model = vec_env
        irs = get_intrinsic_reward(curiosity, vec_env)
        if irs:
            cb_curiosity = IntrinsicReward(irs, log_interval=Ntrain//10)
            callbacks.append(cb_curiosity)
    
    # Build model
    model = make_agent(base_agent, env_for_model, hidden_dim, seed)
    
    # Per-trial save dir
    tag = f"seed{seed}"
    run_dir = os.path.join(save_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    
    # Learn
    model.learn(total_timesteps=Ntrain, callback=callbacks)
    gc.collect()  # Force garbage collection

    if curiosity:
        vec_env.close()
    
    # Save model
    model_path = os.path.join(run_dir, f"{tag}.zip")
    model.save(model_path)
    #timed_print(f"[{agent}] Saved model → {model_path}")
    
    # Metrics
    tsolve = cb.Tsolve if cb.Tsolve is not None else 0
    metrics = {
        "agent": agent,
        "equation": main_eqn,
        "seed": seed,
        "Tsolve": tsolve
    }
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Cleanup
    train_env.close()
    
    return metrics, run_dir

def run_parallel(jobs, n_workers=1):
    rows, run_dirs = [], []
    ctx = mp.get_context("spawn")
    total = len(jobs)
    done = 0
    
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = [ex.submit(run_trial, *job) for job in jobs]
        for fut in as_completed(futures):
            try:
                metrics, run_dir = fut.result()
                rows.append(metrics)
                run_dirs.append(run_dir)
                done += 1
                timed_print(f"✓ [{done}/{total}] Finished: {metrics['agent']} eq={metrics['equation']} seed={metrics['seed']} | Tsolve={metrics['Tsolve']}")
            except Exception as e:
                done += 1
                timed_print(f"✗ [{done}/{total}] Job crashed: {e}")
    return rows, run_dirs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Sweep Script")
    parser.add_argument('--Ntrain', type=int, default=5*10**6, help='Total training timesteps')
    parser.add_argument('--n_trials', type=int, default=3, help='Number of trials per agent')
    parser.add_argument('--base_seed', type=int, default=40, help='Base seed')
    parser.add_argument('--n_workers', type=int, default=6, help='Number of parallel workers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--save_root', type=str, default="data/sweep", help='Save root directory')
    args = parser.parse_args()
    
    Ntrain = args.Ntrain
    n_trials = args.n_trials
    base_seed = args.base_seed
    n_workers = args.n_workers
    hidden_dim = args.hidden_dim
    save_root = args.save_root
    
    agents = ['dqn', 'a2c', 'ppo', 'ppo-ICM', 'ppo-E3B', 'ppo-RND', 'ppo-RIDE', 'ppo-NGU', 'ppo-RE3'][2:]
    agents = ['ppo-RND', 'ppo-NGU', 'ppo', 'ppo-ICM']
    equations = ['a*x', 'x+a', 'a*x+b', 'a/x+b', 'c*(a*x+b)+d', 'c + d/(a*x+b)', '(a*x+b)+e*(c*x+d)', 'e + (a*x+b)/(c*x+d)'][-1]
    equations = ['e + (a*x+b)/(c*x+d)']
    timed_print(f"Running sweep: agents={agents}, equations={equations}, Ntrain={Ntrain}, trials={n_trials}")
    
    # Build job list
    jobs = []
    for agent in agents:
        for equation in equations:
            save_dir = os.path.join(save_root, agent, equation)
            for t in range(n_trials):
                seed = base_seed + t
                jobs.append((agent, 'single_eqn', equation, Ntrain, seed, save_dir, hidden_dim))
    
    # Run jobs
    rows, run_dirs = run_parallel(jobs, n_workers=n_workers)
    
    if not rows:
        timed_print("No results gathered — all trials failed.")
        raise SystemExit(1)
    
    # Compute success rate: fraction of trials where Tsolve > 0
    df = pd.DataFrame(rows)
    summary = df.groupby(['agent', 'equation']).agg(
        success_rate=('Tsolve', lambda x: (x > 0).mean())
    ).reset_index()
    
    timed_print("\n=== Summary ===")
    # Define the desired order of equations
    equations = ['a*x', 'x+a', 'a*x+b', 'a/x+b', 'c*(a*x+b)+d', 'c + d/(a*x+b)', '(a*x+b)+e*(c*x+d)', 'e + (a*x+b)/(c*x+d)']
    # Pivot the DataFrame to create a table with agents as columns and equations as rows
    pivot_summary = summary.pivot(index='equation', columns='agent', values='success_rate')
    # Convert equation index to categorical to preserve order
    pivot_summary.index = pd.Categorical(pivot_summary.index, categories=equations, ordered=True)
    pivot_summary = pivot_summary.sort_index()
    # Format the success_rate to 3 decimal places
    pivot_summary = pivot_summary.map(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
    # Print the pivoted table
    timed_print(pivot_summary.to_string())
    # Save to CSV
    os.makedirs(save_root, exist_ok=True)
    out_csv = os.path.join(save_root, "summary.csv")
    pivot_summary.to_csv(out_csv)
    timed_print(f"Saved summary → {out_csv}")