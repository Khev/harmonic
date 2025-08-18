#!/usr/bin/env python3
import os
import json
import time
import signal
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import TimeoutError
from contextlib import contextmanager, nullcontext
import boto3
from botocore.exceptions import ClientError
import gymnasium as gym
import torch
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

TRIAL_WALLCLOCK_LIMIT = 7 * 24 * 60 * 60  # 7 days hard cap per trial

# Configure logging for AWS debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Keep CPU threads modest when running many workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

# --- Your envs ---
from envs.env_single_eqn_fixed import singleEqn
from envs.env_multi_eqn_fixed import multiEqn
from stable_baselines3.common.vec_env import DummyVecEnv

# --- SB3 ---
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm

# --- Custom bits ---
from rllte.xplore.reward import E3B, ICM, NGU, RE3, RIDE, RND

# --- Eval timeout knobs (seconds) ---
EVAL_TIMEOUT_DET = 1.5  # Increased for abel_level4
EVAL_TIMEOUT_STOCH = 2.0  # Increased for abel_level4


def timed_print(msg):
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{time_str}: {msg}")
    logger.info(msg)


# ==========================
# Lightweight time limiter
# ==========================
class EvalTimeout(Exception):
    pass


def get_device():
    """Return a torch.device and pin a GPU per worker if available."""
    if torch.cuda.is_available():
        world = torch.cuda.device_count()
        ident = None
        try:
            ident = mp.current_process()._identity
        except Exception:
            pass
        if ident:
            idx = (ident[0] - 1) % world
        else:
            idx = os.getpid() % world
        torch.cuda.set_device(idx)
        dev = torch.device(f"cuda:{idx}")
        logger.info(f'Found CUDA: using {dev}')
        return dev
    else:
        logger.info('CUDA not found: using CPU')
        return torch.device('cpu')


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


@contextmanager
def time_limit(seconds: float):
    """
    Raise EvalTimeout if the with-block exceeds `seconds`.
    Uses SIGALRM; becomes a no-op on Windows or non-positive seconds.
    """
    if os.name == "nt" or not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise EvalTimeout()

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


class IntrinsicReward(BaseCallback):
    """
    A more efficient callback for logging intrinsic rewards in RL training.
    """
    def __init__(self, irs, verbose=0, log_interval=100):
        super(IntrinsicReward, self).__init__(verbose)
        self.irs = irs
        self.buffer = None
        self.rewards_internal = []
        self.log_interval = log_interval
        self.last_computed_intrinsic_rewards = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        if isinstance(self.irs, NGU):
            try:
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
                logger.warning(f"NGU.watch() failed: {e}")

        if self.last_computed_intrinsic_rewards is not None:
            intrinsic_reward = self.last_computed_intrinsic_rewards[-1]
            self.rewards_internal.append(intrinsic_reward)

        return True

    def _on_rollout_end(self) -> None:
        device = self.irs.device
        obs = torch.as_tensor(self.buffer.observations, device=device).float()
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = torch.as_tensor(self.locals["new_obs"], device=device).float()
        actions = torch.as_tensor(self.buffer.actions, device=device)
        rewards = torch.as_tensor(self.buffer.rewards, device=device)
        dones = torch.as_tensor(self.buffer.episode_starts, device=device)

        samples = dict(observations=obs, actions=actions,
                      rewards=rewards, terminateds=dones,
                      truncateds=dones, next_observations=new_obs)
        intrinsic_rewards = self.irs.compute(
            samples=samples,
            sync=True
        ).cpu().numpy()

        self.irs.update(samples=samples)

        if intrinsic_rewards.ndim == 1:
            intrinsic_rewards = intrinsic_rewards[:, np.newaxis]
        elif intrinsic_rewards.ndim > 2:
            intrinsic_rewards = intrinsic_rewards.reshape(intrinsic_rewards.shape[0], -1).mean(axis=1, keepdims=True)

        self.last_computed_intrinsic_rewards = intrinsic_rewards
        self.buffer.advantages += intrinsic_rewards
        self.buffer.returns += intrinsic_rewards


# ---------------------------
# Env / agent factories
# ---------------------------
def make_env(env_name: str, gen, seed: int = 0):
    if env_name == 'single_eqn':
        env = singleEqn(main_eqn='a*x+b')
    elif env_name == 'multi_eqn':
        env = multiEqn(gen=gen)
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
    device = get_device()  # Explicit device for PPO
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
            device=device
        )
    else:
        raise ValueError(f"Unknown agent: {agent}")

    return model


# ---------------------------
# Evaluation helpers (timeout-safe)
# ---------------------------
def _set_equation_if_supported(env, eqn):
    seteq = getattr(env, "set_equation", None)
    if callable(seteq):
        seteq(eqn)
    setup = getattr(env, "setup", None)
    if callable(setup):
        setup()


def greedy_solve_one(model, env, eqn, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET):
    try:
        with time_limit(per_eqn_seconds):
            _, _ = env.reset()
            _set_equation_if_supported(env, eqn)
            obs, _ = env.reset()
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if info.get("is_solved", False):
                    return True
                if terminated or truncated:
                    break
            return False
    except EvalTimeout:
        return False
    except Exception as e:
        logger.warning(f"greedy_solve_one failed: {e}")
        return False


def greedy_accuracy(model, env, equations, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET):
    if not equations:
        return None
    solved = 0
    for eqn in equations:
        if greedy_solve_one(model, env, eqn, max_steps=max_steps, per_eqn_seconds=per_eqn_seconds):
            solved += 1
    return solved / len(equations)


def success_at_n(model, env, equations, n_trials=10, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_STOCH):
    if not equations:
        return None

    solved_any = 0
    for eqn in equations:
        try:
            with time_limit(per_eqn_seconds):
                solved_this = False
                for _ in range(n_trials):
                    _, _ = env.reset()
                    _set_equation_if_supported(env, eqn)
                    obs, _ = env.reset()
                    for _ in range(max_steps):
                        action, _ = model.predict(obs, deterministic=False)
                        obs, _, terminated, truncated, info = env.step(action)
                        if info.get("is_solved", False):
                            solved_this = True
                            break
                        if terminated or truncated:
                            break
                    if solved_this:
                        break
                if solved_this:
                    solved_any += 1
        except EvalTimeout:
            pass
        except Exception as e:
            logger.warning(f"success_at_n failed: {e}")
            pass

    return solved_any / len(equations)


# ---------------------------
# Callback with periodic eval and checkpointing
# ---------------------------
class TrainingLogger(BaseCallback):
    def __init__(self, algo_name: str, train_env, eval_env, eval_interval: int, log_interval: int, save_dir: str, s3_bucket: str = None, s3_prefix: str = None, verbose=1):
        super().__init__(verbose)
        self.algo_name = algo_name
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix or ""
        self.curves_path = os.path.join(self.save_dir, "learning_curves.csv")

        self.train_eqns = getattr(train_env, "train_eqns", None)
        self.test_eqns = getattr(train_env, "test_eqns", None)
        self.num_eqns = len(self.train_eqns) if self.train_eqns is not None else 1

        self.Tsolves = {}
        self.Tsolve = None
        self.Tconverge = None

        self.log_steps = []
        self.coverage = []
        self.test_acc = []
        self.test_at10 = []

        os.makedirs(self.save_dir, exist_ok=True)
        self.s3_client = boto3.client('s3') if s3_bucket else None

    def _log_eval(self, step):
        cov = len(self.Tsolves) / self.num_eqns if self.num_eqns else 0.0
        tst = greedy_accuracy(self.model, self.eval_env, self.test_eqns, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET) if self.test_eqns else None
        t10 = success_at_n(self.model, self.eval_env, self.test_eqns, n_trials=10, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_STOCH) if self.test_eqns else None

        self.log_steps.append(step)
        self.coverage.append(cov)
        self.test_acc.append(tst if tst is not None else np.nan)
        self.test_at10.append(t10 if t10 is not None else np.nan)

        tst_s = f"{tst:.2f}" if tst is not None else "NA"
        t10_s = f"{t10:.2f}" if t10 is not None else "NA"
        timed_print(f"[{self.algo_name}]t={step}: coverage={cov:.3f} | test_acc={tst_s} | test@10={t10_s}")

        curves = pd.DataFrame({
            "step": self.log_steps,
            "coverage": self.coverage,
            "test_acc": self.test_acc,
            "test_at10": self.test_at10
        })
        tmp_path = self.curves_path + ".tmp"
        curves.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.curves_path)

        if self.s3_client and self.s3_bucket:
            try:
                s3_key = os.path.join(self.s3_prefix, os.path.basename(self.curves_path))
                self.s3_client.upload_file(self.curves_path, self.s3_bucket, s3_key)
                logger.info(f"Uploaded learning curves to s3://{self.s3_bucket}/{s3_key}")
            except ClientError as e:
                logger.warning(f"Failed to upload learning curves to S3: {e}")

    def _on_training_start(self) -> None:
        timed_print(f"[{self.algo_name}] Training started (train_eqns={self.num_eqns}, test_eqns={len(self.test_eqns) if self.test_eqns else 0})")
        self._log_eval(step=0)

    def _on_step(self) -> bool:
        step = self.num_timesteps
        for info in self.locals.get("infos", []):
            if info.get("is_solved"):
                eqn = info.get("main_eqn", "eqn")
                lhs = info.get("lhs")
                rhs = info.get("rhs")
                if eqn not in self.Tsolves:
                    self.Tsolves[eqn] = step
                    print(f"\033[33m[{self.algo_name}] Solved {eqn} ==> {lhs} = {rhs} at step {step}\033[0m")

                if self.Tconverge is None and len(self.Tsolves) >= self.num_eqns:
                    self.Tconverge = step
                    timed_print(f"[{self.algo_name}] Coverage 100% at step {step}")

        if self.eval_interval and step % self.eval_interval == 0:
            self._log_eval(step)
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{step}.zip")
            self.model.save(checkpoint_path)
            timed_print(f"[{self.algo_name}] Saved checkpoint at step {step} → {checkpoint_path}")

            if self.s3_client and self.s3_bucket:
                try:
                    s3_key = os.path.join(self.s3_prefix, os.path.basename(checkpoint_path))
                    self.s3_client.upload_file(checkpoint_path, self.s3_bucket, s3_key)
                    logger.info(f"Uploaded checkpoint to s3://{self.s3_bucket}/{s3_key}")
                except ClientError as e:
                    logger.warning(f"Failed to upload checkpoint to S3: {e}")

        return True

    def _on_training_end(self) -> None:
        if self.Tsolves:
            self.Tsolve = float(np.mean(list(self.Tsolves.values())))
        else:
            self.Tsolve = float('inf')
        timed_print(f"[{self.algo_name}] Training finished | Tsolve={self.Tsolve} | Tconverge={self.Tconverge}")

        curves = pd.DataFrame({
            "step": self.log_steps,
            "coverage": self.coverage,
            "test_acc": self.test_acc,
            "test_at10": self.test_at10
        })
        curves_path = os.path.join(self.save_dir, "learning_curves.csv")
        curves.to_csv(curves_path, index=False)
        timed_print(f"[{self.algo_name}] Saved curves → {curves_path}")

        if self.s3_client and self.s3_bucket:
            try:
                s3_key = os.path.join(self.s3_prefix, os.path.basename(curves_path))
                self.s3_client.upload_file(curves_path, self.s3_bucket, s3_key)
                logger.info(f"Uploaded final curves to s3://{self.s3_bucket}/{s3_key}")
            except ClientError as e:
                logger.warning(f"Failed to upload final curves to S3: {e}")


# ---------------------------
# Worker: run one trial
# ---------------------------
def run_trial(agent: str, env_name: str, gen, Ntrain: int, eval_interval: int, log_interval: int, seed: int, save_dir: str, curiosity, hidden_dim: int, load_model_path: str = None, s3_bucket: str = None, s3_prefix: str = None):
    # Check for Spot Instance termination notice
    try:
        import requests
    except Exception:
        requests = None

    if requests:
        try:
            r = requests.get("http://169.254.169.254/latest/meta-data/spot/instance-action", timeout=2)
            if r.status_code == 200:
                logger.warning("Spot Interruption notice received. Saving progress and exiting.")
                return None, None
        except Exception:
            pass

    # Single S3 client for all uploads
    s3_client = boto3.client('s3') if s3_bucket else None

    # Per-trial save dir
    tag = f"seed{seed}"
    run_dir = os.path.join(save_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    # Look for latest checkpoint to resume
    if not load_model_path:
        import glob
        checkpoint_glob = os.path.join(run_dir, "checkpoint_*.zip")
        checkpoints = sorted(
            glob.glob(checkpoint_glob),
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]) if "checkpoint_" in x else 0
        )
        if checkpoints:
            load_model_path = checkpoints[-1]
            timed_print(f"[{agent}] Resuming from checkpoint: {load_model_path}")

    # Build envs
    train_env = make_env(env_name, gen, seed=seed)
    eval_env = make_env(env_name, gen, seed=seed + 777)

    # Build or load model
    agent_temp = 'ppo'
    model = make_agent(agent_temp, train_env, hidden_dim, seed=seed, load_path=load_model_path)

    try:
        with time_limit(TRIAL_WALLCLOCK_LIMIT):
            # Callback with eval and checkpointing
            cb = TrainingLogger(
                algo_name=agent,
                train_env=train_env,
                eval_env=eval_env,
                eval_interval=eval_interval,
                log_interval=log_interval,
                save_dir=run_dir,
                s3_bucket=s3_bucket,
                s3_prefix=os.path.join(s3_prefix, tag) if s3_prefix else tag
            )

            # Intrinsic reward
            if curiosity is not None:
                train_env_wrapped = DummyVecEnv([lambda: train_env])
                irs = get_intrinsic_reward(curiosity, train_env_wrapped)
                if irs:
                    cb_curiosity = IntrinsicReward(irs, log_interval=log_interval)
                    cb = [cb, cb_curiosity]

            # Learn
            try:
                model.learn(total_timesteps=Ntrain, callback=cb)
            except KeyboardInterrupt:
                timed_print(f"[{agent}] Interrupted. Saving final model and metrics.")
                model_path = os.path.join(run_dir, f"{tag}.zip")
                model.save(model_path)
                if s3_client and s3_bucket:
                    try:
                        s3_key = os.path.join(s3_prefix, tag, f"{tag}.zip") if s3_prefix else f"{tag}/{tag}.zip"
                        s3_client.upload_file(model_path, s3_bucket, s3_key)
                        logger.info(f"Uploaded interrupted model to s3://{s3_bucket}/{s3_key}")
                    except ClientError as e:
                        logger.warning(f"Failed to upload interrupted model to S3: {e}")
                raise

            # Save artifacts (keep original + stable name)
            model_path = os.path.join(run_dir, f"{tag}.zip")
            model.save(model_path)
            final_model_path = os.path.join(run_dir, "final_model.zip")
            model.save(final_model_path)

            # Upload models to S3
            if s3_client and s3_bucket:
                try:
                    for path in [model_path, final_model_path]:
                        s3_key = os.path.join(s3_prefix, tag, os.path.basename(path)) if s3_prefix else f"{tag}/{os.path.basename(path)}"
                        s3_client.upload_file(path, s3_bucket, s3_key)
                        logger.info(f"Uploaded model to s3://{s3_bucket}/{s3_key}")
                except ClientError as e:
                    logger.warning(f"Failed to upload model to S3: {e}")

            # Final eval summaries (timeout-safe)
            final_test_acc = greedy_accuracy(model, eval_env, getattr(train_env, "test_eqns", []), max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_DET) or 0.0
            final_test_at10 = success_at_n(model, eval_env, getattr(train_env, "test_eqns", []), n_trials=10, max_steps=60, per_eqn_seconds=EVAL_TIMEOUT_STOCH) or 0.0

            if isinstance(cb, list):
                cb = cb[0]
            coverage_final = len(cb.Tsolves)
            num_eqns = cb.num_eqns
            coverage_final_rate = (coverage_final / num_eqns) if num_eqns else 0.0

            metrics = {
                "agent": agent,
                "env": env_name,
                "seed": seed,
                "coverage_final_rate": coverage_final_rate,
                "final_train_acc": 0.0,  # Not computed
                "final_test_acc": final_test_acc,
                "final_test_at10": final_test_at10,
                "Tsolve": cb.Tsolve,
                "Tconverge": cb.Tconverge,
                "num_eqns": num_eqns,
                "model_path": model_path,
                "final_model_path": final_model_path
            }
            metrics_path = os.path.join(run_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            timed_print(f"[{agent}] Saved metrics → {metrics_path}")

            if s3_client and s3_bucket:
                try:
                    s3_key = os.path.join(s3_prefix, tag, "metrics.json") if s3_prefix else f"{tag}/metrics.json"
                    s3_client.upload_file(metrics_path, s3_bucket, s3_key)
                    logger.info(f"Uploaded metrics to s3://{s3_bucket}/{s3_key}")
                except ClientError as e:
                    logger.warning(f"Failed to upload metrics to S3: {e}")

    except EvalTimeout:
        num_eqns = len(getattr(train_env, "train_eqns", []) or [])
        metrics = {
            "agent": agent, "env": env_name, "seed": seed,
            "coverage_final_rate": 0.0, "final_train_acc": 0.0,
            "final_test_acc": 0.0, "final_test_at10": 0.0,
            "Tsolve": float('inf'), "Tconverge": None, "num_eqns": num_eqns,
            "model_path": None, "final_model_path": None, "timeout": True
        }
        metrics_path = os.path.join(run_dir, "metrics.json")
        os.makedirs(run_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        timed_print(f"[{agent}] ⏱️ Timed out at ~{TRIAL_WALLCLOCK_LIMIT}s — saved partial metrics → {metrics_path}")
        if s3_client and s3_bucket:
            try:
                s3_key = os.path.join(s3_prefix or "", tag, "metrics.json")
                s3_client.upload_file(metrics_path, s3_bucket, s3_key)
                logger.info(f"Uploaded timeout metrics to s3://{s3_bucket}/{s3_key}")
            except ClientError as e:
                logger.warning(f"Failed to upload timeout metrics to S3: {e}")

    finally:
        try:
            train_env.close()
            eval_env.close()
        except Exception as e:
            logger.warning(f"Environment cleanup failed: {e}")

    return metrics, run_dir


# ---------------------------
# Pool runner without batching
# ---------------------------
def run_trial_wrapper(args):
    return run_trial(*args)


def run_parallel(jobs, n_workers=4, timeout_per_job=None, s3_bucket=None, s3_prefix=None):
    """
    Submit ALL jobs at once; idle workers immediately pick up the next.
    Results stream back in completion order. A single slow job won't block others.
    """
    rows, run_dirs = [], []
    ctx = mp.get_context("spawn")
    total = len(jobs)
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = [ex.submit(run_trial_wrapper, job + (s3_bucket, s3_prefix)) for job in jobs]
        for fut in as_completed(futures):
            try:
                metrics, run_dir = fut.result(timeout=timeout_per_job)
                if metrics is None and run_dir is None:
                    done += 1
                    timed_print(f"✗ [{done}/{total}] Job terminated (likely Spot interruption)")
                    continue
                rows.append(metrics)
                run_dirs.append(run_dir)
                done += 1
                timed_print(
                    f"✓ [{done}/{total}] Finished: {metrics['agent']} seed={metrics['seed']} "
                    f"| coverage_final={metrics['coverage_final_rate']:.2f} "
                    f"| test_acc={metrics['final_test_acc']:.2f} "
                    f"| test@10={metrics['final_test_at10']:.2f}"
                )
            except Exception as e:
                done += 1
                timed_print(f"✗ [{done}/{total}] Job crashed or timed out: {e}")
    return rows, run_dirs


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL Sweep Script")
    parser.add_argument('--env_name', type=str, default='multi_eqn', help='Environment name')
    parser.add_argument('--agents', nargs='+', default=['ppo', 'ppo-RND', 'ppo-RIDE', 'ppo-NGU', 'ppo-RE3', 'ppo-ICM', 'ppo-E3B'], help='List of agents')
    parser.add_argument('--Ntrain', type=int, default=10**7, help='Total training timesteps')
    parser.add_argument('--eval_interval', type=int, default=10**6, help='Evaluation interval')
    parser.add_argument('--log_interval', type=int, default=10**6, help='Log interval')
    parser.add_argument('--n_trials', type=int, default=2, help='Number of trials per agent')
    parser.add_argument('--base_seed', type=int, default=0, help='Base seed')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--gen', type=str, default='abel_level4', help='Generator for multi_eqn')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for policy network')
    parser.add_argument('--save_root', type=str, default=None, help='Save root directory')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load model from')
    parser.add_argument('--s3_bucket', type=str, default=None, help='S3 bucket for storing results')
    parser.add_argument('--s3_prefix', type=str, default='rl_sweep', help='S3 prefix for organizing results')
    args = parser.parse_args()

    env_name = args.env_name
    agents = args.agents
    Ntrain = args.Ntrain
    eval_interval = Ntrain // 10
    log_interval = Ntrain // 10
    n_trials = args.n_trials
    base_seed = args.base_seed
    n_workers = args.n_workers
    gen = args.gen
    hidden_dim = args.hidden_dim
    save_root = args.save_root or f"data/{gen}_hidden_dim{hidden_dim}"
    load_model_path = args.load_model_path
    s3_bucket = args.s3_bucket
    s3_prefix = args.s3_prefix
    curiosity = None

    timed_print("\n" + "-" * 50)
    timed_print(f"Parallel sweep on {env_name} | agents={agents} | Ntrain={Ntrain} | {curiosity}")
    timed_print(f"Eval timeouts: greedy={EVAL_TIMEOUT_DET}s, success@N={EVAL_TIMEOUT_STOCH}s")
    if load_model_path:
        timed_print(f"Loading initial weights from: {load_model_path}")
    if s3_bucket:
        timed_print(f"Saving results to s3://{s3_bucket}/{s3_prefix}")
    timed_print("-" * 50 + "\n")

    # Build the job list
    jobs = []
    for agent in agents:
        save_root_agent = os.path.join(save_root, agent)
        if agent == 'ppo':
            bump, curiosity_local = 0, None
        else:
            curiosity_type = agent.split('-')[1]
            bump = {'ICM': 1000, 'E3B': 2000, 'RIDE': 3000, 'RND': 4000, 'RE3': 5000, 'NGU': 6000}[curiosity_type]
            curiosity_local = curiosity_type
        
        for t in range(n_trials):
            seed = base_seed + 1000 * t + bump
            jobs.append((agent, env_name, gen, Ntrain, eval_interval, log_interval, seed, save_root_agent, curiosity_local, hidden_dim, load_model_path))

    # Run in parallel without batching
    rows, run_dirs = run_parallel(
        jobs,
        n_workers=n_workers,
        timeout_per_job=None,  # Rely on internal trial timeout
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix
    )

    if not rows:
        timed_print("No results gathered — all trials failed/timeouts?")
        raise SystemExit(1)

    # Aggregate table — ONLY requested metrics, with mean and std
    df = pd.DataFrame(rows)
    summary = df.groupby('agent').agg(
        coverage_mean=('coverage_final_rate', 'mean'),
        coverage_std=('coverage_final_rate', 'std'),
        final_test_acc_mean=('final_test_acc', 'mean'),
        final_test_acc_std=('final_test_acc', 'std'),
        final_test_at10_mean=('final_test_at10', 'mean'),
        final_test_at10_std=('final_test_at10', 'std'),
    ).reset_index()

    timed_print("\n=== Summary over trials ===")
    timed_print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    os.makedirs(save_root, exist_ok=True)
    out_csv = os.path.join(save_root, "summary.csv")
    summary.to_csv(out_csv, index=False)
    timed_print(f"\nSaved summary → {out_csv}")

    if s3_bucket:
        try:
            s3_key = os.path.join(s3_prefix, "summary.csv")
            boto3.client('s3').upload_file(out_csv, s3_bucket, s3_key)
            logger.info(f"Uploaded summary to s3://{s3_bucket}/{s3_key}")
        except ClientError as e:
            logger.warning(f"Failed to upload summary to S3: {e}")