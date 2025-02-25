import os
import gymnasium as gym
import numpy as np
import torch

import minigrid

from envs.env_gridworld import gridWorld
from envs.env_single_eqn import singleEqn
from envs.env_multi_eqn import multiEqn
from envs.env_mazeworld import MazeEnv
from envs.env_mazeworld_random import MazeEnvRandom

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.utils_custom_policies import HarmonicPolicy
from utils.utils_callbacks import *

from rllte.xplore.reward import ICM

# Select environment
#env_name = 'multi_eqn'  # Change as needed
env_name = 'mini_grid'  # Change as needed
agent = 'ppo-harmonic'   # Options: 'ppo', 'ppo-harmonic'
agent = 'ppo'   # Options: 'ppo', 'ppo-harmonic'

print("\n" + "-" * 50)
print(f'Running {agent} PPO on {env_name}')
print("-" * 50 + "\n")

# Initialize the environment
if env_name == 'grid_world':
    env = gridWorld(difficulty='easy')
elif env_name == 'single_eqn':
    env = singleEqn(main_eqn='a/x+b')
elif env_name == 'multi_eqn':
    env = multiEqn(level=4)
elif env_name == 'maze_world_random':
    mode = 'easy'
    env = MazeEnvRandom(mode=mode)
elif env_name == 'mini_grid':
    env = gym.make("MiniGrid-Empty-5x5-v0")
else:
    env = gym.make(env_name)

# Apply action masking
env = ActionMasker(env, lambda env: env.action_mask)
env = DummyVecEnvWithReset([lambda: env])

# Initialize model
policy_kwargs = dict(net_arch=[128, 128, 128])  # Three layers of 256 neurons each

if agent == 'ppo-harmonic':
    policy_kwargs.update(
        distance_norm="L2",
        harmonic_exponent=256,
        weight_reg=0.01
    )
    model = MaskablePPO(
        policy=HarmonicPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        verbose=0
    )
else:
    model = MaskablePPO(
        'MlpPolicy',
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=0
    )

# Train model
Ntrain = 10**5
log_interval = int(0.01*(Ntrain))
callback = AccuracyLoggingCallback(env, Ntrain, log_interval=log_interval)
callback_ir = IntrinsicReward(ICM(env, device=get_device()))  # internal curiosity
model.learn(total_timesteps=Ntrain, callback=[callback, callback_ir])

# Save model
os.makedirs('models', exist_ok=True)
model.save(f'models/{agent}')

# Evaluate model 
print("\nEvaluating model on test equations\n")
test_eqns = env.envs[0].test_eqns
for eqn in test_eqns:
    print('\n')
    obs = env.env_method('set_equation', eqn)
    done, total_reward, steps = False, 0.0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        #breakpoint()
        #obs, reward, done, truncated, info = env.step(action)
        obs, reward, done, info = env.step(action)
        info = info[0]
        total_reward += reward[0]
        steps += 1
        print(f'{info["main_eqn"]} | {info["lhs"]} = {info["rhs"]} | action = {action}')

    print(f"Rollout finished in {steps} steps with total reward {total_reward:.2f}")
