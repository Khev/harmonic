import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
from sympy import sympify
from gymnasium import spaces, Env
from utils.utils_custom_functions import *
from operator import add, sub, mul, truediv
from utils.utils_env import *

logger = logging.getLogger(__name__)

class singleEqn(Env):
    """Environment for solving linear equations using RL."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, main_eqn='a*x+b', state_rep='integer_1d', normalize_rewards=True, verbose=False, \
         cache=False) -> None:
        super().__init__()

        # Static parts
        self.max_expr_length = 20
        self.max_steps = 10
        self.action_dim = 50
        self.observation_dim = 2*self.max_expr_length+1

        # Rewards
        self.reward_solved = +100
        self.reward_invalid_equation = -100
        self.reward_illegal_action = -1

        # Optimizing
        self.cache = cache
        if self.cache:
            self.action_cache = {}  # Cache for make_actions

        # Pars
        self.normalize_rewards = normalize_rewards
        self.state_rep = state_rep
        self.verbose = verbose
        
        # Set main equation to solve
        self.main_eqn = sympify(main_eqn)
        self.lhs = self.main_eqn
        self.rhs = 0
        self.x = symbols('x')

        #  Make feature_dict, actions etc
        self.setup()

        # RL env stuff
        self.state = self.to_vec(self.lhs, self.rhs)
        self.action_space = spaces.Discrete(self.action_dim)

        if state_rep == 'integer_1d':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float64)
        elif state_rep == 'integer_2d':
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim, 2), dtype=np.float64)
        elif state_rep == 'graph_integer_1d' or state_rep == 'graph_integer_2d':
            self.observation_space = spaces.Dict({
                "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim, 2), dtype=np.float64),
                "edge_index": spaces.Box(low=0, high=self.observation_dim, shape=(2, 2*self.observation_dim), dtype=np.int32),
                "node_mask": spaces.Box(low=0, high=1, shape=(self.observation_dim,), dtype=np.int32),
                "edge_mask": spaces.Box(low=0, high=1, shape=(2*self.observation_dim,), dtype=np.int32),
            })

        else:
            raise ValueError(f"Unsupported state representation: {state_rep}")


    def setup(self):
        
        # Ex: {'add':-1,, ... x:1, a:2, ...}
        self.feature_dict = make_feature_dict(self.main_eqn, self.state_rep)

        # Define fixed actions directly
        self.actions_fixed = [
            (custom_expand, None),
            #(custom_simplify, None),
            (custom_factor, None),
            (custom_collect, self.x),     # only allow collecting at x
            (custom_together, None),
            (custom_ratsimp, None),
            (custom_square, None),
            (custom_sqrt, None),
            (mul, -1)
        ]

        if self.cache:
            self.actions, self.action_mask = make_actions_cache(self.lhs, self.rhs, self.actions_fixed, \
                 self.action_dim, self.action_cache)
        else:
            self.actions, self.action_mask = make_actions(self.lhs, self.rhs, self.actions_fixed, self.action_dim)


    def step(self, action_index):

        # Make actions: these are dynamic, so depend on current lhs, rhs
        lhs_old, rhs_old, obs_old = self.lhs, self.rhs, self.obs
        if self.cache:
            action_list, action_mask = make_actions_cache(lhs_old, rhs_old, self.actions_fixed, \
                 self.action_dim, self.action_cache)
        else:
            action_list, action_mask = make_actions(lhs_old, rhs_old, self.actions_fixed, self.action_dim)

        self.actions = action_list
        self.action_mask = action_mask

        # Apply action
        action = action_list[action_index]
        operation, term = action
        lhs_new, rhs_new = operation(lhs_old, term), operation(rhs_old, term)
        obs_new, complexity_obs_new = self.to_vec(lhs_new, rhs_new)

        # Check if valid equation. 
        # Ex1: (lhs, rhs) = (a, b/x), in which case we -> (b/x,a); keep x on lhs
        # Ex2: (lhs,rhs) = (1,0) --> bad... 
        is_valid_eqn, lhs_new, rhs_new = check_valid_eqn(lhs_new, rhs_new)

        # illegal action

        # Check if solved
        is_solved = check_eqn_solved(lhs_new, rhs_new, self.main_eqn)

        # Reward
        reward = self.find_reward(lhs_old, rhs_old, lhs_new, rhs_new, is_valid_eqn, is_solved)

        # Checks on eqns: too long, too complex etc.
        too_many_steps = self.current_steps >= self.max_steps
        terminated = bool(is_solved or too_many_steps or not is_valid_eqn)
        truncated = False

        # Update counters
        self.lhs, self.rhs, self.obs = lhs_new, rhs_new, obs_new
        self.current_steps += 1

        info = {
            'is_solved': is_solved,
            'is_valid_eqn': is_valid_eqn,
            'too_many_steps': too_many_steps,
            'lhs': self.lhs,
            'rhs': self.rhs,
            'main_eqn': self.main_eqn,
            'action_mask': self.action_mask
        }

        if self.verbose:
            print(f'{self.lhs} = {self.rhs}. (Operation,term): {operation_names[operation]}, {term}')

        #print(f'{obs_new} \n')
        
        return obs_new, reward, terminated, truncated, info


    def reset(self, seed=0, options=None):
        self.current_steps = 0
        self.lhs, self.rhs = self.main_eqn, 0
        obs, _ = self.to_vec(self.lhs, self.rhs)
        self.obs = obs
        return obs, {}


    def render(self, mode: str = "human"):
        print(f'{self.lhs} = {self.rhs}')


    def to_vec(self, lhs, rhs):
        if self.state_rep == 'integer_1d':
            return integer_encoding_1d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep == 'integer_2d':
            return integer_encoding_2d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep == 'graph_integer_1d':
            return graph_encoding(lhs, rhs, self.feature_dict, self.max_expr_length)  
        elif self.state_rep == 'graph_integer_2d':
            return graph_encoding(lhs, rhs, self.feature_dict, self.max_expr_length)  
        else:
            raise ValueError(f"Unknown state representation: {self.state_rep}")

    def find_reward(self, lhs_old, rhs_old, lhs_new, rhs_new, is_valid_eqn, is_solved):

        if is_valid_eqn == False:
            reward = self.reward_invalid_equation

        elif is_solved:
            reward = self.reward_solved

        else:
            obs_old_complexity = get_complexity_expression(lhs_old) + get_complexity_expression(rhs_old)
            obs_new_complexity = get_complexity_expression(lhs_new) + get_complexity_expression(rhs_new)
            reward = obs_old_complexity - obs_new_complexity

        if self.normalize_rewards:
            max_reward, min_reward = self.reward_solved, self.reward_invalid_equation
            reward = 2 * (reward - min_reward) / (max_reward- min_reward) - 1

        return reward


    def get_valid_action_mask(self):
        return self.action_mask


# Testing the updated Env class
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = singleEqn()
    #check_env(env)
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        op, term = env.actions[action]
        next_state, reward, done, truncated, info = env.step(action)
        print(f'{env.lhs} = {env.rhs}, {operation_names[op]}, {term}')
        if done:
            if info['is_solved'] == True:
                print('solved!')
            elif info['too_many_steps'] == True:
                print('exceeded max length')




        

