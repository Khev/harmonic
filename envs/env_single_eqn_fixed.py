import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np

from sympy import sympify
from gymnasium import spaces, Env
from utils.utils_custom_functions import *
from operator import add, sub, mul, truediv
from sympy import symbols

#from sympy import symbols, simplify, factor, sqrt, Pow, Basic, Integer, Mul, Add, preorder_traversal, Poly

#from sympy import fraction,  collect, factor_list, denom
from operator import add, sub, mul, truediv
#from sympy.core.sympify import SympifyError

# Additional imports
from utils.utils_env import *

logger = logging.getLogger(__name__)

import signal
from contextlib import contextmanager

class SympyTimeout(Exception): pass

@contextmanager
def time_limit(seconds: float):
    # no-op on Windows or non-positive timeouts
    if os.name == "nt" or not seconds or seconds <= 0:
        yield
        return
    def _handler(signum, frame):
        raise SympyTimeout()
    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)

class singleEqn(Env):
    """Environment for solving linear equations using RL."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, main_eqn='a*x+b', state_rep='integer_1d', normalize_rewards=True, verbose=False, \
         cache=False) -> None:
        super().__init__()

        # Static parts
        self.max_expr_length = 20
        self.max_steps = 5
        self.action_dim = 12
        self.observation_dim = 2*self.max_expr_length+1
        self.sympy_timeout = getattr(self, "sympy_timeout", 0.05)  # seconds; tune (e.g., 0.02–0.1)

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
        self.action_mask = np.ones(self.action_dim)
        
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
        elif state_rep == 'graph_integer_1d':
            self.observation_space = spaces.Dict({
                "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float64),
                "edge_index": spaces.Box(low=0, high=self.observation_dim, shape=(2, 2*self.observation_dim), dtype=np.int32),
                "node_mask": spaces.Box(low=0, high=1, shape=(self.observation_dim,), dtype=np.int32),
                "edge_mask": spaces.Box(low=0, high=1, shape=(2*self.observation_dim,), dtype=np.int32),
            })
        elif state_rep == 'graph_integer_2d':
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

        a, b, c, x = symbols('a b c x')
        terms = [a, b, x]
        operations = [add, sub, mul, truediv]
        actions = [(op, term) for op in operations for term in terms]
        self.actions = actions
        #return actions

    def step(self, action_index):

        # Make actions: these are dynamic, so depend on current lhs, rhs
        lhs_old, rhs_old, obs_old = self.lhs, self.rhs, self.obs

        # Apply action
        action = self.actions[action_index]
        operation, term = action

        timed_out = False
        try:
            with time_limit(self.sympy_timeout):
                lhs_new = operation(lhs_old, term)
                rhs_new = operation(rhs_old, term)
                obs_new, complexity_obs_new = self.to_vec(lhs_new, rhs_new)
        except SympyTimeout:
            timed_out = True
            lhs_new, rhs_new, obs_new = lhs_old, rhs_old, obs_old
            reward = -1
            info = {}
            return obs_new, reward, False, False, info 
            
        # treat as illegal; keep previous state to avoid corrupting env
        lhs_new, rhs_new, obs_new = lhs_old, rhs_old, obs_old

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
            'main_eqn': self.main_eqn
        }

        if self.verbose:
            print(f'{self.lhs} = {self.rhs}. (Operation,term): {operation_names[operation]}, {term}')
        
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
            return graph_encoding_1d(lhs, rhs, self.feature_dict, self.max_expr_length)  
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




        


