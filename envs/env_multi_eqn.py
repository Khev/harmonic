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
from collections import defaultdict

logger = logging.getLogger(__name__)

class multiEqn(Env):
    """
    Environment for solving multiple equations using RL, 
    with a simple curriculum that samples equations inversely 
    proportional to how often they've been solved.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 state_rep='integer_1d', 
                 normalize_rewards=True, 
                 verbose=False,
                 cache=False, 
                 level=3, 
                 generalization='random') -> None:
        super().__init__()

        # Static parts
        self.max_expr_length = 20
        self.max_steps = 10
        self.action_dim = 50
        self.observation_dim = 2*self.max_expr_length + 1

        # Rewards
        self.reward_solved = +100
        self.reward_invalid_equation = -100
        self.reward_illegal_action = -1

        # Optimizing
        self.cache = cache
        if self.cache:
            self.action_cache = {}

        # Pars
        self.normalize_rewards = normalize_rewards
        self.state_rep = state_rep
        self.verbose = verbose

        # Load train/test equations
        self.level = level
        self.generalization = generalization
        eqn_dirn = f"equation_templates"
        # self.train_eqns, self.test_eqns = load_train_test_equations(
        #     eqn_dirn, level, generalization=generalization
        # )

        self.train_eqns = ['a*x','x+b','a*x+b','a/x+b','c*(a*x+b)+d','d/(a*x+b)+c','e*(a*x+b)+(c*x+d)', '(a*x+b)/(c*x+d)+e']
        self.test_eqns = ['f*x', 'x+g', 'f*x+g', 'f/x+g', 'h*(f*x+g)+i', 'i/(f*x+g)+h', 'j*(f*x+g)+(h*x+i)', '(f*x+g)/(h*x+i)+j']

        self.train_eqns = self.train_eqns[:self.level]
        self.test_eqns = self.test_eqns[:self.level]

        print(f'Train_eqns = {self.train_eqns}')
        print(f'Test_eqns = {self.test_eqns}')

        self.train_eqns = [sympify(eqn) for eqn in self.train_eqns]
        self.test_eqns = [sympify(eqn) for eqn in self.test_eqns]

        # Tracking how many times we've solved each eqn
        # Use a dict with keys = the actual sympy expression or string
        self.solve_counts = defaultdict(int)
        self.sample_counts = defaultdict(int)

        # Convert each eqn to a canonical string so we can store counts easily
        self.train_eqns_str = [str(eq) for eq in self.train_eqns]
        self.test_eqns_str = [str(eq) for eq in self.train_eqns]


        # Random initial eqn
        eqn_str = np.random.choice(self.train_eqns_str)
        self.main_eqn = sympify(eqn_str)
        self.lhs = self.main_eqn
        self.rhs = 0
        self.x = symbols('x')

        #  Make feature_dict, actions etc
        self.setup()

        # RL env setup
        self.state = self.to_vec(self.lhs, self.rhs)
        self.action_space = spaces.Discrete(self.action_dim)

        if state_rep == 'integer_1d':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.observation_dim,), 
                dtype=np.float64
            )
        elif state_rep == 'integer_2d':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.observation_dim, 2), 
                dtype=np.float64
            )
        elif state_rep in ['graph_integer_1d', 'graph_integer_2d']:
            self.observation_space = spaces.Dict({
                "node_features": spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(self.observation_dim, 2), 
                    dtype=np.float64
                ),
                "edge_index": spaces.Box(
                    low=0, high=self.observation_dim, 
                    shape=(2, 2*self.observation_dim), 
                    dtype=np.int32
                ),
                "node_mask": spaces.Box(
                    low=0, high=1, 
                    shape=(self.observation_dim,), 
                    dtype=np.int32
                ),
                "edge_mask": spaces.Box(
                    low=0, high=1, 
                    shape=(2*self.observation_dim,), 
                    dtype=np.int32
                ),
            })
        else:
            raise ValueError(f"Unsupported state representation: {state_rep}")


    def setup(self):
        # Build feature dict from all train eqns
        self.feature_dict = make_feature_dict_multi(
            self.train_eqns, self.test_eqns, self.state_rep
        )

        # Define some fixed 'global' transformations
        self.actions_fixed = [
            (custom_expand, None),
            (custom_factor, None),
            (custom_collect, self.x),
            (custom_together, None),
            (custom_ratsimp, None),
            (custom_square, None),
            (custom_sqrt, None),
            (mul, -1),
        ]

        if self.cache:
            self.actions, self.action_mask = make_actions_cache(
                self.lhs, self.rhs, self.actions_fixed, 
                self.action_dim, self.action_cache
            )
        else:
            self.actions, self.action_mask = make_actions(
                self.lhs, self.rhs, self.actions_fixed, self.action_dim
            )


    def step(self, action_index):
        lhs_old, rhs_old, obs_old = self.lhs, self.rhs, self.state

        # Re‐compute dynamic action list for the current state
        if self.cache:
            action_list, action_mask = make_actions_cache(
                lhs_old, rhs_old, self.actions_fixed,
                self.action_dim, self.action_cache
            )
        else:
            action_list, action_mask = make_actions(
                lhs_old, rhs_old, self.actions_fixed, self.action_dim
            )

        self.actions = action_list
        self.action_mask = action_mask

        # Apply selected action
        operation, term = action_list[action_index]
        lhs_new = operation(lhs_old, term)
        rhs_new = operation(rhs_old, term)
        obs_new, _ = self.to_vec(lhs_new, rhs_new)

        # Check validity
        is_valid_eqn, lhs_new, rhs_new = check_valid_eqn(lhs_new, rhs_new)
        is_solved = check_eqn_solved(lhs_new, rhs_new, self.main_eqn)

        # Compute reward
        reward = self.find_reward(lhs_old, rhs_old, lhs_new, rhs_new, is_valid_eqn, is_solved)

        # Termination
        too_many_steps = (self.current_steps >= self.max_steps)
        terminated = bool(is_solved or too_many_steps or not is_valid_eqn)
        truncated = False

        # If solved, update solve_counts for this eqn
        if is_solved:
            eqn_str = str(self.main_eqn)
            self.solve_counts[eqn_str] += 1

        # Update environment state
        self.lhs, self.rhs = lhs_new, rhs_new
        self.state = obs_new
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
            print(f"{self.lhs} = {self.rhs}. (Operation, term): ({operation_names[operation]}, {term})")

        return obs_new, reward, terminated, truncated, info


    def reset(self, seed=None, **kwargs):

        options = kwargs.get('options', 'train')
        # If options is a dict, extract the 'mode' key, defaulting to 'train'
        if isinstance(options, dict):
            options = options.get('mode', 'train')

        if options == 'curriculum':
            eqn_probs = []
            for eqn_str in self.train_eqns_str:
                eqn_probs.append( 1.0 / (1 + self.solve_counts[eqn_str]) )
            eqn_probs = np.array(eqn_probs, dtype=np.float64)
            eqn_probs /= eqn_probs.sum()
            chosen_eqn_str = np.random.choice(self.train_eqns_str, p=eqn_probs)

        elif options in ['train', None]:
            chosen_eqn_str = np.random.choice(self.train_eqns)
        elif options == 'test':
            chosen_eqn_str = np.random.choice(self.test_eqns)
        else:
            ValueError('Wrong option')

        self.main_eqn = sympify(chosen_eqn_str)

        # Increment the count for how many times this eqn has been sampled
        self.sample_counts[chosen_eqn_str] += 1

        self.current_steps = 0
        self.lhs, self.rhs = self.main_eqn, 0
        obs, _ = self.to_vec(self.lhs, self.rhs)
        self.state = obs

        # Recompute actions, masks, etc.
        self.setup()

        return obs, {}



    def to_vec(self, lhs, rhs):
        """
        Convert (lhs, rhs) to a suitable observation 
        given self.state_rep.
        """
        if self.state_rep == 'integer_1d':
            return integer_encoding_1d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep == 'integer_2d':
            return integer_encoding_2d(lhs, rhs, self.feature_dict, self.max_expr_length)
        elif self.state_rep in ['graph_integer_1d', 'graph_integer_2d']:
            return graph_encoding(lhs, rhs, self.feature_dict, self.max_expr_length)
        else:
            raise ValueError(f"Unknown state_rep: {self.state_rep}")


    def find_reward(self, lhs_old, rhs_old, lhs_new, rhs_new, is_valid_eqn, is_solved):
        """
        Reward = 
          +100 if solved
          -100 if invalid eqn
          else ( oldComplexity - newComplexity )
               optionally normalized to [-1, 1].
        """
        if not is_valid_eqn:
            reward = self.reward_invalid_equation
        elif is_solved:
            reward = self.reward_solved
        else:
            old_complex = get_complexity_expression(lhs_old) + get_complexity_expression(rhs_old)
            new_complex = get_complexity_expression(lhs_new) + get_complexity_expression(rhs_new)
            reward = old_complex - new_complex

        if self.normalize_rewards:
            # rescale reward to [-1, 1]
            # min=-100, max=+100
            min_r, max_r = self.reward_invalid_equation, self.reward_solved
            reward = 2.0 * (reward - min_r) / float(max_r - min_r) - 1.0

        return reward

    def render(self, mode="human"):
        print(f"{self.lhs} = {self.rhs}")

    def get_valid_action_mask(self):
        return self.action_mask

    def set_equation(self,main_eqn):
        self.main_eqn, self.lhs, self.rhs = main_eqn, main_eqn, 0
        obs, _ = self.to_vec(self.lhs, self.rhs)
        self.state = obs
        return obs

# Example usage:
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = multiEqn()
    # optional check for Gym API compliance
    # check_env(env, warn=True)

    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        if done:
            if info['is_solved']:
                print("Solved eqn:", info['main_eqn'])
            elif info['too_many_steps']:
                print("Too many steps, giving up on eqn:", info['main_eqn'])
