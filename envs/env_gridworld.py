import numpy as np
import random
import gym
from gym import spaces

class gridWorld(gym.Env):
    def __init__(self, difficulty='hard', grid_size=None, start=None, goal=None, obstacles=None,
                 max_steps=20, randomize=False, possible_starts=None, possible_goals=None, mode="train"):
        super(gridWorld, self).__init__()
        
        self.difficulty = difficulty
        self.randomize = randomize
        self.mode = mode

        if grid_size is None or start is None or goal is None or obstacles is None:
            if difficulty == 'easy':
                self.grid_size = (5, 5)
                self.start = (0, 0)
                self.goal = (4, 4)
                self.obstacles = set()
            elif difficulty == 'hard':
                self.grid_size = (10, 10)
                self.start = (0, 0)
                self.goal = (9, 9)
                self.obstacles = self.generate_obstacles(1, 5)  # 10-20 obstacles
                self.max_steps = 50
        else:
            self.grid_size = grid_size
            self.start = start
            self.goal = goal
            self.obstacles = set(obstacles)

        self.max_steps = max_steps
        self.current_step = 0
        self.state = self.start
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size[0] * self.grid_size[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        
        self.goal_reward = 10
        self.step_reward = -0.01
        self.obstacle_penalty = -1
        self.timeout_penalty = 0
        self.illegal_penalty = -0.2

        # Fixed set of 5 start and goal positions
        #self.possible_starts = [(0, 0), (0, 9), (9, 0), (5, 5), (2, 7)]
        #self.possible_goals = [(9, 9), (9, 1), (1, 9), (7, 2), (4, 4)]

        self.possible_starts = possible_starts
        self.possible_goals = possible_goals

        if self.randomize:
            self.start, self.goal = self._select_fixed_positions()

    def _select_fixed_positions(self):
        idx = random.randint(0, 4)
        return self.possible_starts[idx], self.possible_goals[idx]

    def generate_obstacles(self, min_count, max_count):
        obstacles = set()
        num_obstacles = random.randint(min_count, max_count)
        num_obstacles = (max_count - min_count) // 2
        random.seed(14850)
        while len(obstacles) < num_obstacles:
            pos = (random.randint(0, self.grid_size[0] - 1), random.randint(0, self.grid_size[1] - 1))
            if pos != self.start and pos != self.goal:
                obstacles.add(pos)
        return obstacles

    def get_state_index(self, state):
        return state[0] * self.grid_size[1] + state[1]

    def to_one_hot(self, index):
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        state[index] = 1.0
        return state

    def reset(self):
        if self.randomize:
            self.start, self.goal = self._select_fixed_positions()
        self.state = self.start
        self.current_step = 0
        return self.to_one_hot(self.get_state_index(self.state))

    def step(self, action):
        moves = {
            0: (-1, 0),  # left
            1: (1, 0),   # right
            2: (0, -1),  # down
            3: (0, 1)    # up
        }
        dx, dy = moves[action]
        new_state = (self.state[0] + dx, self.state[1] + dy)

        if not (0 <= new_state[0] < self.grid_size[0] and 0 <= new_state[1] < self.grid_size[1]):
            reward = self.illegal_penalty
            done = False
            info = {}
            new_state = self.state
        elif new_state in self.obstacles:
            reward = self.obstacle_penalty
            done = False
            info = {"info": "Hit Obstacle"}
        elif new_state == self.goal:
            reward = self.goal_reward
            done = True
            info = {"info": "Goal Reached"}
        elif self.current_step >= self.max_steps - 1:
            reward = self.timeout_penalty
            done = True
            info = {"info": "Max Steps Reached"}
        else:
            reward = self.step_reward
            done = False
            info = {"info": "Step Taken"}

        self.state = new_state
        self.current_step += 1
        if self.current_step > self.max_steps:
            done = True
        return self.to_one_hot(self.get_state_index(new_state)), reward, done, info

    def render(self, mode='human'):
        grid = np.zeros(self.grid_size)
        for obs in self.obstacles:
            grid[obs] = -1
        grid[self.start] = 1
        grid[self.goal] = 2
        print(grid)
