import numpy as np
import random
import gym
from gym import spaces

class gridWorld(gym.Env):
    def __init__(self, difficulty='hard', grid_size=None, start=None, goal=None, obstacles=None,
                 max_steps=20, mode="train"):
        super(gridWorld, self).__init__()
        
        self.difficulty = difficulty
        self.randomize = randomize
        self.mode = mode

        if mode == 'train':
            # Possibly fewer obstacles or simpler start/goal pattern
            # e.g. "easy training set"
            self.grid_size = (10, 10)
            self.start = (0, 0)
            self.goal = (9, 9)
            self.obstacles = self.generate_obstacles(5, 5)  
            self.max_steps = 40
        elif mode == 'test':
            # Possibly more obstacles or different arrangement
            # e.g. "harder test set" or just "different" so agent can't memorize
            self.grid_size = (10, 10)
            self.start = (0, 0)
            self.goal = (9, 9)
            self.obstacles = self.generate_obstacles(5, 5)  
            self.max_steps = 40

        # define observation_space, action_space, etc. as before
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.grid_size[0]*self.grid_size[1],),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.goal_reward = 10
        self.step_reward = -0.01
        self.obstacle_penalty = -1
        self.timeout_penalty = 0
        self.illegal_penalty = -0.2

        self.current_step = 0
        self.state = self.start

    def generate_obstacles(self, min_count, max_count):
        # for example, create a set of 'num_obstacles' obstacles randomly
        # ...
        # or define them deterministically if you want a consistent test set

        seed_train, seed_test = 14850, 98112

        if self.mode = 'train'
            random.seed(seed_train)
            obstacles = set()
            num_obstacles = random.randint(min_count, max_count)
            while len(obstacles) < num_obstacles:
                r = random.randint(0, self.grid_size[0]-1)
                c = random.randint(0, self.grid_size[1]-1)
                if (r,c) not in [self.start, self.goal]:
                    obstacles.add((r,c))
        else:
            random.seed(seed_test)
            obstacles = set()
            num_obstacles = random.randint(min_count, max_count)
            while len(obstacles) < num_obstacles:
                r = random.randint(0, self.grid_size[0]-1)
                c = random.randint(0, self.grid_size[1]-1)
                if (r,c) not in [self.start, self.goal]:
                    obstacles.add((r,c))

        return obstacles

    def get_state_index(self, state):
        return state[0]*self.grid_size[1] + state[1]

    def to_one_hot(self, index):
        arr = np.zeros(self.grid_size[0]*self.grid_size[1], dtype=np.float32)
        arr[index] = 1.0
        return arr

    def reset(self):
        self.current_step = 0
        self.state = self.start
        return self.to_one_hot(self.get_state_index(self.start))

    def step(self, action):
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }
        dx, dy = moves[action]
        new_r = self.state[0] + dx
        new_c = self.state[1] + dy
        done = False
        info = {}

        if not (0 <= new_r < self.grid_size[0] and 0 <= new_c < self.grid_size[1]):
            reward = self.illegal_penalty
            new_state = self.state
        elif (new_r, new_c) in self.obstacles:
            reward = self.obstacle_penalty
            new_state = self.state
        elif (new_r, new_c) == self.goal:
            reward = self.goal_reward
            done = True
            info["info"] = "Goal Reached"
            new_state = (new_r, new_c)
        elif self.current_step >= self.max_steps - 1:
            reward = self.timeout_penalty
            done = True
            info["info"] = "Timeout"
            new_state = (new_r, new_c)
        else:
            reward = self.step_reward
            new_state = (new_r, new_c)

        self.state = new_state
        self.current_step += 1
        obs = self.to_one_hot(self.get_state_index(self.state))
        return obs, reward, done, info

    def render(self, mode='human'):
        # optional textual or ASCII rendering
        pass
