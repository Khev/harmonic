import numpy as np
import gym
from gym import spaces
from colorama import Fore, Style, init
init(autoreset=True)

class MazeEnvRandom(gym.Env):
    def __init__(self, mode="easy", train=True, max_steps=None):
        super(MazeEnvRandom, self).__init__()
        self.mode = mode
        self.train = train

        if mode == "easy":
            self.grid_size = (3, 3)
            self.obstacles = {(1, 1)}
            self.train_starts = [(0, 0), (2, 0), (0, 1), (1, 2), (2, 2)]
            self.train_goals = [(2, 2), (0, 2), (2, 1), (1, 0), (0, 0)]
            self.test_starts = [(1, 0), (2, 1), (0, 2), (1, 1), (2, 0)]
            self.test_goals = [(1, 2), (0, 1), (2, 2), (0, 0), (1, 0)]
            self.max_steps = 10 if max_steps is None else max_steps

        elif mode == "medium":
            self.grid_size = (4, 4)  # Reduced from 5x5
            self.obstacles = {
                (1, 1), (2, 2), (3, 1)  # Fewer obstacless
            }
            level = 5
            self.train_starts = [(0, 0), (3, 0), (0, 2), (2, 3), (1, 2)][:level]
            self.train_goals = [(3, 3), (2, 3), (1, 3), (0, 3), (3, 2)][:level]
            self.test_starts = [(1, 0), (2, 1), (3, 2), (0, 1), (1, 3)][:level]
            self.test_goals = [(2, 3), (1, 2), (3, 3), (0, 2), (2, 0)][:level]
            self.max_steps = 30 if max_steps is None else max_steps  # Fewer steps


        elif mode == "hard":
            self.grid_size = (10, 10)
            self.obstacles = {
                (1, 1), (1, 2), (1, 3), (1, 4),
                (4, 1), (4, 2), (4, 3), (4, 4),
                (6, 6), (7, 6), (8, 6), (8, 7),
            }
            self.train_starts = [(0, 0), (5, 0)]
            self.train_goals = [(9, 9), (5, 5)]
            self.test_starts = [(0, 9), (6, 0)]
            self.test_goals = [(9, 0), (6, 9)]
            self.max_steps = 100 if max_steps is None else max_steps

        else:
            raise ValueError("Mode must be 'easy', 'medium', or 'hard'.")


        self.reset()

        num_cells = self.grid_size[0] * self.grid_size[1]
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 * num_cells,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0: left, 1: right, 2: down, 3: up

    def _one_hot_encode(self, pos):
        """Converts (row, col) into one-hot vector"""
        num_cells = self.grid_size[0] * self.grid_size[1]
        one_hot = np.zeros(num_cells, dtype=np.float32)
        index = pos[0] * self.grid_size[1] + pos[1]
        one_hot[index] = 1.0
        return one_hot

    def _get_state(self):
        """Returns one-hot encoded agent + goal position"""
        return np.concatenate([self._one_hot_encode(self.agent_pos), self._one_hot_encode(self.goal_pos)])

    def reset(self):
        idx = np.random.choice(len(self.train_starts) if self.train else len(self.test_starts))
        self.agent_pos = self.train_starts[idx] if self.train else self.test_starts[idx]
        self.goal_pos = self.train_goals[idx] if self.train else self.test_goals[idx]
        self.steps = 0
        return self._get_state()

    def step(self, action):
        moves = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
        new_pos = (self.agent_pos[0] + moves[action][0], self.agent_pos[1] + moves[action][1])

        if new_pos == self.goal_pos:
            return self._get_state(), 10, True, {"info": "Goal Reached"}
        if new_pos in self.obstacles or not (0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]):
            self.steps += 1
            done = self.steps >= self.max_steps
            return self._get_state(), -0.1, done, {"info": "Hit Obstacle"}
        
        self.agent_pos = new_pos
        self.steps += 1
        done = self.steps >= self.max_steps
        return self._get_state(), -0.01, done, {"info": "Step Taken"}

    def render(self):
        """Renders the environment with color coding"""
        grid = [['.' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        
        for (r, c) in self.obstacles:
            grid[r][c] = Fore.RED + 'X' + Style.RESET_ALL

        ar, ac = self.agent_pos
        grid[ar][ac] = Fore.YELLOW + 'A' + Style.RESET_ALL

        gr, gc = self.goal_pos
        grid[gr][gc] = Fore.GREEN + 'G' + Style.RESET_ALL

        print("\n".join(" ".join(row) for row in grid), "\n")

# Example usage:
if __name__ == '__main__':
    env = MazeEnvRandom(mode="medium", train=True)
    env.render()
    obs = env.reset()
    print(f"Observation: {obs}")
    for _ in range(5):
        obs, reward, done, info = env.step(np.random.choice(4))
        env.render()
        print(f"Obs shape: {obs.shape}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            break
