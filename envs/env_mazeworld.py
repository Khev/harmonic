import numpy as np
import gym
from gym import spaces
from colorama import Fore, Style, init
init(autoreset=True)

class MazeEnv(gym.Env):
    def __init__(self, mode="medium", max_steps=None):
        super(MazeEnv, self).__init__()
        self.mode = mode
        
        if mode == "easy":
            # New easy mode: 3x3 maze with an obstacle that forces down, right, then up.
            self.grid_size = (3, 3)
            self.start = (0, 0)
            self.goal = (0, 2)
            # Place an obstacle at (0,1) so the direct path is blocked.
            self.obstacles = {(0, 1)}
            self.max_steps = 10 if max_steps is None else max_steps
        elif mode == 'easy_medium':
            self.grid_size = (3, 4)
            self.start = (0, 0)
            self.goal = (0, 2)
            self.obstacles = {(0, 1),(1,1),(1,2)}
            self.max_steps = 50 if max_steps is None else max_steps
        elif mode == "medium":
            # Current easy mode rebranded as medium: 5x5 maze.
            self.grid_size = (5, 5)
            self.start = (0, 0)
            self.goal = (0, 4)
            # Walls arranged to force a detour.
            self.obstacles = {(0, 1), (1, 1), (2, 1), (3, 1), (3, 2)}
            self.max_steps = 50 if max_steps is None else max_steps
        elif mode == "medium_hard":
            # Current easy mode rebranded as medium: 5x5 maze.
            self.grid_size = (5, 5)
            self.start = (0, 0)
            self.goal = (1, 2)
            # Walls arranged to force a detour.
            self.obstacles = {(0, 1), (1, 1), (2, 1), (3, 1), (3, 2),(2,3),(3,3)}
            self.max_steps = 50 if max_steps is None else max_steps
        elif mode == "hard":
            self.grid_size = (10, 10)
            self.start = (0, 0)
            self.goal = (3, 2)
            # A more complex, deterministic maze for 10x10:
            self.obstacles = {
                # Vertical wall on column 1 from row 0 to 4
                (0,1), (1,1), (2,1), (3,1), (4,1),
                # Horizontal wall from (4,1) to (4,7)
                (4,2), (4,3), (4,4), (4,5), (4,6), (4,7),
                # Vertical wall on column 7 from row 4 to 8
                (5,7), (6,7), (7,7), (8,7),
                # Additional obstacles to form turns
                (2,3), (3,3), (6,2), (7,2), (7,3), (7,4)
            }
            self.max_steps = 100 if max_steps is None else max_steps
        else:
            raise ValueError("mode must be either 'easy', 'medium', or 'hard'.")

        self.current_step = 0
        self.state = self.start

        # One-hot encoding of the grid cell index as observation.
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.grid_size[0] * self.grid_size[1],), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # 0: left, 1: right, 2: down, 3: up

        # Reward structure
        self.goal_reward = 10
        self.step_reward = -0.01
        self.obstacle_penalty = -0.1  #-0.1
        self.illegal_penalty = -0.1

    def get_state_index(self, state):
        # Convert (row, col) to a flat index.
        return state[0] * self.grid_size[1] + state[1]

    def to_one_hot(self, index):
        vec = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        vec[index] = 1.0
        return vec

    def reset(self):
        self.state = self.start
        self.current_step = 0
        return self.to_one_hot(self.get_state_index(self.state))

    def step(self, action):
        # Define actions: 0: left, 1: right, 2: down, 3: up
        moves = {
            0: (0, -1),
            1: (0, 1),
            2: (1, 0),
            3: (-1, 0)
        }
        dx, dy = moves[action]
        new_state = (self.state[0] + dx, self.state[1] + dy)

        # Check boundaries:
        if not (0 <= new_state[0] < self.grid_size[0] and 0 <= new_state[1] < self.grid_size[1]):
            reward = self.illegal_penalty
            done = False
            info = {"info": "Hit boundary"}
            new_state = self.state
        elif new_state in self.obstacles:
            reward = self.obstacle_penalty
            done = False
            info = {"info": "Hit obstacle"}
            new_state = self.state
        elif new_state == self.goal:
            reward = self.goal_reward
            done = True
            info = {"info": "Goal Reached"}
            #print('solved!')
        else:
            reward = self.step_reward
            done = False
            info = {"info": "Step taken"}

        self.state = new_state
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self.to_one_hot(self.get_state_index(new_state)), reward, done, info

    def render(self, mode="human"):
        # Create an empty grid (list of lists) filled with dots.
        rows, cols = self.grid_size
        grid = [['.' for _ in range(cols)] for _ in range(rows)]
        
        # Place obstacles (X in red)
        for (r, c) in self.obstacles:
            grid[r][c] = Fore.RED + 'X' + Style.RESET_ALL

        # Mark the start (S in cyan)
        sr, sc = self.start
        grid[sr][sc] = Fore.CYAN + 'S' + Style.RESET_ALL

        # Mark the goal (G in green)
        gr, gc = self.goal
        grid[gr][gc] = Fore.GREEN + 'G' + Style.RESET_ALL

        # Mark the current position (A in yellow), if it's not the start or goal
        cr, cc = self.state
        if (cr, cc) != self.start and (cr, cc) != self.goal:
            grid[cr][cc] = Fore.YELLOW + 'A' + Style.RESET_ALL

        # Print the grid row by row
        for row in grid:
            print(" ".join(row))

# Example usage:
if __name__ == '__main__':
    print("Easy mode (3x3):")
    env_easy = MazeEnv(mode="easy")
    env_easy.render()
    print("Easy Medium mode (3x3):")
    env = MazeEnv(mode="easy_medium")
    env.render()
    print("\nMedium mode (5x5):")
    env_med = MazeEnv(mode="medium")
    env_med.render()
    print("Medium hard mode (3x3):")
    env = MazeEnv(mode="medium_hard")
    env.render()
    print("\nHard mode (10x10):")
    env_hard = MazeEnv(mode="hard")
    env_hard.render()
