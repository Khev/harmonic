import gym
from gym import spaces
import numpy as np


class colorMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(colorMazeEnv, self).__init__()
        # Maze dimensions
        self.width = 10
        self.height = 10
        self.max_steps = 100
        self.current_step = 0
        self.difficulty = 'hard'

        # Define the maze layout (1 for path, 0 for wall)
        self.maze = np.array([
            [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
        ])
        # Define start points and their corresponding goals by color
        self.start_points = {
            'red': (0, 0),
            'blue': (9, 0),
            'green': (0, 9)
        }
        self.goals = {
            'red': (9, 9),
            'blue': (9, 9),
            'green': (9, 0)
        }
        # Current state (position and goal color)
        self.state = None
        self.current_goal_color = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.width * self.height + 3,), dtype=np.float32)


    def step(self, action):
        # Map action to direction

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        dx, dy = directions[action]
        new_x, new_y = self.state[0] + dx, self.state[1] + dy

        # Check if the new position is valid (within maze and not a wall)
        info = {'solved':False}
        if 0 <= new_x < self.width and 0 <= new_y < self.height and self.maze[new_y, new_x] == 1:
            new_state = (new_x, new_y)
        else:
            # Hitting a wall or out of bounds gives a penalty
            new_state, reward, done, info = self.state, -1, False, {}

        # Counters
        self.state = new_state

        # Check if goal reached
        if self.state == self.goals[self.current_goal_color]:
            info = {"info": "Goal Reached"}
            reward = 10
            done = True
            #print(f'Solved: {self.current_goal_color}')
        else:
            reward = -0.1  # Small penalty for each step
            done = False

        self.current_step += 1
        if self.current_step > self.max_steps:
            done = True

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        obs = np.zeros((self.width * self.height + 3,), dtype=np.float32)
        # Encode agent position
        idx = self.state[1] * self.width + self.state[0]
        obs[idx] = 1.0
        # Encode goal color
        obs[-3 + self.color_to_index(self.current_goal_color)] = 1.0
        return obs

    def reset(self):
        if self.difficulty == 'hard':
            self.current_goal_color = np.random.choice(list(self.start_points.keys()))
        else:
            self.current_goal_color = list(self.start_points.keys())[1]
        self.state = self.start_points[self.current_goal_color]
        self.current_step = 0
        return self._get_observation()

    def render(self, mode='human'):
        # Simple rendering of the maze with agent position and goal
        maze_str = ""
        for y in range(self.height):
            for x in range(self.width):
                if self.state == (x, y):
                    maze_str += 'A'  # Agent
                elif (x, y) == self.goals[self.current_goal_color]:
                    maze_str += self.current_goal_color[0].upper()
                elif self.maze[y, x] == 0:
                    maze_str += '█'  # Wall
                else:
                    maze_str += '·'  # Path
            maze_str += '\n'
        print(maze_str)

    def color_to_index(self, color):
        return {'red': 0, 'blue': 1, 'green': 2}[color]


# Example usage:
if __name__ == "__main__":
    env = colorMazeEnv()
    state = env.reset()
    for _ in range(100):
        action = env.action_space.sample()  # Random action
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            print(f"Goal reached! Reward: {reward}")
            break
