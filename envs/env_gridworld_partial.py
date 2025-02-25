import gym
import numpy as np
import random
from gym import spaces

class gridWorldPartial(gym.Env):
    """
    A simple partially observable GridWorld:
    - 10x10 grid by default
    - An agent starts at a random free cell
    - A goal is placed in another random free cell
    - A small window around the agent's position is returned as the observation (partial observability).
    """

    def __init__(
        self,
        grid_size=(10, 10),
        max_steps=50,
        view_radius=2,
        n_obstacles=10,
        goal_reward=10.0,
        step_reward=-0.01,
        obstacle_penalty=-1.0,
        illegal_penalty=-0.2
    ):
        """
        Args:
            grid_size: (rows, cols) of the grid
            max_steps: episode ends after this many steps if not done
            view_radius: defines the size of partial observation window:
                         final obs is (2*view_radius+1, 2*view_radius+1)
            n_obstacles: number of random obstacles placed at reset
            goal_reward: reward for reaching the goal
            step_reward: reward for a normal step
            obstacle_penalty: reward if the agent walks into an obstacle
            illegal_penalty: reward if the agent tries to step outside the grid
        """
        super(gridWorldPartial, self).__init__()

        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.max_steps = max_steps
        self.view_radius = view_radius

        # Rewards
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.obstacle_penalty = obstacle_penalty
        self.illegal_penalty = illegal_penalty

        self.n_obstacles = n_obstacles
        self.current_step = 0

        # We'll store the grid as a 2D array: 0 = empty, -1 = obstacle, 2 = goal
        self.grid = np.zeros(self.grid_size, dtype=np.int32)

        # (row, col) for agent and goal
        self.agent_pos = None
        self.goal_pos = None

        # 4 discrete actions: 0=left,1=right,2=up,3=down
        self.action_space = spaces.Discrete(4)

        # Partial observation shape: (2*view_radius + 1, 2*view_radius + 1)
        obs_shape = (2*self.view_radius + 1, 2*self.view_radius + 1)
        # We'll store small integers: {0=empty, -1=obstacle, 2=goal, 1=agent} or similar
        # but to pass a Box, we can keep it in range [-1..2] or store them as floats
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0,
            shape=obs_shape,
            dtype=np.float32
        )

    def reset(self):
        """
        Reset the environment:
          1. Clears the grid
          2. Places random obstacles
          3. Places the goal
          4. Places the agent
          5. Returns the partial observation
        """
        self.grid = np.zeros(self.grid_size, dtype=np.int32)
        # Place obstacles randomly
        self._place_obstacles(self.n_obstacles)

        # Place goal
        self.goal_pos = self._random_free_cell()
        gr, gc = self.goal_pos
        self.grid[gr, gc] = 2  # Mark goal with '2'

        # Place agent
        self.agent_pos = self._random_free_cell()
        self.current_step = 0

        return self._get_obs()

    def step(self, action):
        """
        Agent actions:
          0: left, 1: right, 2: up, 3: down
        """
        self.current_step += 1

        # Map actions to row/col changes
        moves = {
            0: (0, -1),   # left
            1: (0,  1),   # right
            2: (-1, 0),   # up
            3: (1,  0),   # down
        }
        dr, dc = moves[action]

        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Check boundaries
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            # Illegal move
            reward = self.illegal_penalty
            done = False
            info = {"reason": "illegal_move"}
        else:
            # Inside grid
            cell_val = self.grid[new_r, new_c]
            if cell_val == -1:
                # Obstacle
                reward = self.obstacle_penalty
                done = False
                info = {"reason": "hit_obstacle"}
            elif cell_val == 2:
                # Goal
                reward = self.goal_reward
                done = True
                info = {"reason": "goal_reached"}
                self.agent_pos = (new_r, new_c)
            else:
                # Normal cell
                reward = self.step_reward
                done = False
                info = {"reason": "moved"}
                self.agent_pos = (new_r, new_c)

        # Check max steps
        if self.current_step >= self.max_steps:
            done = True

        obs = self._get_obs()
        return obs, reward, done, info

    def render(self, mode='human'):
        """
        Print a text-based representation. 
        0 = empty, -1 = obstacle, 2 = goal, A = agent
        """
        grid_str = ""
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    row_str += "A "
                elif self.grid[r, c] == -1:
                    row_str += "X "
                elif self.grid[r, c] == 2:
                    row_str += "G "
                else:
                    row_str += ". "
            grid_str += row_str + "\n"
        print(grid_str)

    # --------------------------
    # Internal helper methods
    # --------------------------
    def _get_obs(self):
        """
        Returns a (2*view_radius+1, 2*view_radius+1) partial view around the agent.
        We'll map the agent's cell to '1' in the returned array, goal to '2', obstacle to '-1', empty to '0'.
        Any area outside the grid is filled with -1 to represent "unseen" or "invalid".
        """
        pr = self.view_radius
        ar, ac = self.agent_pos

        # We'll create an array for the local region
        obs_shape = (2*pr+1, 2*pr+1)
        local_obs = np.ones(obs_shape, dtype=np.float32) * -1.0

        # Determine the bounds in the global grid
        rmin = ar - pr
        rmax = ar + pr
        cmin = ac - pr
        cmax = ac + pr

        for rr in range(rmin, rmax+1):
            for cc in range(cmin, cmax+1):
                rr_local = rr - rmin
                cc_local = cc - cmin
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    cell_val = self.grid[rr, cc]
                    if (rr, cc) == self.agent_pos:
                        local_obs[rr_local, cc_local] = 1.0  # agent
                    else:
                        local_obs[rr_local, cc_local] = float(cell_val)

        return local_obs

    def _random_free_cell(self):
        """
        Returns a random cell in the grid that is not an obstacle and not the goal (if already placed).
        """
        while True:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            if self.grid[r, c] == 0:  # empty
                return (r, c)

    def _place_obstacles(self, n):
        """
        Places n obstacles randomly in the grid.
        """
        self.grid[:] = 0  # first ensure empty
        for _ in range(n):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            self.grid[r, c] = -1

