import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from collections import deque
import pygame

import logging
logger = logging.getLogger(__name__)

class FrozenLakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def generate_grid(self, n):
        def check_valid(grid, n):
            # Use BFS to check that path from start and end exists
            visited = set()
            q = deque()
            q.append((0,0))
            while q:
                r, c = q.popleft()
                if not (r, c) in visited:
                    visited.add((r, c))
                    # Explore neighbours up, down, left, right
                    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                    for x, y in directions:
                        r_new = r + x
                        c_new = c + y
                        if r_new < 0 or r_new >= n or c_new < 0 or c_new >= n:
                            continue
                        if grid[r_new][c_new] == "G":
                            return True
                        if grid[r_new][c_new] != "H":
                            q.append((r_new, c_new))
            return False

        valid = False
        while not valid:
            grid = [["F" for i in range(n)] for j in range(n)]
            # distribute holes over frozen lake with 25% of the tiles as holes
            num_holes = n*n//4
            tiles = [i for i in range(1, n*n-1)]
            holes = np.random.choice(tiles, num_holes, replace=False)
            for hole in holes:
                r = hole // n
                c = hole % n
                grid[r][c] = "H"

            grid[0][0] = "S"
            grid[-1][-1] = "G"
            valid = check_valid(grid, n)
            # print(grid)
        return grid

    def __init__(self, kwargs, n=4):
        """
        Description:
          A robot starts at the top left of a grid, and is tasked to travel to
          the bottom right of the grid, without falling into any holes

          Observation:
            Type: Tuple(Discrete(n), Discrete(n))
            4x4 grid
            Num    Observation
            0      first row/col
            1      second row/col
            2      third row/col
            3      fourth row/col
            ...    ...

          Action:
            Type: Discrete(4)
            Num    Action
            0      Up
            1      Right
            2      Down
            3      Left

          Reward:
            Reward is +1 when it reaches the frisbee at the bottom right. i.e. lands on goal "G"
            Reward is -1 when it falls into a hole. i.e. lands on hole "H"
            Reward is 0 for all other cases

          Starting State:
            Robot starts at top left corner (0, 0)

          Episode Termination:
            When robot reaches the frisbee or falls into a hole

        """
        self.grid_size = n
        self.observation_space = spaces.Tuple((spaces.Discrete(n), spaces.Discrete(n)))
        self.action_space = spaces.Discrete(4)
        self.grid = self.generate_grid(n)
        self.viewer = None
        self.state = np.array([0, 0])
        self.hide_video = kwargs['video']

        if not self.hide_video:
            # Create pygame environment
            self.screen_width = n * 60 + 100
            self.screen_height = n * 60
            pygame.init()
            pygame.display.set_caption('Frozen Lake RL')
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)
            self.robot = pygame.image.load("robot.png")
            self.robot = pygame.transform.scale(self.robot, (60, 60))
            self.map = pygame.image.load("frozen lake.jpg")
            self.map = pygame.transform.scale(self.map, (self.screen_width, self.screen_height))
            self.hole = pygame.image.load("hole.png")
            self.hole = pygame.transform.scale(self.hole, (60, 60))
            self.goal = pygame.image.load("frisbee.png")
            self.goal = pygame.transform.scale(self.goal, (60, 60))

            self.draw(self.grid_size, self.state, episode=0)

            self.game_speed = 60


    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, y = self.state
        if action == 0:
            y = max(0, y-1)
        elif action == 1:
            x = min(self.grid_size-1, x+1)
        elif action == 2:
            y = min(self.grid_size-1, y+1)
        elif action == 3:
            x = max(0, x-1)
        self.state = (x, y)

        done = True if (self.grid[x][y] == "H" or self.grid[x][y] == "G") else False

        reward = 0
        if self.grid[x][y] == "H":
            reward -= 1
        elif self.grid[x][y] == "G":
            reward += 1
        # print("State: ({},{})".format(x,y))

        return np.array(self.state, dtype=np.int), reward, done, {}

    def reset(self):
        self.state = np.array([0, 0])
        return self.state

    def draw(self, n, state, episode=0):
        x, y = state
        self.screen.blit(self.map, (0, 0))
        for i in range(n + 1):  # draw grid lines
            pygame.draw.line(self.screen, color=(0, 0, 0), start_pos=(100, 60 * i), end_pos=(100 + 60 * n, 60 * i),
                             width=3)
            pygame.draw.line(self.screen, color=(0, 0, 0), start_pos=(100 + 60 * i, 0), end_pos=(100 + 60 * i, 60 * n),
                             width=3)
        text = self.font.render("Episode {}".format(episode), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (50, 30)
        self.screen.blit(text, text_rect)

        self.screen.blit(self.robot, (100 + y * 60, x * 60))
        self.screen.blit(self.goal, (100 + (n - 1) * 60, (n - 1) * 60))
        for i in range(n):
            for j in range(n):
                if self.grid[i][j] == "H":
                    self.screen.blit(self.hole, (100 + j * 60, i * 60))
        pygame.display.flip()

    def render(self, episode=0, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # and end the rl simulation

        self.draw(self.grid_size, self.state, episode)
        self.clock.tick(self.game_speed)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None