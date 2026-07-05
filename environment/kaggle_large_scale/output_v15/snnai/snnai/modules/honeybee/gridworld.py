"""Grid-world environment for honeybee spatial navigation."""
import numpy as np


class GridWorld:
    """A simple NxN grid-world navigation task.

    The agent starts at ``start`` and receives a positive reward when it
    reaches ``goal``. Each step incurs a small penalty to encourage short
    trajectories.

    Parameters
    ----------
    size : int, optional
        Side length of the square grid (default 5).
    start : tuple[int, int], optional
        Initial agent position (default (0, 0)).
    goal : tuple[int, int], optional
        Target position (default (4, 4)).
    obstacles : sequence[tuple[int, int]] | None, optional
        Set of grid cells that cannot be entered (default None).
    """

    def __init__(self, size=5, start=(0, 0), goal=(4, 4), obstacles=None, shaped=False):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.shaped = shaped
        self.reset()

    def reset(self):
        """Reset the agent to the start position.

        Returns
        -------
        tuple[int, int]
            The current agent position.
        """
        self.pos = self.start
        self._prev_dist = self._dist(self.pos)
        return self.pos

    def _dist(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def step(self, action):
        """Execute one discrete action.

        Parameters
        ----------
        action : int
            Action index: 0=up, 1=down, 2=left, 3=right.

        Returns
        -------
        tuple[tuple[int, int], float, bool]
            The new position, the reward, and whether the goal was reached.
        """
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        next_pos = (
            self.pos[0] + moves[action][0],
            self.pos[1] + moves[action][1],
        )
        if (
            0 <= next_pos[0] < self.size
            and 0 <= next_pos[1] < self.size
            and next_pos not in self.obstacles
        ):
            self.pos = next_pos
        if self.pos == self.goal:
            reward = 10.0
            done = True
        else:
            d = self._dist(self.pos)
            if self.shaped:
                reward = -0.1 + (self._prev_dist - d) * 0.5
            else:
                reward = -0.1
            self._prev_dist = d
            done = False
        return self.pos, reward, done
