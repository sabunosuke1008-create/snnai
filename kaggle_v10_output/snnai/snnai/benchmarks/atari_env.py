"""Minimal Atari-style catch environment for SNN RL."""
import random


class CatchEnv:
    """A simple 5x5 catch game.

    A ball falls from the top. The agent controls a paddle at the
    bottom and must catch the ball.

    State: (ball_x, ball_y, paddle_x) as floats in [0, grid_size).
    Actions: 0=left, 1=stay, 2=right.
    """

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.ball_x = random.randint(0, self.grid_size - 1)
        self.ball_y = 0
        self.paddle_x = self.grid_size // 2
        return self._get_state()

    def _get_state(self):
        # Relative horizontal distance and ball height (normalized)
        dx = (self.ball_x - self.paddle_x) / self.grid_size
        dy = self.ball_y / self.grid_size
        return (float(dx), float(dy))

    def step(self, action):
        # Move paddle
        if action == 0 and self.paddle_x > 0:
            self.paddle_x -= 1
        elif action == 2 and self.paddle_x < self.grid_size - 1:
            self.paddle_x += 1

        # Move ball down
        self.ball_y += 1

        # Check catch/miss
        if self.ball_y == self.grid_size - 1:
            if self.paddle_x == self.ball_x:
                reward = 1.0
                done = True
            else:
                reward = -1.0
                done = True
        else:
            reward = 0.0
            done = False

        return self._get_state(), reward, done
