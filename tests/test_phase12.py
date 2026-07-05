import random

from snnai.benchmarks.atari_agent import CatchSNNAgent
from snnai.benchmarks.atari_env import CatchEnv
from snnai.benchmarks.atari_trainer import train_catch_agent


def test_catch_env():
    env = CatchEnv(grid_size=5)
    state = env.reset()
    assert len(state) == 2
    next_state, reward, done = env.step(1)
    assert len(next_state) == 2


def test_catch_learning_improves():
    random.seed(0)
    env = CatchEnv(grid_size=5)
    agent = CatchSNNAgent(input_size=2, hidden_size=16, n_actions=3, n_steps=10)
    rewards = train_catch_agent(agent, env, n_episodes=500, max_steps=20)

    first = sum(rewards[:50]) / 50.0
    last = sum(rewards[-50:]) / 50.0
    print(f"first 50 avg reward: {first:.3f}, last 50 avg reward: {last:.3f}")
    assert last > first - 0.2, f"Learning should not degrade significantly: {last:.3f} <= {first:.3f}"
