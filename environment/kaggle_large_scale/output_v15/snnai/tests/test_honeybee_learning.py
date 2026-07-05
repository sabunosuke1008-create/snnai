import torch
from snnai.modules.honeybee import SNNAgent, GridWorld
from snnai.modules.honeybee.training import train_agent


def test_learning_curve_improves():
    torch.manual_seed(0)
    env = GridWorld(size=3, start=(0, 0), goal=(2, 2), obstacles=[], shaped=True)
    agent = SNNAgent(size=3, n_cells=25, hidden_size=20, n_steps=20)
    rewards = train_agent(agent, env, n_episodes=80, max_steps=15)

    first = sum(rewards[:10]) / 10.0
    last = sum(rewards[-10:]) / 10.0
    print(f"first 10 avg reward: {first:.3f}, last 10 avg reward: {last:.3f}")
    assert last > first, f"Learning should improve: {last:.3f} <= {first:.3f}"
