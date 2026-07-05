"""Simple training loop for the honeybee grid-world agent."""
from .gridworld import GridWorld
from .gridworld_agent import SNNAgent


def train_agent(
    agent=None,
    env=None,
    n_episodes=40,
    max_steps=30,
):
    """Train an SNNAgent on the GridWorld.

    Returns
    -------
    list[float]
        Total reward per episode.
    """
    if env is None:
        env = GridWorld(size=5, start=(0, 0), goal=(4, 4), obstacles=[(2, 2)], shaped=True)
    if agent is None:
        agent = SNNAgent(size=env.size, n_cells=25, hidden_size=20)

    rewards = []
    baseline = 0.0
    alpha = 0.1
    for _ in range(n_episodes):
        pos = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action, spikes_in, spikes_hid, spikes_out = agent.select_action(pos)
            next_pos, reward, done = env.step(action)
            baseline = alpha * reward + (1 - alpha) * baseline
            agent.update(spikes_in, spikes_hid, spikes_out, reward - baseline, action)
            total_reward += reward
            pos = next_pos
            if done:
                break
        rewards.append(total_reward)
    return rewards
