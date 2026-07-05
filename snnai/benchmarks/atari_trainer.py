"""Training loop for the Catch SNN agent."""
from snnai.benchmarks.atari_agent import CatchSNNAgent
from snnai.benchmarks.atari_env import CatchEnv


def train_catch_agent(agent=None, env=None, n_episodes=200, max_steps=20):
    """Train a CatchSNNAgent on the Catch environment.

    Returns
    -------
    list[float]
        Total reward per episode.
    """
    if env is None:
        env = CatchEnv(grid_size=5)
    if agent is None:
        agent = CatchSNNAgent(input_size=3, hidden_size=16, n_actions=3, n_steps=10)

    rewards = []
    baseline = 0.0
    alpha = 0.1
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action, spikes_in, spikes_hid, spikes_out = agent.select_action(state)
            next_state, reward, done = env.step(action)
            baseline = alpha * reward + (1 - alpha) * baseline
            agent.update(spikes_in, spikes_hid, spikes_out, reward - baseline, action)
            total_reward += reward
            state = next_state
            if done:
                break
        rewards.append(total_reward)
    return rewards
