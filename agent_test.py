from agent import ReinforceAgent


import numpy as np
from agent import ReinforceAgent


def test_agent_initialization():
    agent = ReinforceAgent(state_space_size=20, action_space_size=5)
    assert agent is not None


def test_agent_can_output_action():
    max_steps = 5
    state_space_size = 20
    action_space_size = 5
    agent = ReinforceAgent(state_space_size=state_space_size,
                           action_space_size=action_space_size)
    actions = agent.produce_action(
        np.zeros(shape=(max_steps, state_space_size)))
    assert actions.shape == (max_steps,)
    for i in range(max_steps):
        assert actions[i] >= 0 and actions[i] < action_space_size


def test_compute_discounted_rewards_sanity_check():
    rewards = [1, 1, 1, 1]
    gamma = 1
    discounted_rewards = ReinforceAgent._compute_discounted_rewards(
        rewards, gamma)
    assert all([a == b for a, b in zip(discounted_rewards, [4, 3, 2, 1])])


def test_compute_discounted_rewards_correctness():
    rewards = [1, 1, 1]
    gamma = 0.5
    discounted_rewards = ReinforceAgent._compute_discounted_rewards(
        rewards, gamma)
    assert all([a == b for a, b in zip(
        discounted_rewards, [1 + 0.5 + 0.5**2, 1 + 0.5, 1])])


def test_normalize_discounted_rewards_sanity_check():
    discounted_rewards = [5, 4, 3, 2, 1]
    normalized_discounted_rewards = ReinforceAgent._normalize_discounted_rewards(
        discounted_rewards)
    for normalized_discounted_reward in normalized_discounted_rewards:
        assert normalized_discounted_reward < 5


def test_train_network_sanity_check():
    state_space_size = 2
    action_space_size = 2
    agent = ReinforceAgent(state_space_size=state_space_size,
                           action_space_size=action_space_size)
    agent.train(rewards=[1, 1, 1], states=[[1, 1], [1, 1], [
                1, 1]], action_probabilities=[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], actions=[2, 2, 2])
