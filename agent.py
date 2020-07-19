from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class ReinforceAgent:
    def __init__(self, state_space_size: int, action_space_size: int, gamma: float = 0.9) -> None:
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.policy_network = self._build_policy_network(
            state_space_size=self.state_space_size, action_space_size=self.action_space_size)

    def produce_action(self, state: np.array) -> List[int]:
        """
        Produces an action with the REINFORCE network where we
        feed the state into the network and choose the action
        (index) with the highest probability.
        """
        action_probabilities = self.policy_network(state).numpy()
        optimal_actions = np.argmax(action_probabilities, axis=1)
        return optimal_actions

    def train(self, rewards: List[float], states: List[List[float]], action_probabilities: List[List[float]], actions: List[int]) -> List[float]:
        """
        Trains the underlying network with the gameplay trajectory
        where the expected probabilities are the product of the log
        of current probabilities and the normalized discounted rewards.
        """
        discounted_rewards = self._compute_discounted_rewards(
            rewards, self.gamma)
        normalized_discounted_rewards = self._normalize_discounted_rewards(
            discounted_rewards)
        state_tensor = tf.convert_to_tensor(states)
        target_action_probabilities = np.array(action_probabilities)
        for i in range(len(states)):
            target_action_probabilities[actions[i]] = normalized_discounted_rewards[i]
        result = self.policy_network.fit(
            x=state_tensor, y=target_action_probabilities)
        return result.history['loss']

    @staticmethod
    def _build_policy_network(state_space_size, action_space_size) -> Sequential:
        """
        A helper function to build the policy network for the
        REINFORCE agent where the input is the state and the
        output is the probabilities of actions.
        """
        policy_network = Sequential()
        policy_network.add(
            Dense(units=64, input_dim=state_space_size, activation='relu', kernel_initializer='he_uniform'))
        policy_network.add(Dense(units=32, activation='relu',
                                 kernel_initializer='he_uniform'))
        policy_network.add(
            Dense(units=action_space_size, activation='softmax'))
        policy_network.compile(loss='categorical_crossentropy',
                               optimizer=tf.optimizers.Adam(learning_rate=0.01))
        return policy_network

    @staticmethod
    def _compute_discounted_rewards(rewards, gamma):
        discounted_reward = 0
        discounted_rewards = []
        for reward in rewards[::-1]:
            discounted_reward = gamma * discounted_reward + reward
            discounted_rewards.append(discounted_reward)
        return discounted_rewards[::-1]

    @staticmethod
    def _normalize_discounted_rewards(rewards: List[float]) -> List[int]:
        normalized_discounted_rewards = (
            rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)
        return normalized_discounted_rewards
