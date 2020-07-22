from typing import List
import numpy as np
import tensorflow as tf
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.backend import dtype


def eval(model, env, max_eps, action_space_size):
  total_reward = 0.0
  for _ in range(max_eps):
    done = False
    state = env.reset()
    while not done:
      action_prob = model(tf.convert_to_tensor([state]))
      action = sample_action(action_prob, action_space_size)
      state, reward, done, _ = env.step(action)
      total_reward += reward
  avg_reward = total_reward / max_eps
  return avg_reward


def build_model(state_space_size, action_space_size):
    policy_network = Sequential()
    policy_network.add(
        Dense(units=64, input_dim=state_space_size, activation='relu', kernel_initializer='he_uniform'))
    policy_network.add(Dense(units=32, activation='relu',
                             kernel_initializer='he_uniform'))
    policy_network.add(
        Dense(units=action_space_size, activation='softmax'))
    return policy_network


def sample_action(probs, action_space_size):
    prob = np.array(probs[0])
    prob /= prob.sum()
    return np.random.choice(action_space_size, p=prob)


def compute_discounted_rewards(rewards, gamma):
    discounted_reward = 0
    discounted_rewards = []
    for reward in rewards[::-1]:
        discounted_reward = gamma * discounted_reward + reward
        discounted_rewards.append(discounted_reward)
    return discounted_rewards[::-1]


def normalize_discounted_rewards(discounted_rewards):
    normalized_discounted_rewards = (
        discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)
    return normalized_discounted_rewards


def train(max_eps=6000):
    env = gym.make('CartPole-v0')
    eval_env = gym.make('CartPole-v0')
    state_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    print('action space size is {0}, state space size is {1}'.format(
        action_space_size, state_space_size))
    model = build_model(state_space_size, action_space_size)
    model.summary()
    gamma = 0.9
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    for eps in range(max_eps):
        done = False
        state = env.reset()
        rewards = []
        action_probs = []
        actions = []
        with tf.GradientTape() as tape:
            while not done:
                action_prob = model(tf.convert_to_tensor([state]))
                action = sample_action(action_prob, action_space_size)
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                action_probs.append(action_prob[0])
                actions.append(action)
                state = next_state
            discounted_rewards = compute_discounted_rewards(rewards, gamma)
            normalized_discounted_rewards = tf.convert_to_tensor(
                normalize_discounted_rewards(discounted_rewards), dtype=tf.float32)
            probs = tf.stack(action_probs)
            clipped_probs = tf.clip_by_value(probs, 1e-8, 1.0 - 1e-8)
            onehot_actions = tf.one_hot(
                actions, action_space_size, dtype=tf.float32)
            log_likelihood = tf.multiply(
                onehot_actions, tf.math.log(clipped_probs))
            loss = tf.math.reduce_sum(
                tf.multiply(-log_likelihood, tf.expand_dims(normalized_discounted_rewards, axis=1)))
        policy_gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(
            zip(policy_gradients, model.trainable_weights))
        score = eval(model, eval_env, 10, action_space_size)
        print('Done {0}/{1} with score {2}'.format(eps, max_eps, score))


train()
