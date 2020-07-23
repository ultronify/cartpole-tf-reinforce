import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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


def get_action(probs):
    return np.argmax(probs)
