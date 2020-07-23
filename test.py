import gym
import tensorflow as tf
from util import get_action


def test(max_eps=10):
    env = gym.make('CartPole-v0')
    model = tf.saved_model.load('./best_model')
    env = gym.make('CartPole-v0')
    for eps in range(max_eps):
        done = False
        state = env.reset()
        eps_reward = 0.0
        while not done:
            env.render()
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            action_prob = model(state_tensor)
            action = get_action(action_prob)
            state, reward, done, _ = env.step(action)
            eps_reward += reward
        print('Finished episode {0} with score {1}'.format(eps, eps_reward))
    env.close()


if __name__ == '__main__':
    test()
