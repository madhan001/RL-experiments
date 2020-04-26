import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sb
import random
from collections import deque

tf.compat.v1.disable_eager_execution()

env = gym.make('MountainCar-v0')
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.97
LEARNING_RATE = 0.001


n_outputs = env.action_space.n

model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu',input_shape=env.observation_space.shape),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(n_outputs, activation='linear')
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')


def epsilon_greedy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(0, env.action_space.n)
    else:
        q_values = model.predict(state)
        action = np.argmax(q_values[0])
        return action


replay_buffer = deque(maxlen=2000000)


def play_one_step(env, state, epsilon):
    action = epsilon_greedy(state, epsilon)
    next_state, reward, done, _ = env.step(action)

    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done


def learn(batch_size):

    if len(replay_buffer) < batch_size:
        return

    samples = random.sample(replay_buffer, BATCH_SIZE)

    states = []
    new_states = []
    for sample in samples:
        state, action, reward, new_state, done = sample
        states.append(state)
        new_states.append(new_state)

    states = np.array(states).reshape(BATCH_SIZE, 2)

    new_states = np.array(new_states).reshape(BATCH_SIZE, 2)

    targets = model.predict(states)
    new_state_targets = model.predict(new_states)

    i = 0
    for sample in samples:
        state, action, reward, new_state, done = sample
        target = targets[i]
        if done:
            target[action] = reward
        else:
            next_q = max(new_state_targets[i])
            target[action] = reward + next_q * DISCOUNT_FACTOR
        i += 1

    model.fit(states, targets, epochs=1, verbose=0)


a_rewards = []
for episode in range(1000):
    state = env.reset().reshape(1, 2)
    goal = False
    t_reward = 0
    max_position = -99
    for step in range(200):
        epsilon = max(1 - episode / 400, 0.01)

        new_state, reward, done = play_one_step(env, state, epsilon)
        new_state = new_state.reshape(1,2)

        if new_state[0][0] >= 0.5:
            reward += 10

        if new_state[0][0] > max_position:
            max_position = new_state[0][0]

        t_reward += reward
        state = new_state
        learn(BATCH_SIZE)

        if new_state[0][0] >= 0.5:
            print("GOAL REACHED")
            goal = True
            env.render()
        if episode % 50 == 0:
            env.render()
        if done:
            #print("___DONE___")
            a_reward = t_reward / step
            a_rewards.append(t_reward)
            break

    print('episode = {} score = {} max_pos = {} goal_reached = {}'.format(episode, t_reward, max_position, goal))


sb.lineplot(data=a_rewards)
