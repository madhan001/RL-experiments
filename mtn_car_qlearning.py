import gym
import numpy as np

# q_table = np.array()

episodes = 10000
learning_rate = 0.1
discount = 0.95
epsilon = 0.9
min_eps = 0
env = gym.make('MountainCar-v0')
env.reset()

# create discrete observation space
num_states = ((env.observation_space.high - env.observation_space.low) * np.array([10, 100])).round(0).astype(
    int)  # num_states is a 2-array

# initialize q table with random values
q_table = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], env.action_space.n))

state = env.reset()

#make states discrete
state_disc = ((state - env.observation_space.low) * np.array([10, 100])).round(0).astype(int)

reduction = (epsilon - min_eps) / episodes

for i in range(episodes):
    done = False

    while not done:

        env.render()

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[state_disc[0], state_disc[1]])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        print(action)

        state_new, reward, done, _ = env.step(action)
        print(reward)
        state_new_disc = ((state - env.observation_space.low) * np.array([10, 100])).round(0).astype(int)

        if not done:

            max_q_next = np.max(q_table[state_new_disc])
            q_current = q_table[state_disc, action]

            #0.5 is near the flag thing
            if done and state_new[0] >= 0.5:
                q_table[state_new[0], state_new[1], action] = reward

            #q-table update
            else:
                diff = learning_rate * (reward +
                                        discount * np.max(q_table[state_new_disc[0],
                                                                  state_new_disc[1]]) -
                                        q_table[state_new_disc[0], state_new_disc[1], action])

                q_table[state_new_disc[0], state_new_disc[1], action] += diff

        elif state_new[0] >= env.goal_position:
            # q_table[discrete_state + (action,)] = reward
            q_table[state_disc, action] = 0

        state_disc = state_new_disc

        if epsilon > min_eps:
            epsilon -= reduction

    env.close()