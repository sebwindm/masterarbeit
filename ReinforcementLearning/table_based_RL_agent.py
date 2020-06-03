import gym
import gym_jobshop
import numpy as np
import random

def productionagent(verbose=False):
    # List of all scores
    scores = []

    # Creating the gym environment
    env = gym.make('jobshop-v0')

    # Initializing the Q-table of size state-space x action-space with zeros
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # Set the hyper-parameters
    epsilon = 1.0
    epsilon_min = 0.005
    epsilon_decay = 0.99993
    episodes = 1 # 50000
    max_periods = 1000 # 100
    learning_rate = 0.65
    gamma = 0.65

    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        state = env.reset()
        done = False
        score = 0

        for period in range(max_periods):
            # # With the probabilty of (1 - epsilon) take the best action in the Q-table
            # if random.uniform(0, 1) > epsilon:
            #     action = np.argmax(Q[state, :])
            # # Else take a random action
            # else:
            #     action = env.action_space.sample()

            # Step the game forward
            action = 0
            next_state, reward, done, _ = env.step(action)

            # Add up the score
            #score += reward
            score = reward

            # Update the Q-table with the Q-function
            # Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (
            #             reward + gamma * np.max(Q[next_state, :]))

            # Set the next state as the current state
            state = next_state

            if done:
                break

        # # Reducing the epsilon each episode (Exploration-Exploitation trade-off)
        # if epsilon >= epsilon_min:
        #     epsilon *= epsilon_decay

        scores.append(score)

        if verbose:
            print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))

    return scores


if __name__ == "__main__":
    productionagent(verbose=True)
