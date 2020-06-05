import gym
import gym_jobshop
import numpy as np
import random

def productionagent(verbose=False):
    # List of all scores
    scores = []

    # Creating the gym environment
    env = gym.make('jobshop-v0')

    print("Observation state sample mit Gym generiert:")
    #print("Order pool | Work center 1 | Work center 2 | Work center 3 | FGI | Shipped goods")
    example = env.observation_space.sample()
    print(str(example)  + " Dimensions: " + str(example.ndim))

    print("Env state am Anfang: "  + str(env.reset()) + " Dimensions: " + str(env.state.ndim))

    print("Dimensions vom flattened observation space: " + str(gym.spaces.flatdim(env.observation_space)))
    print("Dimensions vom aktuellen observation space: " + str(np.array(env.observation_space).ndim))

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

            #     action = env.action_space.sample()

            # Step the game forward
            action = 0
            next_state, reward, done, info = env.step(action)

            if period == max_periods-1:
                print("Observation state nach 1000 Perioden:")
                #print("Order pool | Work center 1 | Work center 2 | Work center 3 | FGI | Shipped goods")
                print(str(next_state) + " Dimensions: " + str(env.state.ndim))

            # Add up the score
            score += reward

            # Set the next state as the current state
            state = next_state

            if done:
                break
        # Reducing the epsilon each episode (Exploration-Exploitation trade-off)
        if epsilon >= epsilon_min:
            epsilon *= epsilon_decay

        scores.append(score)

        if verbose:
            print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))

    print("Env state am Ende: "  + str(env.reset()) + " Dimensions: " + str(env.state.ndim))
    return scores


if __name__ == "__main__":
    productionagent(verbose=True)
