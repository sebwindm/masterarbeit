import gym, time
import gym_jobshop
import numpy as np
from statistics import mean


def productionagent(verbose=False):

    # Creating the gym environment
    env = gym.make('jobshop-v0')

    #print("Observation state sample mit Gym generiert:")
    #print("Order pool | Work center 1 | Work center 2 | Work center 3 | FGI | Shipped goods")
    #example = env.observation_space.sample()
    #print(str(example)  + " Dimensions: " + str(example.ndim))

    #print("Env state am Anfang: "  + str(env.reset()) + " Dimensions: " + str(env.state.ndim))

    #print("Dimensions vom flattened observation space: " + str(gym.spaces.flatdim(env.observation_space)))
    #print("Dimensions vom aktuellen observation space: " + str(np.array(env.observation_space).ndim))


    scores = [] # list of final scores after each episode
    episodes = 5 # 30
    max_periods = 1000 # 8000

    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        env.reset()
        score = 0

        for period in range(max_periods):
            action = 0 # keep default capacity of bottleneck machines
            next_state, reward, done, info = env.step(action)

            #if period == max_periods-1:
                #print("Observation state nach 1000 Perioden:")
                #print("Order pool | Work center 1 | Work center 2 | Work center 3 | FGI | Shipped goods")
                #print(str(next_state) + " Dimensions: " + str(env.state.ndim))

            # Add up the score
            score += reward

        scores.append(score)

        if verbose:
            print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))
    print("Agent finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))

    #print("Env state am Ende: "  + str(env.reset()) + " Dimensions: " + str(env.state.ndim))

    return scores


if __name__ == "__main__":
    simulation_start_time = time.time()
    scores = productionagent(verbose=True)

