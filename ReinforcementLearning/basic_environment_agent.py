import gym, time
import gym_jobshop
import numpy as np
from statistics import mean


def productionagent(verbose=False):
    """
    This agent runs the environment with the default settings. His actions are fixed, he doesn't use states.
    The results are identical to just running src/main.py.
    """
    # Creating the gym environment
    env = gym.make('jobshop-v0')

    # print("Observation state sample mit Gym generiert:")
    # print("Order pool | Work center 1 | Work center 2 | Work center 3 | FGI | Shipped goods")
    # example = env.observation_space.sample()
    # print(str(example)  + " Dimensions: " + str(example.ndim))

    # print("Env state am Anfang: "  + str(env.reset()) + " Dimensions: " + str(env.state.ndim))

    # print("Dimensions vom flattened observation space: " + str(gym.spaces.flatdim(env.observation_space)))
    # print("Dimensions vom aktuellen observation space: " + str(np.array(env.observation_space).ndim))

    scores = []  # list of final scores after each episode
    episodes = 1  # 30

    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        env.reset()
        score = 0
        done = False
        scoredebugger = []
        while not done:
            action = 0  # keep default capacity of bottleneck machines
            next_state, reward, done, info = env.step(action,debug=True)
            # Add up the score
            score += reward
            scoredebugger.append(score)
        scores.append(score)

        if verbose:
            print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))
    print("Agent finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    print(info)

    return scores


if __name__ == "__main__":
    simulation_start_time = time.time()
    scores = productionagent(verbose=True)
