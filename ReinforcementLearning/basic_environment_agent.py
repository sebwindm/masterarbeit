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
            #print(reward)
            scoredebugger.append(score)

        scores.append(score)

        if verbose:
            print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))
    print("Agent finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))

    env.post_results()
    return scores


if __name__ == "__main__":
    simulation_start_time = time.time()
    scores = productionagent(verbose=True)

