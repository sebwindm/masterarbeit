import gym, time
import gym_jobshop
import random
from statistics import mean

def random_agent(verbose=False):
    """
    This agent runs the environment with the default settings. His actions are random, he doesn't use states.
    Every new period, a new random action is taken (one out the possible actions 0,1 or 2).
    """
    # Create the gym environment
    env = gym.make('jobshop-v0')

    scores = []  # list of final scores after each episode
    episodes = 10  # 30

    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        env.reset()
        score = 0
        done = False
        while not done:
            action = random.randrange(0, 3)  # set action to a random number between 0 and 2
            next_state, reward, done, info = env.step(action,debug=True)
            # Add up the score
            score += reward

        scores.append(score)

        if verbose:
            print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))
    print("Agent finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))

    env.post_results()
    return scores


if __name__ == "__main__":
    simulation_start_time = time.time()
    scores = random_agent(verbose=True)
