import gym, time
import gym_jobshop
from statistics import mean

def default_agent(verbose=False):
    """
    This agent runs the environment with the default settings. His actions are fixed, he doesn't use states.
    The agent's action is always action 0, so he keeps the capacity of the bottleneck machine(s)
    always at the same default level of 1.0
    The results are identical to just running src/main.py.
    """
    # Creating the gym environment
    env = gym.make('jobshop-v0')

    scores = []  # list of final scores after each episode
    episodes = 10  # 30

    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        env.reset()
        score = 0
        done = False
        while not done:
            action = 0  # keep default capacity of bottleneck machines
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
    scores = default_agent(verbose=True)

