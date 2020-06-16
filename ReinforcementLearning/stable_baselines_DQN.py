import gym_jobshop
import gym, time
from statistics import mean

import stable_baselines
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('jobshop-v0')


def train_DQN():
    simulation_start_time = time.time()
    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("deepq_jobshop")
    print("Training finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    return


def predict_with_DQN():
    simulation_start_time = time.time()
    model = DQN.load("deepq_jobshop")

    scores = []  # list of final scores after each episode
    episodes = 30  # 30
    max_periods = 1000  # 8000

    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        next_state = env.reset()
        score = 0

        for period in range(max_periods):  # predict for x periods
            action, _states = model.predict(next_state)
            next_state, reward, done, info = env.step(action)
            score += reward
        scores.append(score)

        print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))

        # print("Observation space at the end: " + str(next_state))
    print("Prediction finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    return scores


def check_environment():
    from stable_baselines.common.env_checker import check_env
    stable_baselines.common.env_checker.check_env(env)


def evaluate_policy():
    model = DQN.load("deepq_jobshop")
    mean_reward, std_deviation = stable_baselines.common.evaluation.evaluate_policy(model, env, n_eval_episodes=10,
                                                                                    deterministic=True,
                                                                                    render=False, callback=None,
                                                                                    reward_threshold=None,
                                                                                    return_episode_rewards=False)
    print("Mean reward: " + str(mean_reward) + " | Standard deviation: " + str(std_deviation))
    return


if __name__ == "__main__":
    # check_environment()
    train_DQN()
    #evaluate_policy()
    #predict_with_DQN()
