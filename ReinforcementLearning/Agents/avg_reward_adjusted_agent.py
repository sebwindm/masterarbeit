import gym, gym_jobshop, time
from ReinforcementLearning.average_reward_adjusted_algorithm import DQNAverageRewardAdjusted
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from statistics import mean


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        training_duration = 10000
        if self.model.num_timesteps % training_duration == 0:
            answer = input('Continue training for another '+ str(training_duration) + ' steps? y or n \n')
            if answer == "y":
                return True
            elif answer == "n":
                self.model.save("dqn_avg_reward_adjusted")
                print("Model saved")
                return False

        return True



def train_agent():
    """
    Instantiate the environment, train the agent and after training save the trained model to a file.
    """
    # Create environment
    env = gym.make('jobshop-v0')
    #env = gym.make('CartPole-v1')
    # Instantiate the agent with a modified DQN that is average reward adjusted
    # DQNAverageRewardAdjusted is based on stable_baselines3.dqn.DQN
    # MlpAverageRewardAdjustedPolicy is based on stable_baselines3.dqn.policies.DQNPolicy
    model = DQNAverageRewardAdjusted('MlpAverageRewardAdjustedPolicy', env, verbose=1, learning_starts=100, tensorboard_log="./gym_jobshop_tensorboard_logs/")
    # Train the agent
    start_time = time.time()
    custom_callback = CustomCallback()
    print("Training start")
    model.learn(total_timesteps=300000, callback=custom_callback)
    total_time = time.time() - start_time
    print(f"Took {total_time:.2f}s")
    # Save the agent
    model.save("dqn_avg_reward_adjusted")
    return


def evaluate_agent():
    """
    Evaluate the trained agent's performance using the evaluate_policy() function of Stable Baselines 3
    """
    # We create a separate environment for evaluation
    eval_env = gym.make('jobshop-v0')
    model = DQNAverageRewardAdjusted.load("dqn_avg_reward_adjusted")
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1)
    print(f'Custom DQN - Mean reward: {mean_reward} +/- {std_reward:.2f}')
    return


def predict_with_DQN():
    """
    Evaluate the trained agent's performance using a custom built function.
    It is not very different from evaluate_Agent(), except that it returns only the sum of rewards.
    """
    simulation_start_time = time.time()
    env = gym.make('jobshop-v0')
    #env = gym.make('CartPole-v1')
    model = DQNAverageRewardAdjusted.load("dqn_avg_reward_adjusted",env=env)

    scores = []  # list of final scores after each episode
    episodes = 1  # 30
    max_periods = 8000  # 8000

    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        next_state = env.reset()
        score = 0

        for period in range(max_periods):  # predict for x periods
            action, _states, _ = model.predict(next_state)
            #action, _states = model.predict(next_state)
            #print("action: ",action)
            next_state, reward, done, info = env.step(action)
            score += reward
        scores.append(score)

        print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))

        # print("Observation space at the end: " + str(next_state))
    print("Prediction finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    return scores


def delete_tensorboard_logs():
    import shutil
    shutil.rmtree('../gym_jobshop_tensorboard_logs')
    print("Deleted all Tensorboard logs")
    return


if __name__ == "__main__":
    answer = input('Type...to.. \n'
                   '"a" train the model (creates model file)\n'
                   '"b" delete Tensorboard logs\n'
                   '"c" predict 1 episode and print sum of rewards\n'
                   '"d" predict 1 episode and print mean + std of reward\n'
                   '"e" not implemented\n'
                   '"f" not implemented\n'
                   )
    if answer == "a":
        train_agent()
    if answer == "b":
        delete_tensorboard_logs()
    if answer == "c":
        predict_with_DQN()
    if answer == "d":
        evaluate_agent()
    if answer == "e":
        raise NotImplementedError
    if answer == "f":
        raise NotImplementedError
