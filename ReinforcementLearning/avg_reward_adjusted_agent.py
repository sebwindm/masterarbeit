import gym, gym_jobshop, time, random
from Algorithm.average_reward_adjusted_algorithm import DQNAverageRewardAdjusted # ignore the error message in PyCharm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from statistics import mean



# Create environment
env = gym.make('jobshop-v0')
# env = gym.make('CartPole-v1')
number_of_evaluation_episodes = 5 # default = 30

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.default_training_steps = 100000
        self.train_for_x_steps = self.default_training_steps

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        Take user input to decide whether to train for 1 or many steps or to quit training and save the model.
        """
        if self.model.num_timesteps % self.train_for_x_steps == 0:
            user_input = input('\nTimestep ' + str(self.model.num_timesteps) + ' done: press...and then Enter...\n'
                '"y"... to train to the next increment of ' + str(
                self.default_training_steps) + ' steps\n'
                '"x" ... to enter an amount of steps to train\n'
                '"Enter"... to train for 1 step \n'
                '"s"... to print statistics and train for 1 step \n'
                '"n"... to stop training and save the model\n')
            if user_input == "y":  # train up to the next increment of the default amount of steps
                # note that this trains not for the amount of self.default_training_steps,
                # but instead it trains up to the next multiple of self.default_training_steps
                # Example: let's assume self.default_training_steps = 1000
                # if you are at step 1 it will train up to step 1000,
                # if you are at step 999 it will also train up to step 1000
                self.train_for_x_steps = self.default_training_steps
                return True
            elif user_input == "":  # if Enter key gets pressed, train for one step
                self.train_for_x_steps = 1
                return True
            elif user_input == "s" and self.model.num_timesteps > 0:  # Print observation state and some statistics,
                # afterwards train for one step
                observation = env.get_observation().tolist()
                separator_indices = [0,10,13,17,
                                     22,32,35,39,
                                     44,54,57,61,
                                     66,76,79,83,
                                     88,98,101,105,
                                     110,120,123,127
                                     ]
                separator_names = ["OP1: ", "WC1:","FGI1: ","SG1: ",
                                   "OP2: ", "WC2:","FGI2: ","SG2: ",
                                   "OP3: ", "WC3:","FGI3: ","SG3: ",
                                   "OP4: ", "WC4:","FGI4: ","SG4: ",
                                   "OP5: ", "WC5:","FGI5: ","SG5: ",
                                   "OP6: ", "WC6:","FGI6: ","SG6: "
                                   ]
                for i in separator_indices:
                    observation.insert(i + separator_indices.index(i) ,separator_names[separator_indices.index(i)])
                print("Observation state:", observation)
                print("Unnormalized reward:", self.model.current_unnormalized_reward)
                print("Costs per production step: ",env.get_cost_rundown())
                print("Rho: ", self.model.rho[0])
                q1, q2, q3, action = self.model.get_q_values_for_current_observation()
                print("Predicted action: ",action, " | Q values for current observation: \n",
                      q1, q2, q3)
                print("-----------------------------------------------------------------------")
                self.train_for_x_steps = 1
            elif user_input == "s" and self.model.num_timesteps <= 0:
                raise ValueError("You need to train for at least one step before accessing statistics")
            elif user_input == "x":
                steps = input('Train for how many steps?\n')
                try:
                    steps = int(steps)
                except ValueError: "Please enter a an integer number"
                self.train_for_x_steps = steps
                return True


            elif user_input == "n":  # quit training and save model file
                self.model.save("dqn_avg_reward_adjusted")
                print("Model saved")
                return False

            else:  # if any other key gets pressed, train for one step
                self.train_for_x_steps = 1
                return True


def train_agent():
    """
    Train the agent and after training save the trained model to a file.
    """
    env.reset()
    # Instantiate the agent with a modified DQN that is average reward adjusted
    # DQNAverageRewardAdjusted is based on stable_baselines3.dqn.DQN
    # MlpAverageRewardAdjustedPolicy is based on stable_baselines3.dqn.policies.DQNPolicy
    model = DQNAverageRewardAdjusted('MlpAverageRewardAdjustedPolicy', env, verbose=1, learning_starts=100) #,tensorboard_log="./gym_jobshop_tensorboard_logs/"
    # Train the agent
    start_time = time.time()
    custom_callback = CustomCallback()
    print("Training start")
    # Train the model. Default total_timesteps for 1 machine = 300000
    #model.learn(total_timesteps=10000000) # learn with no callback
    model.learn(total_timesteps=500000, callback=custom_callback)  # learn with custom callback
    total_time = time.time() - start_time
    print(f"Took {total_time:.2f}s")
    # Save the agent
    model.save("dqn_avg_reward_adjusted_3_machines")
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
    env.reset()
    simulation_start_time = time.time()
    model = DQNAverageRewardAdjusted.load("dqn_avg_reward_adjusted_3_machines", env=env)

    scores = []  # list of final scores after each episode
    episodes = number_of_evaluation_episodes
    max_periods = 8000  # 8000

    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        next_state = env.reset()
        score = 0

        for period in range(max_periods):  # predict for x periods
            action, _states, _ = model.predict(next_state)
            # action, _states = model.predict(next_state)
            # print("action: ",action)
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
    shutil.rmtree('./gym_jobshop_tensorboard_logs')
    print("Deleted all Tensorboard logs")
    return


def evaluate_with_default_action():
    env.reset()
    simulation_start_time = time.time()
    model = DQNAverageRewardAdjusted.load("dqn_avg_reward_adjusted_3_machines", env=env)

    scores = []  # list of final scores after each episode
    episodes = number_of_evaluation_episodes
    max_periods = 8000  # 8000
    action = 0
    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        next_state = env.reset()
        score = 0
        for period in range(max_periods):  # predict for x periods
            next_state, reward, done, info = env.step(action)
            score += reward
        scores.append(score)

        print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))

    print("Evaluation finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    return scores


def evaluate_with_random_action():
    env.reset()
    simulation_start_time = time.time()
    model = DQNAverageRewardAdjusted.load("dqn_avg_reward_adjusted_3_machines", env=env)

    scores = []  # list of final scores after each episode
    episodes = number_of_evaluation_episodes
    max_periods = 8000  # 8000
    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        next_state = env.reset()
        score = 0
        for period in range(max_periods):  # predict for x periods
            action = random.randrange(0, 3)  # set action to a random number between 0 and 2
            next_state, reward, done, info = env.step(action)
            score += reward
        scores.append(score)

        print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))

    print("Evaluation finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    return scores

if __name__ == "__main__":
    answer = input('Type...to.. \n'
                   '"a" or "Enter" train the model (creates model file)\n'
                   '"b" delete Tensorboard logs\n'
                   '"c" evaluate with trained agent\n'
                   '"d" evaluate with default action only\n'
                   '"e" evaluate with random action \n'
                   )
    if answer == "a"or answer == "":
        train_agent()
    if answer == "b":
        delete_tensorboard_logs()
    if answer == "c":
        predict_with_DQN()
    if answer == "d":
        evaluate_with_default_action()
    if answer == "e":
        evaluate_with_random_action()

