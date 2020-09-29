"""
How to: 
Run this script to train or evaluate an agent. 

train_DQN() will train a model using the average reward adjusted algorithm.
This model can then be evaluated with evaluate_with_DQN() and compared to
an agent that runs only random actions [evaluate_with_random_action()]
or to an agent that runs always the default action 0 [evaluate_with_default_action()]

Some linters might find errors in this file, feel free to ignore them.
"""
import gym, gym_jobshop, time, random
from Algorithm.average_reward_adjusted_algorithm import DQNAverageRewardAdjusted  # ignore the error message in PyCharm
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from statistics import mean

number_of_evaluation_episodes = 5  # default = 30
number_of_training_timesteps = 1000000  # default for 1 machine: 300000, for 3 machines 1000000

# Create environment
env = gym.make('jobshop-v0')


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from `BaseCallback`.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    Todo: documentation
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
                separator_indices = [0, 10, 13, 17,
                                     22, 32, 35, 39,
                                     44, 54, 57, 61,
                                     66, 76, 79, 83,
                                     88, 98, 101, 105,
                                     110, 120, 123, 127
                                     ]
                separator_names = ["OP1: ", "WC1:", "FGI1: ", "SG1: ",
                                   "OP2: ", "WC2:", "FGI2: ", "SG2: ",
                                   "OP3: ", "WC3:", "FGI3: ", "SG3: ",
                                   "OP4: ", "WC4:", "FGI4: ", "SG4: ",
                                   "OP5: ", "WC5:", "FGI5: ", "SG5: ",
                                   "OP6: ", "WC6:", "FGI6: ", "SG6: "
                                   ]
                for i in separator_indices:
                    observation.insert(i + separator_indices.index(i), separator_names[separator_indices.index(i)])
                print("Observation state:", observation)
                print("Unnormalized reward:", self.model.current_unnormalized_reward,
                      "| Last used action:", self.model.last_used_action)
                print("Costs per production step: ", env.get_cost_rundown())
                print("Rho: ", self.model.rho[0])
                q1, q2, q3, action = self.model.get_q_values_for_current_observation()
                print("Predicted action: ", action, " | Q values for current observation: \n",
                      q1, q2, q3)
                print("-----------------------------------------------------------------------")
                self.train_for_x_steps = 1
            elif user_input == "s" and self.model.num_timesteps <= 0:
                raise ValueError("You need to train for at least one step before accessing statistics")
            elif user_input == "x":
                steps = input('Train for how many steps?\n')
                try:
                    steps = int(steps)
                except ValueError:
                    "Please enter a an integer number"
                self.train_for_x_steps = steps
                return True

            elif user_input == "n":  # quit training and save model file
                self.model.save("dqn_avg_reward_adjusted")
                print("Model saved")
                return False

            else:  # if any other key gets pressed, train for one step
                self.train_for_x_steps = 1
                return True


def train_DQN():
    """
    Train the agent and after training save the trained model to a file.
    Important settings:
    * number_of_machines: must be either 1 or 3
    * number_of_training_timesteps: set the amount of periods the agent(s) should train for
    Todo: further documentation
    """
    env.reset()
    # Instantiate the agent with a modified DQN that is average reward adjusted
    # DQNAverageRewardAdjusted is based on stable_baselines3.dqn.DQN
    # MlpAverageRewardAdjustedPolicy is based on stable_baselines3.dqn.policies.DQNPolicy
    #
    # Setup model depending on the amount of machines
    if env.number_of_machines == 1:
        model = DQNAverageRewardAdjusted('MlpAverageRewardAdjustedPolicy', env,
                                         verbose=1,
                                         # tensorboard_log="./gym_jobshop_tensorboard_logs/",
                                         learning_starts=100,
                                         buffer_size=1000000, # 1 machine: 1000000, 3 machines 25000
                                         tau=1.0, # 1 machine: 1.0, 3 machines: 1e-5
                                         gamma=0.99, # 1 machine: 0.99, 3 machines: 1.0
                                         exploration_fraction=0.15,  # 1 machine: 0.15, 3 machines = 0.4
                                         exploration_final_eps=0.05, # 1 machine: 0.05, 3 machines: 0.01
                                         alpha=0.01,
                                         alpha_min=1e-5,  # 1 machine: 1e-5, 3 machines: 1e-6
                                         alpha_decay_rate=0.55, # 1 machine: 0.55, 3 machines: 0.15
                                         alpha_decay_steps=15000 # 1 machine 15000, 3 machines 50000
                                         # decay steps 15000 for 100k learning steps
                                         )
    elif env.number_of_machines == 3:
        model = DQNAverageRewardAdjusted('MlpAverageRewardAdjustedPolicy', env,
                                         verbose=1,
                                         # tensorboard_log="./gym_jobshop_tensorboard_logs/",
                                         learning_starts=100,
                                         buffer_size=25000, # 1 machine: 1000000, 3 machines 25000
                                         tau=1e-5, # 1 machine: 1.0, 3 machines: 1e-5
                                         gamma=1.0, # 1 machine: 0.99, 3 machines: 1.0
                                         exploration_fraction=0.4,  # 1 machine: 0.1 - 0.15, 3 machines = 0.4
                                         exploration_final_eps=0.01, # 1 machine: 0.05, 3 machines: 0.01
                                         alpha=0.01,
                                         alpha_min=1e-6,  # 1 machine: 1e-5, 3 machines: 1e-6
                                         alpha_decay_rate=0.15, # 1 machine: 0.55, 3 machines: 0.15
                                         alpha_decay_steps=50000 # 1 machine 15000, 3 machines 50000
                                         # decay steps 15000 for 100k learning steps
                                         )
    # Train the agent
    start_time = time.time()
    custom_callback = CustomCallback()
    print("Training with", env.number_of_machines, "machines")
    # Train the model. Default total_timesteps for 1 machine = 300000
    # model.learn(total_timesteps=10000000) # learn with no callback
    model.learn(total_timesteps=number_of_training_timesteps, callback=custom_callback)  # learn with custom callback
    total_time = time.time() - start_time
    print(f"Took {total_time:.2f}s")
    # Save the agent
    print("Saving model")
    model.save("dqn_avg_reward_adjusted_" + str(number_of_machines) + "_machine")
    return


def evaluate_with_DQN():
    """
    Not documented yet
    """
    print("Evaluating with DQN prediction")
    env.reset()
    simulation_start_time = time.time()
    model = DQNAverageRewardAdjusted.load("dqn_avg_reward_adjusted_" + str(number_of_machines) + "_machine", env=env)

    scores = []  # list of final scores after each episode
    episodes = number_of_evaluation_episodes
    max_periods = 8000  # 8000

    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        next_state = env.reset()
        score = 0

        for period in range(max_periods):  # predict for x periods
            action, _states, _ = model.predict(next_state)
            next_state, reward, done, info = env.step(action)
            score += reward
        scores.append(score)

        print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))

    print("Prediction finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    return scores


def delete_tensorboard_logs():
    """
      # not documented yet
    """
    import shutil
    shutil.rmtree('./gym_jobshop_tensorboard_logs')
    print("Deleted all Tensorboard logs")
    return


def evaluate_with_default_action():
    """
      # not documented yet
    """
    action = 0
    print("Evaluating with default action", action)
    env.reset()
    simulation_start_time = time.time()
    scores = []  # list of final scores after each episode
    episodes = number_of_evaluation_episodes
    max_periods = 8000  # 8000
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
    """
      # not documented yet
    """
    print("Evaluating with random action")
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


def train_a2c():
    """
      # not documented yet
    """
    from stable_baselines3 import A2C
    from stable_baselines3.a2c import MlpPolicy
    print("Training with A2C algorithm")
    model = A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=number_of_training_timesteps)
    model.save("a2c")
    return


def train_ppo():
    """
      # not documented yet
    """
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    print("Training with PPO algorithm")
    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=number_of_training_timesteps)
    model.save("ppo")
    return


def evaluate_other_algos(algorithm):
    """
      # not documented yet
    """
    if algorithm == "a2c":
        from stable_baselines3 import A2C
        model = A2C.load("a2c")
    if algorithm == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load("ppo")
    scores = []  # list of final scores after each episode
    episodes = number_of_evaluation_episodes
    print("Evaluating with", algorithm)
    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        next_state = env.reset()
        score = 0

        for period in range(8000):  # predict for x periods
            action, _states = model.predict(next_state)
            # action, _states = model.predict(next_state)
            # print("action: ",action)
            next_state, reward, done, info = env.step(action)
            score += reward
        scores.append(score)

        print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    return scores


if __name__ == "__main__":
    answer = input('Type...to... \n'
                   '"a" or "Enter" ... train the model (creates model file)\n'
                   '"b" ... delete Tensorboard logs\n'
                   '"c" ... evaluate with trained agent\n'
                   '"d" ... evaluate with default action only\n'
                   '"e" ... evaluate with random action \n'
                   )
    if answer == "a" or answer == "":
        train_DQN()
    if answer == "b":
        delete_tensorboard_logs()
    if answer == "c":
        evaluate_with_DQN()
    if answer == "d":
        evaluate_with_default_action()
    if answer == "e":
        evaluate_with_random_action()
    if answer == "ppo":  # not documented yet
        train_ppo()
    if answer == "a2c":  # not documented yet
        train_a2c()
    if answer == "ppo_eval":  # not documented yet
        evaluate_other_algos("ppo")
    if answer == "a2c_eval":  # not documented yet
        evaluate_other_algos("a2c")
