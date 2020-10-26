"""
How to: 
Run this script to train or evaluate an agent. 

train_DQN() will train a model using the average reward adjusted algorithm.
This model can then be evaluated with evaluate_with_DQN() and compared to
an agent that runs only random actions [evaluate_with_random_action()]
or to an agent that runs always the default action 0 [evaluate_with_default_action()]

Some linters might find errors in this file, feel free to ignore them.

A note on the csv exports from the job shop environment:
The total cost of metrics per episode and the reward per period are different due to the cost reset
after the warmup period.
Total cost of metrics per episode takes into account the cost reset, rewards per episode does not.
"""
import gym, gym_jobshop, time, random, os
from stable_baselines3.common.callbacks import BaseCallback
from statistics import mean

# Change the values below to adjust some training and evaluation parameters
number_of_machines = 3
number_of_evaluation_episodes = 30  # default = 30
number_of_training_timesteps = 1000000  # default for 1 machine: 300000, for 3 machines 1000000
default_action = 0  # The action that the default agent always uses
csv_metrics_per_episode = True
csv_rewards_per_period = False


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


def initialize_environment(number_of_machines=3, csv_metrics_per_episode=False, csv_rewards_per_period=False, global_prefix=""):
    # Create environment
    env = gym.make('jobshop-v0', csv_metrics_per_episode=csv_metrics_per_episode, csv_rewards_per_period=csv_rewards_per_period,
                   number_of_machines=number_of_machines, global_prefix=global_prefix)
    return env


def train_ARA_DiRL(number_of_machines):
    """
    Train the algorithm and after training save the trained model to a file.
    Important settings:
    * number_of_machines: must be either 1 or 3
    * number_of_training_timesteps: set the amount of periods the agent(s) should train for
    Documentation at Reinforcement Learning/Algorithm/ARA_DiRL
    """
    from Algorithm.ARA_DiRL import DQNAverageRewardAdjusted  # ignore the error message in PyCharm
    env = initialize_environment(number_of_machines=number_of_machines, csv_metrics_per_episode=False,
                                 csv_rewards_per_period=False, global_prefix="ARA_DIRL_train")
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
                                         buffer_size=1000000,  # default 1000000
                                         tau=1.0,  # default 1.0
                                         gamma=0.99,  # default 0.99
                                         exploration_fraction=0.15,  # default  0.15
                                         exploration_final_eps=0.05,  # default  0.05
                                         alpha=0.01,  # default  0.01
                                         alpha_min=1e-5,  # default  1e-5
                                         alpha_decay_rate=0.55,  # default  0.55
                                         alpha_decay_steps=15000,  # default 15000
                                         # decay steps 15000 for 100k learning steps
                                         # decay steps must increase with training duration
                                         seed=1,
                                         )
    elif env.number_of_machines == 3:
        model = DQNAverageRewardAdjusted('MlpAverageRewardAdjustedPolicy', env,
                                         verbose=1,
                                         # tensorboard_log="./gym_jobshop_tensorboard_logs/",
                                         learning_starts=100,
                                         buffer_size=100000,  # default 100000
                                         tau=1.0,  # default 1.0
                                         gamma=0.99,  # default 0.990
                                         exploration_fraction=0.4,  # default = 0.4
                                         exploration_final_eps=0.01,  # default 0.01
                                         alpha=0.01,  # default 0.01
                                         alpha_min=1e-6,  # default 1e-6
                                         alpha_decay_rate=0.55,  # default 0.55
                                         alpha_decay_steps=50000,  # default 50000
                                         # decay steps 15000 for 100k learning steps
                                         # decay steps must increase with training duration
                                         seed=1,
                                         )
    else:
        raise ValueError("wrong number of machines")
    # Train the agent
    start_time = time.time()
    custom_callback = CustomCallback()
    print("Training ARA-DiRL with", env.number_of_machines, "machines")
    # Train the model. Default total_timesteps for 1 machine = 300000
    # model.learn(total_timesteps=10000000) # learn with no callback
    model.learn(total_timesteps=number_of_training_timesteps, callback=custom_callback)  # learn with custom callback
    total_time = time.time() - start_time
    print(f"Took {total_time:.2f}s")
    # Save the agent
    print("Saving ARA-DiRL model")
    model.save("dqn_avg_reward_adjusted_" + str(env.number_of_machines) + "_machine")
    return


def evaluate_with_ARA_DiRL(number_of_machines, csv_metrics_per_episode, csv_rewards_per_period):
    """
    Evaluate using ARA-DiRL. Documentation see Reinforcement Learning/Algorithm/ARA_DiRL
    """
    from Algorithm.ARA_DiRL import DQNAverageRewardAdjusted  # ignore the error message in PyCharm
    print("Evaluating with ARA-DiRL prediction")
    env = initialize_environment(number_of_machines=number_of_machines, csv_metrics_per_episode=csv_metrics_per_episode,
                                 csv_rewards_per_period=csv_rewards_per_period, global_prefix="ARA_DIRL_eval")
    simulation_start_time = time.time()
    model = DQNAverageRewardAdjusted.load("dqn_avg_reward_adjusted_" + str(env.number_of_machines) + "_machine",
                                          env=env)

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

    print("ARA-DiRL evaluation finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    return scores


def evaluate_with_default_action(action, number_of_machines, csv_metrics_per_episode, csv_rewards_per_period):
    """
    This is an agent that uses always the same action in each period.
    The function takes the desired default action as an argument.
    Change default_action on top of this file for easy switching.
    """
    print("Evaluating with default action", action)
    env = initialize_environment(number_of_machines=number_of_machines, csv_metrics_per_episode=csv_metrics_per_episode,
                                 csv_rewards_per_period=csv_rewards_per_period, global_prefix="default"+str(action))
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

    print("Default action evaluation finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    return scores


def evaluate_with_random_action(number_of_machines, csv_metrics_per_episode, csv_rewards_per_period):
    """
      # not documented yet
    """
    print("Evaluating with random action")
    env = initialize_environment(number_of_machines=number_of_machines, csv_metrics_per_episode=csv_metrics_per_episode,
                                 csv_rewards_per_period=csv_rewards_per_period, global_prefix="random")
    simulation_start_time = time.time()

    scores = []  # list of final scores after each episode
    episodes = number_of_evaluation_episodes
    max_periods = 8000  # 8000
    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        next_state = env.reset()
        random.seed(env.random_seed)
        score = 0
        for period in range(max_periods):  # predict for x periods
            action = random.randrange(0, 3)  # set action to a random number between 0 and 2
            next_state, reward, done, info = env.step(action)
            score += reward
        scores.append(score)

        print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))

    print("Random action evaluation finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    return scores


def train_a2c(number_of_machines):
    """
    See https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html for documentation
    """
    from stable_baselines3 import A2C
    from stable_baselines3.a2c import MlpPolicy
    print("Training with A2C algorithm")
    env = initialize_environment(number_of_machines=number_of_machines, csv_metrics_per_episode=False,
                                 csv_rewards_per_period=False, global_prefix="a2c_train")
    model = A2C(MlpPolicy, env, verbose=1, seed=1)
    model.learn(total_timesteps=number_of_training_timesteps)
    model.save("a2c_" + str(number_of_machines))
    print("Finished A2C training")
    return


def train_ppo(number_of_machines):
    """
    See https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html for documentation
    """
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    print("Training with PPO algorithm")
    env = initialize_environment(number_of_machines=number_of_machines, csv_metrics_per_episode=False,
                                 csv_rewards_per_period=False, global_prefix="ppo_train")
    model = PPO(MlpPolicy, env, verbose=1, seed=1)
    model.learn(total_timesteps=number_of_training_timesteps)
    model.save("ppo_" + str(number_of_machines))
    print("Finished PPO training")
    return


def train_vanilla_dqn(number_of_machines):
    """
    Vanilla DQN refers to the basic implementation of a DQN by Stable Baselines 3
    It is not to be confused with ARA-DiRL that is an extended version of this vanilla_dqn
    See https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html for documentation
    """
    from stable_baselines3 import DQN
    from stable_baselines3.dqn import MlpPolicy
    print("Training with vanilla_dqn algorithm")
    env = initialize_environment(number_of_machines=number_of_machines, csv_metrics_per_episode=False,
                                 csv_rewards_per_period=False, global_prefix="vanilla_dqn_train")
    model = DQN(MlpPolicy, env, verbose=1, seed=1)
    model.learn(total_timesteps=number_of_training_timesteps)
    model.save("vanilla_dqn_" + str(number_of_machines))
    print("Finished vanilla_dqn training")
    return


def evaluate_other_algos(algorithm, number_of_machines, csv_metrics_per_episode, csv_rewards_per_period):
    """
      # not documented yet
    """
    env = initialize_environment(number_of_machines=number_of_machines, csv_metrics_per_episode=csv_metrics_per_episode,
                                 csv_rewards_per_period=csv_rewards_per_period, global_prefix=str(algorithm) + "_eval")
    if algorithm == "a2c":
        from stable_baselines3 import A2C
        model = A2C.load("a2c_" + str(number_of_machines))
    if algorithm == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load("ppo_" + str(number_of_machines))
    if algorithm == "vanilla_dqn":
        from stable_baselines3 import DQN
        model = DQN.load("vanilla_dqn_" + str(number_of_machines))
    else:
        raise ValueError("wrong algorithm name")
    scores = []  # list of final scores after each episode
    episodes = number_of_evaluation_episodes
    print("Evaluating with", algorithm)
    for episode in range(episodes):
        # Reset the game-state, done and score before every episode
        next_state = env.reset()
        random.seed(env.random_seed)
        score = 0

        for period in range(8000):  # predict for x periods
            action, _states = model.predict(next_state)
            # action, _states = model.predict(next_state)
            # print("action: ",action)
            next_state, reward, done, info = env.step(action)
            score += reward
        scores.append(score)

        print("Episode: {}/{}, score: {}".format(episode + 1, episodes, score))
    print("Finished evaluating", algorithm)
    print("Final average score over " + str(episodes) + " episodes: " + str(mean(scores)))
    return scores


if __name__ == "__main__":
    answer = input('Type...to... \n'
                   'TRAINING: ' + str(number_of_training_timesteps) + ' steps\n'
                                                                      '"a" or "Enter" ... train the ARA-DQN agent \n'
                                                                      '"b" ... train the PPO agent \n'
                                                                      'EVALUATION: ' + str(
        number_of_evaluation_episodes) + ' episodes\n'
                                         '"c" ... evaluate with ARA-DQN agent \n'
                                         '"d" ... evaluate with default action ' + str(default_action) + '\n'
                                                                                                         '"e" ... evaluate with random action \n'
                                                                                                         '"f" ... evaluate with PPO agent \n'
                   )
    if answer == "a" or answer == "":
        train_ARA_DiRL(number_of_machines=number_of_machines)
    if answer == "f":
        evaluate_other_algos("ppo", number_of_machines=number_of_machines, csv_metrics_per_episode=csv_metrics_per_episode, csv_rewards_per_period=csv_rewards_per_period)
    if answer == "c":
        evaluate_with_ARA_DiRL(number_of_machines=number_of_machines)
    if answer == "d":
        evaluate_with_default_action(number_of_machines=number_of_machines)
    if answer == "e":
        evaluate_with_random_action(number_of_machines=number_of_machines)
    if answer == "ppo" or answer == "b":
        train_ppo(number_of_machines=number_of_machines)
    if answer == "a2c":
        train_a2c(number_of_machines=number_of_machines)
    if answer == "vanilla_dqn":
        train_vanilla_dqn(number_of_machines=number_of_machines)
    if answer == "ppo_eval":
        evaluate_other_algos("ppo", number_of_machines=number_of_machines, csv_metrics_per_episode=False, csv_rewards_per_period=False)
    if answer == "a2c_eval":
        evaluate_other_algos("a2c", number_of_machines, csv_metrics_per_episode, csv_rewards_per_period)
    if answer == "vanilla_dqn_eval":
        evaluate_other_algos("vanilla_dqn", number_of_machines, csv_metrics_per_episode, csv_rewards_per_period)
