##### Filter tensorflow version warnings #####
import os

# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging

tf.get_logger().setLevel(logging.ERROR)
##### ##### ##### ##### ##### ##### #####

import gym_jobshop
import gym, time
from statistics import mean

import stable_baselines
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.callbacks import BaseCallback
env = gym.make('jobshop-v0')


def train_DQN():
    simulation_start_time = time.time()
    model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./gym_jobshop_tensorboard_logs/")
    custom_callback = CustomCallback()
    # model = MlpPolicy
    # Call Tensorboard logs from a terminal in folder "masterarbeit" (root folder of the project)
    # tensorboard --logdir ReinforcementLearning/gym_jobshop_tensorboard_logs/DQN_1

    # keyboard input: was will der user machen
    # a) trainiere für x steps
    # 10000
    # b) gebe aktuelle werte aus
    # dqn proba_step(aktuellster state vom environment als observation + 2 fixe observations
    # dqn step(s.o.
    model.learn(total_timesteps=10000, callback=custom_callback)

    model.save("deepq_jobshop")
    print("Training finished after " + str(round(time.time() - simulation_start_time, 4)) + " seconds")
    return


def predict_with_DQN():
    simulation_start_time = time.time()
    model = DQN.load("deepq_jobshop")

    scores = []  # list of final scores after each episode
    episodes = 1  # 30
    max_periods = 8000  # 8000

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


def delete_tensorboard_logs():
    import shutil
    shutil.rmtree('./gym_jobshop_tensorboard_logs')
    print("Deleted all Tensorboard logs")
    return





class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        print("TRAINING START")
        # print("model.sess: ", self.model.sess)
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        ###### GET Q-VALUES
        # print(model.policy().step(obs=env.get_observation(), state=None, mask=None, deterministic=False))
        # policy.q_values
        #####

        # print("self.model.policy = ", self.model.policy.q_values)

        # print("action_space",self.model.action_space)
        # print("q values: ", self.model.policy.get_q_values(self.model.policy))

        # print("proba_step: ",self.model.policy.proba_step(self, env.state))
        # sess = self.model.sess
        # result: self.model =  <stable_baselines.deepq.dqn.DQN object at 0x7fc110406080>

        if self.model.num_timesteps % 1000 == 0:
            answer = input('Continue training? y or n \n')
            if answer == "y":
                return True
            elif answer == "n":
                self.model.save("deepq_jobshop")
                print("Model saved")
                return False

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


if __name__ == "__main__":
    answer = input('Type...to.. \n'
                   '"a" train the model (creates model file)\n'
                   '"b" delete Tensorboard logs\n'
                   '"c" predict using the model file\n'
                   '"d" predict 10 episodes and print mean + std of reward\n '
                   '"e" print various model parameters\n'
                   '"f" get Q-Values (BROKEN)\n'
                   )
    if answer == "a":
        train_DQN()
    if answer == "b":
        delete_tensorboard_logs()
    if answer == "c":
        predict_with_DQN()
    if answer == "d":
        evaluate_policy()

    if answer == "e":
        loaded_model = DQN.load("deepq_jobshop", verbose=1)
        # show the saved hyperparameters
        print("loaded:", "gamma =", loaded_model.gamma,
              "learning_rate =", loaded_model.learning_rate,
              "seed =", loaded_model.seed)
        print("Probability of the three possible actions for current state: ",
              str(loaded_model.action_probability(env.get_observation())))
        print(loaded_model.get_parameter_list())
        print(loaded_model.get_parameters())

    if answer == "f":
        model = DQN.load("deepq_jobshop")
        print(model)
        print(model.policy)
        # print(model.policy().step(obs=env.get_observation(), state=None, mask=None, deterministic=False))
