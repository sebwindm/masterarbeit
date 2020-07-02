import gym, gym_jobshop
from ReinforcementLearning.dqn_average_reward_adjusted import DQN_average_reward_adjusted
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
#env = gym.make('CartPole-v1')
env = gym.make('jobshop-v0')

# Instantiate the agent
model = DQN_average_reward_adjusted('MlpPolicy', env, verbose=0,learning_starts = 100)

from stable_baselines3.common.callbacks import BaseCallback


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
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        print("Training start")
        #print(self.model) #<stable_baselines3.dqn.dqn.DQN object at 0x7f7e8760dd30>

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
        #print(self.model.policy)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print("Training end")
        pass


custom_callback = CustomCallback()

# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
# print("Random agent avg reward, std reward: ", mean_reward, std_reward)

# Train the agent
model.learn(total_timesteps=100000, callback=custom_callback)
# Save the agent
model.save("dqn_pytorch")

# Load the trained agent
# model = DQN.load("dqn_pytorch")

# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
# print("PyTorch DQN agent avg reward, std reward: ", mean_reward, std_reward)
# Enjoy trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)

