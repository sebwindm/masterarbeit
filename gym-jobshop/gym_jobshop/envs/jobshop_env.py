# External module imports
import numpy as np
import gym
from gym import error, spaces, logger, utils
from gym.utils import seeding

# Own module imports
from gym_jobshop.envs.src import main


class JobShopEnv(gym.Env):
    """
    Description:
        Units of time measurement are steps, periods and episodes.
        1 period = 960 steps
        1 episode = 8000 periods
        The reinforcement learning agent can take an action once every period by using the step() method

    Source:

    Observation:

    Actions:
        Type: Discrete(3)
        Num |   Action
        0   |   Decrease capacity
        1   |   Keep capacity
        2   |   Increase capacity

    Reward:
        Reward is the final cost after each episode
    Starting state:

    Episode Termination:

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # self.seed()
        self.viewer = None
        self.state = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(3)

    # def seed(self, seed=None):
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     return [seed]

    def step(self, action):
        """
        Step one period ahead.
        :param action: action must be either 0, 1 or 2 and is used to adjust the processing times of machines
        More info at  main.py -> adjust_processing_times()
        :return: return state, reward, done, {}
        """
        # Check if action is valid
        assert self.action_space.contains(action), "%r (%s) invalid action" % (action, type(action))

        # self.state = ()
        #
        # done = False
        #
        # if not done:
        #     reward = 1.0
        # elif self.steps_beyond_done is None:
        #     # Pole just fell!
        #     self.steps_beyond_done = 0
        #     reward = 1.0
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned done = True. You "
        #             "should always call 'reset()' once you receive 'done = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_done += 1
        #     reward = 0.0

        # Adjust processing times (or capacity) depending on action
        main.adjust_processing_times(action)
        # Step one period ( = 960 steps) ahead
        for i in range(960):
            main.step_one_step_ahead()


        reward = main.get_results()
        done = False
        return np.array(self.state), reward, done, {}


    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        main.reset()

        return np.array(self.state)

    def render(self, mode='human', close=False):
        print("Nothing to show")
