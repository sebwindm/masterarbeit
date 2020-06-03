# External module imports
import numpy as np
import gym
from gym import error, spaces, logger, utils
from gym.utils import seeding

# Custom module imports
from gym_jobshop.envs.src import main


class JobShopEnv(gym.Env):
    """
    Description:
        Units of time measurement are steps, periods and episodes.
        1 period = 960 steps
        1 episode = 8000 periods
        The reinforcement learning agent can take an action once every period by using the step() method

    Source:

    Observations:
        Type: Box

        - Amount of orders per product type inside the order pool. Example: [18,23,8,25,12,11]
        - Amount of orders per product type inside the work centers (machines + WIPs), three work centers in total.
            Example:    [18,23,8,25,12,11]
                        [18,23,8,25,12,11]
                        [18,23,8,25,12,11]
        - Amount of orders per product type inside finished goods inventory [18,23,8,25,12,11]
        - Amount of orders per product type inside shipped goods inventory [18,23,8,25,12,11]

                        | Order pool | Work center 1 | Work center 2 | Work center 3 | FGI | Shipped goods
        Product type 1  |   [18,23,8,25,12,11]
        Product type 2  |   [12,52,44,64,3,33]
        Product type 3  |   [x,x,x,x,x,x] usw mit den richtigen Zahlen
        Product type 4  |   [x,x,x,x,x,x]
        Product type 5  |   [x,x,x,x,x,x]
        Product type 6  |   [x,x,x,x,x,x]


    Actions:
        Type: Discrete(3)
        Num |   Action
        0   |   Keep capacity
        1   |   Increase capacity
        2   |   Increase capacity

    Reward:
        Reward is the final cost after each episode
    Starting state:

    Episode Termination:
        Episodes end after 8000 periods, there are no other termination conditions.
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
        Step one period (= 960 simulation steps) ahead.
        :param action: Integer number, must be either 0, 1 or 2. Used to adjust the processing times of all machines
        More info at  main.py -> adjust_processing_times()
        :return: observation (represents an observed state, indicated by an integer number),
        reward (floating-point number),
        done (boolean value, tells whether to terminate the episode),
        info (diagnostic information for debugging)
        """
        # Verify if action is valid
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

        self.state = main.get_current_environment_state()
        observation = np.array(self.state)

        reward = main.get_results_from_this_period()

        done = False # must be True or False. Not used since episodes always run for the full duration
        info = None # Not used
        return observation, reward, done, info


    def reset(self):
        main.reset()
        self.state = main.get_current_environment_state()

        return np.array(self.state)

    def render(self, mode='human', close=False):
        raise Exception("Function render() is not supported in this environment.")
