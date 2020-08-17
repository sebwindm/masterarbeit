# External module imports
import numpy as np
import gym, csv, datetime

# Custom module imports
from gym_jobshop.envs.src import main


def get_environment_state():
    #print("state from env: ",np.array(main.get_current_environment_state()).flatten())
    return np.array(main.get_current_environment_state()).flatten()


class JobShopEnv(gym.Env):
    """
    Description:
        Units of time measurement are steps, periods and episodes.
        1 period = 960 steps
        1 episode = 8000 periods
        The reinforcement learning agent can take an action once every period by using the step() method

    Source:

    Observations:
        Type: Box(low=0, high=30, shape=(1, 96), dtype=np.int_)
        The observation space contains information on the current state of some production metrics.
        It shows the amount of orders in each stage of production, filtered by the product type of orders and
        sorted by earliness/lateness measured in periods (sorting only applies for order pool, FGI and shipped orders).
        The state is always one array of arrays, containing six sub-arrays,
        and one sub-array per product type (1-6). Each array has 16
        elements, which contain the amount of orders inside the six production steps/stages.

    Example state in theory:
        order pool | WC1 | WC2 | WC3 | FGI     | Shipped
    1   x,x,x,x    | x   | x   | x   | x,x,x,x | x,x,x,x,x
    2   x,x,x,x    | x   | x   | x   | x,x,x,x | x,x,x,x,x
    3   x,x,x,x    | x   | x   | x   | x,x,x,x | x,x,x,x,x
    4   x,x,x,x    | x   | x   | x   | x,x,x,x | x,x,x,x,x
    5   x,x,x,x    | x   | x   | x   | x,x,x,x | x,x,x,x,x
    6   x,x,x,x    | x   | x   | x   | x,x,x,x | x,x,x,x,x
    -> 1-6 = product type
    -> x = amount of orders in the respective production stage
    -> more than one x per production stage = order amounts are separated and sorted by
        earliness/lateness/duedates in periods

    Actual state example:
    [ 0  2  0  6  0  0  0  9  0  0  0  0  0  0  1  0  0  4  2  6  0  0  0  2
  0  0  0  0  0  0  0  0  0  6  0 10  3  0  0  2  0  0  0  0  1  0  0  0
  0  3  2  7  1  0  0  2  0  0  0  0  1  0  0  1  0  3  0  7  1  0  0  3
  0  0  0  0  1  0  0  0  0  4  1  8  1  0  0  2  0  0  0  0  0  2  0  0]


    TODO: FURTHER EXPLANATION REQUIRED

        Observation:
        Type: Box(96,)
        Num    Observation                                              Min         Max
        0       No. of orders in order pool due in 1 period             0           15
        1       No. of orders in order pool due in 2 periods            0           15
        2       No. of orders in order pool due in 3 periods            0           15
        3       No. of orders in order pool due in 4 or more periods    0           30
        4       No. of orders in work center 1                          0           15
        5       No. of orders in work center 2                          0           15
        6       No. of orders in work center 3                          0           15
        7       No. of orders in FGI early by 1 period                  0           30
        8       No. of orders in FGI early by 2 periods                 0           15
        9       No. of orders in FGI early by 3 periods                 0           15
        10      No. of orders in FGI early by 4 or more periods         0           15
        11      No. of orders shipped in time                           0           15
        12      No. of orders shipped late by 1 period                  0           15
        13      No. of orders shipped late by 2 periods                 0           15
        14      No. of orders shipped late by 2 periods                 0           15
        15      No. of orders shipped late by 4 or more periods         0           15
        ... AND SO ON for the other product types. The 16 observations above are just for product type 1.
        All other product types have the exact same structure, so this goes up to 16*6 = 96 observations
    Actions:
        Type: Discrete(3)
        Num |   Action
        0   |   Keep capacity
        1   |   Increase capacity by 25%
        2   |   Increase capacity by 50%

    Reward:
        Reward is the final cost after each episode. It is always a negative number, e.g. -246601

    Starting state:
        All stages of production start with zero orders inside them, thus the starting state is an
        array with six arrays, each consisting of six elements that all have the value 0.

    Episode Termination:
        Episodes end after 8000 periods, there are no other termination conditions.
    """

    # metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None
        main.initialize_random_numbers()
        self.episode_counter = -1
        self.period_counter = 0
        self.state = self.reset()

        self.action_space = gym.spaces.Discrete(3) # discrete action space with three possible actions
        # Below is the lower boundary of the observation space. It is an array of 96 elements, all are 0.
        # Due to the state logic of the production system, there cannot be any state below 0.
        self.low = np.empty(96, dtype=np.float32)
        self.low.fill(0)
        # Below is the upper boundary of the observation space. It is an array of 96 elements, all are
        # either 15 or 30. The number should be as low as possible, but high enough that the real numbers
        # don't exceed the upper limit. 15 was chosen as the upper limit for most values, but for some that
        # tend to exceed 15 the upper limit of 30 was chosen.
        # As the neural network uses the upper boundary of the observation space as a denominator/bottom
        # in fractions, a higher number as upper boundary would add unnecessary noise, whereas a lower
        # number reduces noise. Thus it's advisable to keep the upper boundary numbers as close to the highest
        # occuring real numbers as possible.
        self.high = np.array([
            # prod type 1
            15,15,15,30, # order pool
            15,15,15, # work centers
            30,15,15,15, # FGI
            15,15,15,15,15, # shipped
            # prod type 2
            15,15,15,30, # order pool
            15,15,15, # work centers
            30,15,15,15, # FGI
            15,15,15,15,15, # shipped
            # prod type 3
            15,15,15,30, # order pool
            15,15,15, # work centers
            30,15,15,15, # FGI
            15,15,15,15,15, # shipped
            # prod type 4
            15,15,15,30, # order pool
            15,15,15, # work centers
            30,15,15,15, # FGI
            15,15,15,15,15, # shipped
            # prod type 5
            15,15,15,30, # order pool
            15,15,15, # work centers
            30,15,15,15, # FGI
            15,15,15,15,15, # shipped
            # prod type 6
            15,15,15,30, # order pool
            15,15,15, # work centers
            30,15,15,15, # FGI
            15,15,15,15,15 # shipped
                              ])

        self.observation_space = gym.spaces.flatten_space(
             gym.spaces.Box(low=self.low, high=self.high, dtype=np.float32))

        print(self.low)
        print(self.high)
    def step(self, action, debug=True):
        """
        Step one period (= 960 simulation steps) ahead.
        :param action: Integer number, must be either 0, 1 or 2. Used to adjust the processing times of
        bottleneck machines. More info at main.py -> adjust_processing_times()
        :return: observation (array of arrays, contains production metrics),
        reward (floating-point number, indicates the cost that accumulated during the period),
        done (boolean value, tells whether to terminate the episode),
        info (diagnostic information for debugging)
        """
        # Verify if action is valid
        assert self.action_space.contains(action), "%r (%s) invalid action" % (action, type(action))
        # Adjust processing times of bottleneck machines (capacity) depending on action
        main.adjust_processing_times(action)
        # Step one period ( = 960 steps) ahead
        for i in range(960):
            main.step_one_step_ahead()
        self.period_counter += 1
        # Retrieve new state from the environment
        self.state = get_environment_state()
        observation = self.state
        # Obtain cost that accumulated during this period
        reward = main.get_results_from_this_period() / 300
        done = main.is_episode_done()  # Episode ends when 8000 periods are reached
        info = {}  # Not used

        if self.period_counter % 5000 == 0:
            print("Period " + str(self.period_counter) + " done")
        return observation, reward, done, info

    def reset(self):
        main.reset()
        self.state = get_environment_state()
        self.episode_counter += 1
        return self.state


    def post_results(self):
        print(main.get_info())

    def get_observation(self):
        return get_environment_state()
