# Python stdlib imports
import csv
# External module imports
import numpy as np
import gym
# Custom module imports
from gym_jobshop.envs.src import global_settings, main


def get_environment_state():
    """
    Retrieve observation state from inside the main simulation
    """
    return np.array(main.get_current_environment_state()).flatten()


def evaluate_episode():
    """
    Return various performance metrics from inside the simulation.
    Description can be found at src/performance_measurement.py -> evaluate_episode()
    """
    return main.get_episode_results()


class JobShopEnv(gym.Env):
    """
    Description:
        Units of time measurement are steps, periods and episodes.
        1 period = 960 steps
        1 episode = 8000 periods
        The reinforcement learning agent can take an action once every period by using the step() method
        TODO: further documentation needed
    Source:

    Observations:
        Type: Box(low=self.low, high=self.high)
        The observation space contains information on the current state of some production metrics.
        It shows the amount of orders in each stage of production, filtered by the product type of orders and
        sorted by earliness/lateness measured in periods (sorting only applies for order pool, FGI and shipped orders).
        The state is always one array of arrays, containing six sub-arrays,
        and one sub-array per product type (1-6). Each sub-array has 22
        elements, which contain the amount of orders inside the six production steps/stages.
        While the production simulation works with arrays, the Gym environment flattens
        all arrays into one single array before it gets passed to the agent as an observation.
        The structure remains the same after flattening, just all square brackets and commas are removed.
        See the examples below for a better understanding. Flattening is done with numpy.ndarray.flatten

        Normalization: currently the observation state is not normalized. Should it be required
        to use a normalized observation state, this must be done inside the agent/algorithm.

    Structure of the observation state without real numbers and before flattening:
        Order pool              | WC1 | WC2 | WC3 | FGI     | Shipped
    1   x,x,x,x,x,x,x,x,x,x     | x   | x   | x   | x,x,x,x | x,x,x,x,x
    2   x,x,x,x,x,x,x,x,x,x     | x   | x   | x   | x,x,x,x | x,x,x,x,x
    3   x,x,x,x,x,x,x,x,x,x     | x   | x   | x   | x,x,x,x | x,x,x,x,x
    4   x,x,x,x,x,x,x,x,x,x     | x   | x   | x   | x,x,x,x | x,x,x,x,x
    5   x,x,x,x,x,x,x,x,x,x     | x   | x   | x   | x,x,x,x | x,x,x,x,x
    6   x,x,x,x,x,x,x,x,x,x     | x   | x   | x   | x,x,x,x | x,x,x,x,x
    -> row 1-6 = product type
    -> x = amount of orders in the respective production stage
    -> more than one x per production stage = order amounts are separated and sorted by
        earliness/lateness/due dates in periods

    Example with real numbers after flattening. This is how the observation state looks when
    it gets passed to the agent:
    [0 0 3 0 3 0 0 0 0 0 0 0 0 4 0 0 0 0 0 1 0 0 0 3 2 0 4 2 0 2 0 0 0 0 0 2 0
     0 0 0 0 0 0 0 0 0 0 2 1 3 2 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 3 2 0 1 2 0 3
     1 0 0 0 0 5 0 0 0 0 0 0 1 0 0 2 0 1 0 1 0 1 2 0 0 0 0 3 0 0 0 0 1 0 0 0 0
     1 1 3 3 0 2 1 1 0 0 0 0 2 0 0 0 0 0 1 0 0]


        Observation:
        Type: Box(132,)
        Num    Observation                                              Min         Max
        0       No. of orders in order pool due in 1 period             0           15
        1       No. of orders in order pool due in 2 periods            0           15
        2       No. of orders in order pool due in 3 periods            0           15
        3       No. of orders in order pool due in 4 periods            0           15
        4       No. of orders in order pool due in 5 periods            0           15
        5       No. of orders in order pool due in 6 periods            0           15
        6       No. of orders in order pool due in 7 periods            0           15
        7       No. of orders in order pool due in 8 periods            0           15
        8       No. of orders in order pool due in 9 periods            0           15
        9       No. of orders in order pool due in 10 periods           0           15
        10      No. of orders in work center 1                          0           15
        11      No. of orders in work center 2                          0           15
        12      No. of orders in work center 3                          0           15
        13      No. of orders in FGI early by 1 period                  0           30
        14      No. of orders in FGI early by 2 periods                 0           15
        15      No. of orders in FGI early by 3 periods                 0           15
        16      No. of orders in FGI early by 4 or more periods         0           15
        17      No. of orders shipped in time                           0           15
        18      No. of orders shipped late by 1 period                  0           15
        19      No. of orders shipped late by 2 periods                 0           15
        20      No. of orders shipped late by 3 periods                 0           15
        21      No. of orders shipped late by 4 or more periods         0           15
        ... AND SO ON for the other product types. The 22 observations above are just for product type 1.
        All other product types have the exact same structure, so this goes up to 22*6 = 132 observations

    Actions:
        Type: Discrete(3)
        Num |   Action
        0   |   Keep capacity
        1   |   Increase capacity by 25%
        2   |   Increase capacity by 50%

    Reward:
        Reward is the final cost after each episode. It is always a negative number, e.g. -246
        It is recommended to normalize the reward either by setting self.normalization_denominator
        to a value higher than 1 or by normalizing inside the agent/algorithm.

    Starting state:
        All stages of production start with zero orders inside them, thus the starting state is an
        array with six arrays, each consisting of 22 elements that all have the value 0.

    Episode Termination:
        Episodes end after 8000 periods, there are no other termination conditions.
    """

    def __init__(self,
                 csv_metrics_per_episode: bool = False,
                 # Creates a CSV file containing metrics like costs, lateness, etc
                 csv_rewards_per_period: bool = False,
                 # Creates a CSV file with metrics for algorithm training (period, reward)
                 #####
                 number_of_machines: int = 3,  # Amount of machines used in the simulation. Must be 1 or 3
                 global_prefix: str = "",  # Prefix for all file outputs from this environment
                 simple_observation_space: bool = False  # Set to true to activate a "simple" observation space
                 # which only contains one value: the amount of orders inside work center 3
                 # it is still a vector with 132 values, but all other values are 0
                 ):
        main.initialize_random_numbers()
        self.episode_counter = 0
        self.period_counter = 0
        self.cost_rundown = [0, 0, 0, 0]
        self.csv_metrics_per_episode = csv_metrics_per_episode
        self.csv_rewards_per_period = csv_rewards_per_period
        self.number_of_machines = number_of_machines
        self.global_prefix = global_prefix
        self.random_seed = 0  # this random seed is currently only used for the agent,
        # not inside the job shop simulation
        self.is_simple_observation_space_active = simple_observation_space
        ########
        # Set shop type depending on number of machines given at
        # initialization of JobShopEnv object:
        if number_of_machines == 3:
            main.setup_environment(number_of_machines)
            global_settings.shop_type = "job_shop"
        elif number_of_machines == 1:
            main.setup_environment(number_of_machines)
            global_settings.shop_type = "job_shop_1_machine"
        else:
            raise ValueError(
                "Please set JobShopEnv.number_of_machines to 1 or 3, e.g. with gym.make('jobshop-v0', number_of_machines=3)")
        print("Created an environment with", self.number_of_machines, "machines using shop type",
              global_settings.shop_type)
        self.action_space = gym.spaces.Discrete(3)  # discrete action space with three possible actions
        # Below is the lower boundary of the observation space. It is an array of 132 elements, all are 0.
        # Due to the state logic of the production system, there cannot be any state below 0.
        self.low = np.empty(132, dtype=np.float32)
        self.low.fill(0)
        # Below is the upper boundary of the observation space. It is an array of 132 elements, all are
        # either 15 or 30. The number should be as low as possible, but high enough that the real numbers
        # don't exceed the upper limit. 15 was chosen as the upper limit for most values, but for some that
        # tend to exceed 15 the upper limit of 30 was chosen.
        # As the neural network uses the upper boundary of the observation space as a denominator/bottom
        # in fractions, a higher number as upper boundary would add unnecessary noise, whereas a lower
        # number reduces noise. Thus it's advisable to keep the upper boundary numbers as close to the highest
        # occurring real numbers as possible.
        self.high = np.array([
            # prod type 1
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  # order pool
            15, 15, 15,  # work centers
            30, 15, 15, 15,  # FGI
            15, 15, 15, 15, 15,  # shipped
            # prod type 2
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  # order pool
            15, 15, 15,  # work centers
            30, 15, 15, 15,  # FGI
            15, 15, 15, 15, 15,  # shipped
            # prod type 3
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  # order pool
            15, 15, 15,  # work centers
            30, 15, 15, 15,  # FGI
            15, 15, 15, 15, 15,  # shipped
            # prod type 4
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  # order pool
            15, 15, 15,  # work centers
            30, 15, 15, 15,  # FGI
            15, 15, 15, 15, 15,  # shipped
            # prod type 5
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  # order pool
            15, 15, 15,  # work centers
            30, 15, 15, 15,  # FGI
            15, 15, 15, 15, 15,  # shipped
            # prod type 6
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  # order pool
            15, 15, 15,  # work centers
            30, 15, 15, 15,  # FGI
            15, 15, 15, 15, 15  # shipped
        ])

        self.observation_space = gym.spaces.flatten_space(
            gym.spaces.Box(low=self.low, high=self.high, dtype=np.float32))
        self.state = self.reset()
        self.csv_results_file_name = '../Evaluation/' + self.global_prefix + "_" + \
                                     str(self.number_of_machines) + '_env_metrics_per_episode.csv'
        # Create CSV file for writing results after each episode
        if self.csv_metrics_per_episode is True:
            with open(self.csv_results_file_name, mode='w') as results_CSV:
                results_writer = csv.writer(results_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                results_writer.writerow(['Episode', 'total_cost', 'wip_cost', 'fgi_cost',
                                         'lateness_cost', 'overtime_cost', 'amount_of_shipped_orders',
                                         'bottleneck_utilization', 'late_orders', 'early_orders',
                                         'sum_of_lateness', 'sum_of_tardiness', 'average_flow_time'])
                results_CSV.close()

        # Create CSV file for writing training metrics after each period
        # Not to be confused with rewards_per_period.csv from the ARA-DiRL algorithm
        self.env_rewards_file_name = '../Evaluation/' + self.global_prefix + "_" + \
                                     str(self.number_of_machines) + '_env_rewards_per_period.csv'
        if self.csv_rewards_per_period is True:
            with open(self.env_rewards_file_name, mode='w') as rewards_csv:
                results_writer = csv.writer(rewards_csv, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                results_writer.writerow(['Period', 'Reward'])
                rewards_csv.close()

    def step(self, action):
        """
        Step one period (= 960 simulation steps) ahead.
        :param action: Integer number, must be either 0, 1 or 2. Used to adjust the processing times of
        bottleneck machines. More info at main.py -> adjust_processing_times()
        :return:
        * observation: array of arrays, contains production system metrics. See class JobShopEnv docstrings
        * reward: floating-point number, indicates the cost that accumulated during the period
        * done: boolean value, tells whether to terminate the episode
            -> Note that your algorithm should call env.reset() when done is returned as True.
            -> The environment doesn't reset itself, even if done is returned as True.
        * info: not used, but some algorithms expect at least empty curly brackets
        """
        # Verify if action is valid
        assert self.action_space.contains(action), "%r (%s) invalid action" % (action, type(action))
        # Adjust processing times of bottleneck machines (capacity) depending on action
        main.adjust_processing_times(action)
        # Step one period ( = 960 steps) ahead
        reward, environment_state1, self.cost_rundown, done = main.step_one_period_ahead()
        self.period_counter += 1
        self.write_csv_rewards_per_period(reward)

        def get_orders_in_work_center_3(state):
            list_of_wc3_indices = [12,34,56,78,100,122]
            orders_in_wc3 = 0
            for i in list_of_wc3_indices:
                orders_in_wc3 += state[i]
            return orders_in_wc3

        # Retrieve new state from the environment
        # With the parameter is_simple_observation_space_active = True
        # you can return a very simple observation that follows the design of the normal observation space
        # in that it is a list with 132 values, but of these values only the first value is relevant
        # The first value of that observation is the amount of orders inside work center 3 (inventory + machine)
        # This allows the development of heuristics/training of RL agents with just that limited information
        if self.is_simple_observation_space_active == False:
            self.state = get_environment_state()  # return normal observation
        else:
            complex_state = get_environment_state()
            empty_state = np.empty(132, dtype=np.float32)
            empty_state.fill(0)
            simple_state = empty_state
            simple_state[0] = get_orders_in_work_center_3(complex_state)
            self.state = simple_state
        observation = self.state

        info = {}  # Not used, but is expected to exist by some algorithms
        if self.period_counter % 8000 == 0:
            print("Period " + str(self.period_counter) + " done")
        if done is True:
            self.episode_counter += 1
            if self.csv_metrics_per_episode is True:
                self.write_csv_metrics_per_episode()
        return observation, reward, done, info

    def reset(self):
        """
        Reset the environment to the default values
        """
        main.reset()
        self.random_seed += 1
        self.state = get_environment_state()
        return self.state

    def get_cost_rundown(self):
        """
        Return a list of which costs occurred where. Useful for debugging.
        Currently used inside the agent's custom callback.
        """
        return self.cost_rundown

    def debug_observation(self):  # todo: used for debugging, may be removed in final release
        """
        Return the observation state with descriptive text for easier debugging
        Example for what gets returned:
        ['OP1: ', 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 'WC1:', 0, 0, 0, 'FGI1: ', 0, 0, 0, 0,
        'SG1: ', 0, 0, 0, 0, 0, 'OP2: ', 0, 0, 0, 0, 1, 0, 0, 3, 1, 0, 'WC2:', 0, 0, 0,
        'FGI2: ', 0, 0, 0, 0, 'SG2: ', 0, 0, 0, 0, 0, 'OP3: ', 0, 0, 0, 0, 2, 0, 2, 0, 2, 0,
        'WC3:', 0, 0, 0, 'FGI3: ', 0, 0, 0, 0, 'SG3: ', 0, 0, 0, 0, 0,
        'OP4: ', 0, 0, 0, 0, 0, 1, 1, 0, 5, 0, 'WC4:', 0, 0, 0, 'FGI4: ', 0, 0, 0, 0,
        'SG4: ', 0, 0, 0, 0, 0, 'OP5: ', 0, 0, 0, 0, 1, 2, 1, 3, 0, 0, 'WC5:', 0, 0, 0,
        'FGI5: ', 0, 0, 0, 0, 'SG5: ', 0, 0, 0, 0, 0, 'OP6: ', 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
        'WC6:', 0, 0, 0, 'FGI6: ', 0, 0, 0, 0, 'SG6: ', 0, 0, 0, 0, 0]
        """
        observation = get_environment_state().tolist()
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
        time1, time2 = main.get_current_time()
        return observation, time1, time2

    def get_observation(self):
        return get_environment_state()

    def write_csv_metrics_per_episode(self):
        """
        Write episode results as a row to the CSV
        The total_cost takes into account the cost reset after the warmup period.
        """
        results = evaluate_episode()
        total_cost, wip_cost, fgi_cost, lateness_cost, overtime_cost, amount_of_shipped_orders, \
        bottleneck_utilization, late_orders, early_orders, sum_of_lateness, sum_of_tardiness, \
        average_flow_time = [results[i] for i in range(len(results))]
        with open(self.csv_results_file_name, mode='a') as results_CSV:
            results_writer = csv.writer(results_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow([self.episode_counter, total_cost, wip_cost, fgi_cost,
                                     lateness_cost, overtime_cost, amount_of_shipped_orders,
                                     bottleneck_utilization, late_orders, early_orders, sum_of_lateness,
                                     sum_of_tardiness, average_flow_time])
            results_CSV.close()
        return

    def write_csv_rewards_per_period(self, reward):
        """
        The reward per period does not take into account the cost reset after the warmup period.
        Thus the reward written in this function is different from the
        reward/total_cost from write_csv_metrics_per_episode()
        """
        with open(self.env_rewards_file_name, mode='a') as rewards_csv:
            results_writer = csv.writer(rewards_csv, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow([self.period_counter, reward])
            rewards_csv.close()
        return
