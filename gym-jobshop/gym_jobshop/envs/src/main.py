# Own module imports
from gym_jobshop.envs.src import global_settings, environment, order_generation, debugging, order_processing, \
    order_release, order_movement, performance_measurement, class_Machine

# Python native module (stdlib) imports
import time, random


def initialize_random_numbers():
    """
    Initialize random number stream
    """
    global_settings.random_seed = -1
    return


def setup_environment(number_of_machines):
    """
    Call this function after switching shop types
    """
    if number_of_machines == 3:
        environment.machine_A = class_Machine.Machine(
            "Machine A", 30, 130, global_settings.processingtime_machine_A_job_shop)  # default 80
        environment.machine_B = class_Machine.Machine(
            "Machine B", 30, 130, global_settings.processingtime_machine_B_job_shop) # default 77.5
        environment.machine_C = class_Machine.Machine(
            "Machine C", 65, 125, global_settings.processingtime_machine_C_job_shop)  # default 95
        environment.list_of_all_machines = [environment.machine_A, environment.machine_B, environment.machine_C]
        # Generate WIP (work in process) inventories
        # each WIP inventory is associated with one machine (and each machine with one inventory)
        # when an order arrives at a machine, the order first gets placed inside the WIP inventory
        # if the machine is not processing an order, it pulls one order from the WIP according to certain rules
        environment.wip_A = []
        environment.wip_B = []
        environment.wip_C = []
        environment.list_of_all_wip_elements = [environment.wip_A, environment.wip_B, environment.wip_C]
        environment.list_of_inventories = [environment.wip_A, environment.wip_B, environment.wip_C,
                                           environment.finished_goods_inventory, environment.shipped_orders,
                                           environment.order_pool]
        environment.bottleneck_machine = environment.machine_C

    elif number_of_machines == 1:
        environment.machine_A = class_Machine.Machine(
            "Machine A", 30, 130, global_settings.processingtime_machine_A_job_shop_1_machine)  # 106.1999115 gives roughly 90% utilization
        environment.list_of_all_machines = [environment.machine_A]
        # Generate WIP (work in process) inventories
        # each WIP inventory is associated with one machine (and each machine with one inventory)
        # when an order arrives at a machine, the order first gets placed inside the WIP inventory
        # if the machine is not processing an order, it pulls one order from the WIP according to certain rules
        environment.wip_A = []
        environment.list_of_all_wip_elements = [environment.wip_A]
        environment.list_of_inventories = [environment.wip_A, environment.finished_goods_inventory,
                                           environment.shipped_orders, environment.order_pool]
        environment.bottleneck_machine = environment.machine_A

    else:
        raise ValueError("Wrong shop_type",global_settings.shop_type)

    return


def reset():
    """
    Reset the simulation parameters inside the environment to the default values.
    This clears all inventories, lists and machines and resets all metrics like costs etc.
    main.reset() is not to be confused with JobShopEnv.reset() of the actual Gym environment in jobshop_env.py
    """
    global_settings.random_seed += 1
    random.seed(global_settings.random_seed)
    global_settings.reset_global_settings()
    performance_measurement.reset_all_costs()
    environment.reset_machines()
    environment.reset_inventories()
    global_settings.bottleneck_utilization_per_step = 0
    debugging.verify_reset()
    return get_current_environment_state()


def get_current_environment_state():
    """
    NOTE: the text below is outdated. The latest documentation of the environment state is always in the
    docstring for class JobShopEnv() inside jobshop_env.py
    TODO: edit text below

    Get environment's state as an array, with each array element representing one of the possible product_types of orders.
    Each array element contains the amounts of orders in each stage of production for the associated product_type.
    The order within the array elements is as follows: Order pool | Work center 1 | Work center 2 | Work center 3 |
    FGI | Shipped goods. For the order pool, FGI and shipped goods, there is not just the total amount of orders
    inside that stage, but a sub-array containing the amounts of orders by earliness/lateness in periods.
    For example, the order pool sub-array contains four elements, the first indicating the amount of orders
    due in 1 period, the 2nd for orders due in 2 periods up to the 4th element which  features orders due in 4
    or more periods. The FGI array contains order amounts sorted by earliness(early by 1 up to 4+ periods) and
    the shipped orders contain the amounts sorted by lateness (in time, late by 1 up to 4+ periods)
    :return: state, an array with 22x6 (= 132) elements

    Example state:
    order pool | WC1 | WC2 | WC3 | FGI | Shipped
1   x,x,x,x | x | x | x | x,x,x,x | x,x,x,x,x
2   x,x,x,x | x | x | x | x,x,x,x | x,x,x,x,x
3   x,x,x,x | x | x | x | x,x,x,x | x,x,x,x,x
4   x,x,x,x | x | x | x | x,x,x,x | x,x,x,x,x
5   x,x,x,x | x | x | x | x,x,x,x | x,x,x,x,x
6   x,x,x,x | x | x | x | x,x,x,x | x,x,x,x,x
1-6 => product type
    """
    state = []
    for product_type_element in [1, 2, 3, 4, 5, 6]:
        state.append(environment.get_order_amounts_by_product_type(product_type_element))
    debugging.verify_observation_state(state) # sanity check for the observation state
    return state


def adjust_processing_times(action):
    """
    The input parameter action can have one of three values as seen in the table below. Depending on the
    value of action, we either decrease the processing times of bottleneck machines in the system or keep the default
    settings.
    Note that...
        -> a decrease in processing times means an increase in capacity
        -> an increase in capacity means a higher global_settings.processing_times_multiplier
        -> this only applies to bottleneck machines (machine E in the flow shop, machine C in the job shop)
    :param action:
        Influence processing times depending on the following table.
        Num |   Action
        0   |   Keep capacity (= keep processing times)
        1   |   Increase capacity by 25% (= decrease processing times by 25%)
        2   |   Increase capacity by 50% (= decrease processing times by 50%)
        (the percentage numbers above are just examples, real numbers are inside
        global_settings.overtime_multiplier_1, overtime_multiplier_2 and overtime_multiplier_3)
    :return: nothing gets returned
    """
    if action == 0:  # run on default capacity (equal to 16 working hours per day or three shifts)
        global_settings.processing_times_multiplier = global_settings.overtime_multiplier_1
    elif action == 1:  # run on higher capacity
        global_settings.processing_times_multiplier = global_settings.overtime_multiplier_2
    elif action == 2:  # run on maximum capacity
        global_settings.processing_times_multiplier = global_settings.overtime_multiplier_3
    return


def step_one_step_ahead():
    """
    Steps the simulation forward by one virtual minute
    """
    debugging.verify_all()
    # If end of warmup period is reached, reset all costs
    if global_settings.current_time == global_settings.warmup_duration * global_settings.duration_of_one_period:
        performance_measurement.reset_all_costs()
    # Generate orders
    # if current time == time at which to generate next order, generate order.
    # then update time at which to generate next order
    if global_settings.current_time == global_settings.time_of_next_order_arrival:
        order_generation.generate_order()
    # Move orders between machines and inventories; ship orders
    order_movement.move_orders()
    # Measure bottleneck utilization
    performance_measurement.measure_bottleneck_utilization()
    # Process orders in machines
    order_processing.process_orders()

    global_settings.current_time += 1  # increase simulation step by 1
    return


def step_one_period_ahead():
    """
    Run the simulation for 960 steps (=1 period) and return the results on the end of the period.
    :return:
    * reward -> Integer: total cost from the period
    * environment_state -> Array: state of the environment
    * cost_rundown -> Array: a list indicating which costs occured where
    * done -> Boolean: indicates whether the period has ended (False if it has not yet ended)

    Note that your algorithm should call env.reset() when done is returned as True.
    The environment doesn't reset itself, even if done is returned as True.
    """
    # Reset temporary lists for shipped orders
    global_settings.shipped_orders_by_prodtype_and_lateness = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                                               [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    global_settings.temp_amount_of_shipped_orders = 0
    # Release orders
    order_release.release_orders()
    # Run for 960 steps
    for timestep in range(global_settings.duration_of_one_period):
        step_one_step_ahead()
        order_movement.ship_orders()
    # After each period:
    # Measure cost
    performance_measurement.update_total_cost()
    # Get results
    reward, cost_rundown = get_results_from_this_period()
    environment_state = get_current_environment_state()
    # Check if current episode is done (default: episodes are done after 8000 periods)
    if global_settings.current_time >= (global_settings.number_of_periods * global_settings.duration_of_one_period):
        done = True  # Note that your algorithm should call env.reset() when done is returned as True
        print(bottleneck())
        performance_measurement.evaluate_episode()
    else:
        done = False

    return reward, environment_state, cost_rundown, done


def get_results_from_this_period():
    """
    Get the costs of the current period (it has actually passed already, this functions gets called at the end)
    which will be used as the Gym environment's reward.
    Reinforcement Learning agents try to maximize reward, so we cannot return the actual cost as reward.
    Otherwise, the agents would try to maximize the cost, since internally the cost is a value greater than zero.
    Therefore returned costs are multiplied by -1 to ensure that the returned value is negative.
    :return:
    * total cost: (total cost of the period that just passed) * -1
    * cost_rundown: a cost rundown with more details (list of the costs per production step)
    """
    cost = global_settings.temp_cost_this_period
    cost_rundown = ["WIP: ", global_settings.temp_wip_cost, "FGI: ", global_settings.temp_fgi_cost,
                    "Late:", global_settings.temp_lateness_cost, "Overtime: ", global_settings.temp_overtime_cost]

    if cost < 0:  # sanity check stops the run if cost is negative
        raise ValueError("get_results_from_this_period() received negative costs where they should be positive")
    return cost * -1, cost_rundown


def get_info():
    """
    Return some useful information from the simulation results.
    """
    return (
            "Iteration " + str(global_settings.random_seed) + " finished. Orders shipped: " + str(len(
                environment.shipped_orders)) + " | WIP cost: " + str(
                global_settings.sum_shopfloor_cost) + " | FGI cost: " + str(
                global_settings.sum_fgi_cost) + " | lateness cost: " + str(global_settings.sum_lateness_cost) +
            " | overtime cost: " + str(global_settings.sum_overtime_cost) +
            " | total cost: " + str(global_settings.total_cost) +
            " | Bottleneck utilization: " + str(global_settings.bottleneck_utilization_per_step /
                                                (
                                                        global_settings.duration_of_one_period *
                                                        global_settings.number_of_periods))
    )


def bottleneck():  # used for debugging. todo: delete for final release
    return ("Bottleneck utilization: ", round(
            global_settings.bottleneck_utilization_per_step / global_settings.maximum_simulation_duration,2),
            "| Overtime:",global_settings.processing_times_multiplier,
            )


def get_current_time():
    """
    Used for debugging in env.debug_observation(). While useful, it is not necessary to run the simulation.
    """
    return global_settings.current_time, global_settings.current_time / global_settings.duration_of_one_period


def get_episode_results():
    return performance_measurement.evaluate_episode()


if __name__ == '__main__':
    """
    The code below runs when main.py gets executed. 
    While it's mostly useful for debugging (direct access to everything that happens inside the environment),
    it is recommended to access the simulation over the gym-jobshop environment.
    """
    simulation_start_time = time.time()
    iterations_remaining = global_settings.repetitions

    while iterations_remaining > 0:
        reset()
        reward = 0
        print("Starting simulation. Iteration #" + str(global_settings.random_seed))
        # START SIMULATION MAIN LOOP
        for period in range(global_settings.number_of_periods):
            """
            This is the main loop of the simulation. It increases a global timer with every step of the loop, 
            thus providing a mechanism to track the time of everything that happens inside the simulation. 
            The sequence of actions in the production system is as follows for every step of the simulation:
            - Generate orders (only at certain times) and add them to the order pool
            - Release orders (once every period) from the order pool to the shop floor
            - Process orders inside the machines
            - Move orders between inventories and machines; ship orders
            - Measure the costs (once every period for the current period)
            """
            for i in range(global_settings.duration_of_one_period):
                step_one_step_ahead()
        # END MAIN LOOP

        # ANALYSIS
        print("Iteration " + str(global_settings.random_seed) + " finished. Orders shipped: " + str(len(
            environment.shipped_orders)) +
              " | WIP cost: " + str(global_settings.sum_shopfloor_cost) +
              " | FGI cost: " + str(global_settings.sum_fgi_cost) +
              " | lateness cost: " + str(global_settings.sum_lateness_cost) +
              " | overtime cost: " + str(global_settings.sum_overtime_cost) +
              " | total cost: " + str(global_settings.total_cost))
        print("Bottleneck utilization: " +
              str(global_settings.bottleneck_utilization_per_step /
                  (global_settings.duration_of_one_period * global_settings.number_of_periods)))

        global_settings.random_seed += 1
        iterations_remaining -= 1

    print(str(global_settings.repetitions) + " iterations done. ")
    print("Simulation ran for " + str(round(time.time() - simulation_start_time, 4)) + ' seconds and '
          + str(global_settings.number_of_periods) + " periods per iteration.")
