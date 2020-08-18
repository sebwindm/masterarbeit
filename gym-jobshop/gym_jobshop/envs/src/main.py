# Own module imports
from gym_jobshop.envs.src import environment, order_generation, debugging, csv_handler, order_processing, \
    global_settings, order_release, order_movement, performance_measurement

# Python native module (stdlib) imports
import time, random


def initialize_random_numbers():
    global_settings.random_seed = -1
    return


def reset():
    # these are used at the beginning of the production main loop
    # this function is not to be confused with reset() of the actual Gym environment in jobshop_env.py
    ################################################## INITIAL SETUP & RESET ##################################################
    global_settings.random_seed += 1
    random.seed(global_settings.random_seed)
    global_settings.reset_global_settings()
    performance_measurement.reset_all_costs()
    environment.reset_machines()
    environment.reset_inventories()
    debugging.verify_reset()
    if global_settings.shop_type != "flow_shop":
        return get_current_environment_state()
    else:
        print("States have not been implemented for global_settings.shop_type == flow_shop")


def get_current_environment_state():
    """
    Get environment's state as an array, with each array element representing one of the possible product_types of orders.
    Each array element contains the amounts of orders in each stage of production for the associated product_type.
    The order within the array elements is as follows: Order pool | Work center 1 | Work center 2 | Work center 3 |
    FGI | Shipped goods. For the order pool, FGI and shipped goods, there is not just the total amount of orders
    inside that stage, but a sub-array containing the amounts of orders by earliness/lateness in periods.
    For example, the order pool sub-array contains four elements, the first indicating the amount of orders
    due in 1 period, the 2nd for orders due in 2 periods up to the 4th element which  features orders due in 4
    or more periods. The FGI array contains order amounts sorted by earliness(early by 1 up to 4+ periods) and
    the shipped orders contain the amounts sorted by lateness (in time, late by 1 up to 4+ periods)
    :return: state, an array with 16x6 (= 96) elements

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

    # print("State from main.py: ", state)
    return state


def adjust_processing_times(action):
    """
    The input parameter action can have three values as seen in the table below. Depending on the
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
    :return: nothing gets returned
    """
    if action == 0:  # run on default capacity (equal to 16 working hours or three shifts)
        global_settings.processing_times_multiplier = 1.0
    elif action == 1:  # run on higher capacity (TBD)
        global_settings.processing_times_multiplier = 1.125
    elif action == 2:  # run on maximum capacity (extra working shift, operates 24/7)
        global_settings.processing_times_multiplier = 1.25
    return


def step_one_step_ahead():
    debugging.verify_all()
    # If end of warmup period is reached, reset all costs
    if global_settings.current_time == global_settings.warmup_duration * global_settings.duration_of_one_period:
        performance_measurement.reset_all_costs()
    ################# Generate orders #################
    # if current time == time at which to generate next order, generate order.
    # then update time at which to generate next order
    if global_settings.current_time == global_settings.time_of_next_order_arrival:
        order_generation.generate_order()

    ################# Move orders between machines and inventories; ship orders #################
    order_movement.move_orders()

    ################# Process orders in machines #################
    order_processing.process_orders()

    ################# Measure utilization of machines and wips  #################
    if global_settings.create_steps_csv == True:
        performance_measurement.utilization_per_step()

    ################# Measure incurred costs #################
    if global_settings.current_time % global_settings.duration_of_one_period == 0:
        performance_measurement.update_total_cost()

    # Measure bottleneck utilization
    performance_measurement.measure_bottleneck_utilization()
    ################# Release orders to WIP once every period #################
    if global_settings.current_time % global_settings.duration_of_one_period == 0:
        order_release.release_orders()

    global_settings.current_time += 1  # increase simulation step by 1

    return


def get_results_from_this_period():
    """
    Get the costs of the current period which will be used in the Gym environment's reward mechanism.
    Reinforcement Learning agents try to maximize reward, so we cannot return the actual cost as reward. Otherwise, the
    agents would try to maximize the cost. Therefore returned costs are multiplied by -1.
    :return: (total cost of the simulation iteration) * -1
    """
    cost = global_settings.temp_cost_this_period
    return cost * -1


def get_exponentially_smoothed_reward():
    """
    UNUSED FUNCTION
    TODO: delete this?
    """
    latest_reward = get_results_from_this_period()
    if len(global_settings.past_rewards == 5):
        global_settings.past_rewards.pop([0])
    global_settings.past_rewards.append(latest_reward)

    new_smoothed_reward = 0

    a = 5
    test = [1, 10, 100, 1000, 10000]
    weights = [0.1 * a, 0.2, 0.3, 0.4, 0.5]

    new_result = []

    for i in test:
        new_result.append(i * weights[test.index(i)])

    print(new_result)
    return


def is_episode_done():
    """
    Check if current episode is done (default: episodes are done after 8000 periods)
    :return: boolean
    """
    if global_settings.current_time >= (global_settings.number_of_periods * global_settings.duration_of_one_period):
        done = True
    else:
        done = False
    return done


if __name__ == '__main__':
    csv_handler.initialize_csv_files()

    simulation_start_time = time.time()
    iterations_remaining = global_settings.repetitions

    while iterations_remaining > 0:
        reset()
        reward = 0
        print("Starting simulation. Iteration #" + str(global_settings.random_seed))
        ################################################## START SIMULATION MAIN LOOP ##################################################
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

        ################################################## END MAIN LOOP ##################################################

        ################################################## ANALYSIS ##################################################
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
        # Append simulation results to CSV file
        csv_handler.write_simulation_results()
        # Measure order flow times. This currently supports only 1 iteration,
        # file output may behave unexpectedly for more iterations
        if global_settings.create_orders_csv == True:
            performance_measurement.measure_order_flow_times()

        global_settings.random_seed += 1
        iterations_remaining -= 1

    print(str(global_settings.repetitions) + " iterations done. ")
    print("Simulation ran for " + str(round(time.time() - simulation_start_time, 4)) + ' seconds and '
          + str(global_settings.number_of_periods) + " periods per iteration.")


def get_info():
    return (
            "Iteration " + str(global_settings.random_seed) + " finished. Orders shipped: " + str(len(
        environment.shipped_orders)) + " | WIP cost: " + str(
        global_settings.sum_shopfloor_cost) + " | FGI cost: " + str(
        global_settings.sum_fgi_cost) + " | lateness cost: " + str(global_settings.sum_lateness_cost) +
            " | overtime cost: " + str(global_settings.sum_overtime_cost) +
            " | total cost: " + str(global_settings.total_cost) +
            " | Bottleneck utilization: " + str(global_settings.bottleneck_utilization_per_step /
                                                (
                                                            global_settings.duration_of_one_period * global_settings.number_of_periods))
    )
