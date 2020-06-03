# Own module imports
from gym_jobshop.envs.src import environment, order_generation, debugging, csv_handler, order_processing, \
    global_settings, order_release, order_movement, performance_measurement

# Python native module imports
import time, random



def reset():
    # these are used at the beginning of the production main loop
    ################################################## INITIAL SETUP & RESET ##################################################
    random.seed(global_settings.random_seed)
    global_settings.reset_global_settings()
    performance_measurement.reset_all_costs()
    environment.reset_machines()
    environment.reset_inventories()
    debugging.verify_reset()
    return

def adjust_processing_times(action):
    """
    The input parameter action can have three values as seen in the table below. Depending on the
    value of action, we either keep the default or decrease the processing times of all machines in the system.
    Note that...
        -> a decrease in processing times means an increase in capacity
        -> an increase in capacity means a higher global_settings.processing_times_multiplier
    For the default value, we assume the production to run for two shifts per day, that is 16 hours (or 960 minutes)
    :param action:
        Influence processing times depending on the following table.
        Num |   Action
        0   |   Keep capacity (= keep processing times)
        1   |   Increase capacity by 25% (= decrease processing times by 25%)
        2   |   Increase capacity by 50% (= decrease processing times by 50%)
    :return: nothing gets returned
    """
    if action == 0: # run on default capacity (equal to 16 working hours or three shifts)
        global_settings.processing_times_multiplier = 1.0
    elif action == 1: # run on higher capacity (TBD)
        global_settings.processing_times_multiplier = 1.25
    elif action == 2: # run on maximum capacity (extra working shift, operates 24/7)
        global_settings.processing_times_multiplier = 1.5
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

    ################# Measure incurred costs #################
    if global_settings.current_time % global_settings.duration_of_one_period == 0:
        performance_measurement.measure_cost()

    ################# Measure utilization of machines and wips  #################
    if global_settings.create_steps_csv == True:
        performance_measurement.utilization_per_step()

    ################# Release orders to WIP once every period #################
    if global_settings.current_time % global_settings.duration_of_one_period == 0:
        order_release.release_orders()

    # runtime info
    # if global_settings.current_time % 960000 == 0 and global_settings.current_time != 0:
    #     print("Don't worry, still running. 1000 periods have passed since the last message.")
    global_settings.current_time += 1  # increase simulation step by 1

    return

def get_results():
    return [global_settings.total_cost, len(environment.shipped_orders)]

if __name__ == '__main__':
    csv_handler.initialize_csv_files()

    simulation_start_time = time.time()
    iterations_remaining = global_settings.repetitions

    while iterations_remaining > 0:
        reset()

        print("Starting simulation. Iteration #" + str(global_settings.random_seed))
        ################################################## START SIMULATION MAIN LOOP ##################################################
        while global_settings.current_time < global_settings.maximum_simulation_duration:
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
            step_one_step_ahead()
        ################################################## END MAIN LOOP ##################################################

        ################################################## ANALYSIS ##################################################
        print("Iteration " + str(global_settings.random_seed) + " finished. Orders shipped: " + str(len(
            environment.shipped_orders)) +
              " | WIP cost: " + str(global_settings.sum_shopfloor_cost) +
              " | FGI cost: " + str(global_settings.sum_fgi_cost) +
              " | lateness cost: " + str(global_settings.sum_lateness_cost) +
              " | total cost: " + str(global_settings.total_cost))

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


