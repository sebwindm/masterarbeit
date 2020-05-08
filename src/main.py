# Own module imports
from src import order_generation, global_settings, order_movement, order_release, debugging, environment, \
    order_processing, performance_measurement

# Python native module imports
import time, random, csv, datetime
# Generate prefix for CSVs
if global_settings.processing_time_distribution == "uniform":
    processing = "UB"
else: processing = "EB"
if global_settings.demand_distribution == "uniform":
    demand = "UD"
else: demand = "ED"
if global_settings.shop_type == "flow_shop":
    shop = "FS"
else: shop = "JS"
csv_prefix = str(datetime.datetime.now().strftime("%d.%m.%Y")) + "_" + shop + "_" + processing + "_" + demand

# Create CSV file to store results after each iteration
with open(str('../' + csv_prefix) + '_simulation_results.csv', mode='w') as results_CSV:
    results_writer = csv.writer(results_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Iteration','Runtime(steps)', 'Orders shipped', 'WIP cost', 'FGI cost', 'Lateness cost', 'Total cost'])
    results_CSV.close()
# Create CSV file to track utilization of wips/machines each step
if global_settings.create_steps_csv == True:
    with open('../steps.csv', mode='w') as steps_CSV:
        results_writer = csv.writer(steps_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(['Step', 'Orders in WIP_1','Orders in WIP_2','Orders in WIP_3',
                                     'Orders in WIP_4','Orders in WIP_5','Orders in WIP_6',
                                     'util_m1', 'util_m2','util_m3','util_m4',
                                     'util_m5','util_m6','Utilization of all machines',])
        steps_CSV.close()

simulation_start_time = time.time()
iterations_remaining = global_settings.repetitions
while iterations_remaining > 0:
    random.seed(global_settings.random_seed)
    ################################################## SETUP ##################################################
    # Setup the simulation environment
    global_settings.reset_global_settings()
    performance_measurement.reset_all_costs()
    environment.reset_machines()
    environment.reset_inventories()
    debugging.verify_reset()


    print("Starting simulation. Iteration #" + str(global_settings.random_seed))
    ################################################## START SIMULATION ##################################################
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
        debugging.verify_all()
        # If end of warmup period is reached, reset all costs
        if global_settings.current_time == global_settings.warmup_duration * global_settings.duration_of_one_period * global_settings.granularity_multiplier:
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
        global_settings.current_time += (1  * global_settings.granularity_multiplier) # increase simulation step by 1
    ################################################## END MAIN LOOP ##################################################



    ################################################## ANALYSIS ##################################################
    print("Iteration " + str(global_settings.random_seed) + " finished. Orders shipped: " + str(len(environment.shipped_orders)) +
        " | WIP cost: " + str(global_settings.sum_shopfloor_cost) +
          " | FGI cost: " + str(global_settings.sum_fgi_cost) +
          " | lateness cost: " + str(global_settings.sum_lateness_cost) +
          " | total cost: " + str(global_settings.total_cost))


    # Append results to CSV file
    with open(str('../' + csv_prefix) + '_simulation_results.csv', mode='a') as results_CSV:
        results_writer = csv.writer(results_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow([global_settings.random_seed, global_settings.maximum_simulation_duration,
                                 len(environment.shipped_orders), global_settings.sum_shopfloor_cost,
                                 global_settings.sum_fgi_cost, global_settings.sum_lateness_cost,
                                 global_settings.total_cost])
        results_CSV.close()
    if global_settings.create_orders_csv == True:
        performance_measurement.measure_order_flow_times()
    global_settings.random_seed += 1
    iterations_remaining -= 1
    print("                                                                                                       ")





print(str(global_settings.repetitions) + " iterations done. ")
print("Simulation ran for " + str(round(time.time() - simulation_start_time, 4)) + ' seconds and '
      + str(global_settings.number_of_periods) + " periods per iteration.")

