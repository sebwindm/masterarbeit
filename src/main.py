# Own module imports
from src import order_generation, global_settings, order_movement, order_release, debugging, environment, \
    order_processing, performance_measurement

# Python native module imports
import time
simulation_start_time = time.time()
import random
random.seed(global_settings.random_seed)
################################################## SETUP ##################################################
# Setup the simulation environment
environment.set_new_processing_times()
environment.setup_order_pool(global_settings.amount_of_orders_to_generate_initially) # generate x initial orders
print("Initial orders generated. Orders in pool: " + str(len(environment.order_pool)))

print("Starting simulation.")
################################################## START SIMULATION ##################################################
while global_settings.current_time < global_settings.maximum_simulation_duration:
    debugging.verify_all()

    # if current time == time at which to generate next order, generate order.
    # then update time at which to generate next order
    if global_settings.current_time == global_settings.time_of_next_order_arrival:
        order_generation.generate_order()

    # Release orders to WIP once every period
    if global_settings.current_time % global_settings.duration_of_one_period == 0:
        order_release.release_orders()
    # Move orders between machines and inventories
    order_movement.move_orders()
    # Process orders in machines
    order_processing.process_orders()

    # Measure incurred costs
    performance_measurement.measure_cost()

    if global_settings.current_time % 960000 == 0 and global_settings.current_time != 0:
        print("Don't worry, still running. 1000 periods have passed since the last message.")
    global_settings.current_time += 1 # increase simulation step by 1
################################################## END MAIN LOOP ##################################################



################################################## ANALYSIS ##################################################
print("                                                                                                       ")
print("Simulation ran for " + str(global_settings.maximum_simulation_duration) + " steps and " + str(round(time.time() - simulation_start_time, 4)) + ' seconds')

print("Orders in FGI: " + str(len(environment.finished_goods_inventory)))
print("Orders in shipped_orders: " + str(len(environment.shipped_orders)))
print("WIP cost: " + str(global_settings.sum_wip_cost) +
      " | FGI cost: " + str(global_settings.sum_fgi_cost) +
      " | lateness cost: " + str(global_settings.sum_lateness_cost) +
      " | total cost: " + str(global_settings.total_cost))



