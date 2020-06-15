from gym_jobshop.envs.src import environment, global_settings


def process_orders():
    # Processing works as follows:
    # If there is an order inside the machine,
    # reduce its remaining processing time by 1 * global_settings.processing_times_multiplier
    #
    # For more information on global_settings.processing_times_multiplier, see the documentation inside global_settings
    # When the order's remaining processing time hits 0, the logic from order_movement.py comes into play
    # and moves the order out of the machine into the next inventory (e.g. WPI or FGI)
    # Additionally, the bottleneck machines (machine E in the flow shop, machine C in the job shop)
    # can work overtime (= capacity increase), which is represented by a reduction of processing times.
    # Example: normal machines reduce 1 remaining processing time of an order per step. A bottleneck machine
    # on overtime reduces 1.5 remaining processing time of the order in each step, thereby reducing the overall
    # processing time of the order by 50%. This of course causes higher operating costs for the machine,
    # which are taken into account in performance_measurement.py
    for machine in environment.list_of_all_machines:
        if len(machine.orders_inside_the_machine) == 1:
            if machine.name == environment.bottleneck_machine.name:
                machine.orders_inside_the_machine[0].processing_time_remaining -= global_settings.processing_times_multiplier
            else:
                machine.orders_inside_the_machine[0].processing_time_remaining -= 1
    return
