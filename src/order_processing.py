from src import environment, global_settings


def process_orders():
    # Processing works as follows:
    # If there is an order inside the machine, reduce its remaining processing time by 1
    # When the order's remaining processing time hits 0, the logic from order_movement.py comes into play
    # and moves the order out of the machine into the next inventory (e.g. WPI or FGI)
    for machine in environment.list_of_all_machines:
        if len(machine.orders_inside_the_machine) == 1:
            machine.orders_inside_the_machine[0].processing_time_remaining -= (1 * global_settings.granularity_multiplier)
    return
