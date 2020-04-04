from src import environment


def process_orders():
    # Processing works as follows:
    # If there is an order inside the machine, reduce its remaining processing time by 1
    # When the order's remaining processing time hits 0, the logic from order_movement.py comes into play
    # and moves the order out of the machine into the next inventory (e.g. WPI or FGI)
    list_of_machines = [environment.machine_A, environment.machine_B, environment.machine_C, environment.machine_D,
                        environment.machine_E, environment.machine_F]
    for machine in list_of_machines:
        if len(machine.orders_inside_the_machine) == 1:
            machine.orders_inside_the_machine[0].processing_time_remaining -= 1
    return
