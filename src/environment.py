from src import global_settings, class_Machine
import random

random.seed(global_settings.random_seed)
# instantiate machine object with name and processing time per order
machine_A = class_Machine.Machine("Machine A", 30, 130)
machine_B = class_Machine.Machine("Machine B", 80, 240)
machine_C = class_Machine.Machine("Machine C", 50, 260)
machine_D = class_Machine.Machine("Machine D", 50, 370)
machine_E = class_Machine.Machine("Machine E", 200, 370)
machine_F = class_Machine.Machine("Machine F", 110, 320)
list_of_all_machines = [machine_A, machine_B, machine_C, machine_D, machine_E, machine_F]


def set_next_order_arrival_time():
    # Update the time_of_next_order_arrival to a random time within the interval specified in global_settings
    global_settings.time_of_next_order_arrival = \
        global_settings.current_time + round(random.uniform(
            global_settings.next_order_arrival_lower_bound * global_settings.granularity_multiplier,
            global_settings.next_order_arrival_upper_bound * global_settings.granularity_multiplier))
    return


def set_new_random_processing_time(machine_object):
    """
    Set processing time for the desired machine to a random value between the lower bound and the upper bound
    (influenced by the global granularity multiplier)
    :param machine_object: instance/object of class Machine, e.g. machine_A
    :return: nothing
    """
    machine_object.processing_time = \
        round(random.uniform(machine_object.processing_time_lower_bound * global_settings.granularity_multiplier,
                             machine_object.processing_time_upper_bound * global_settings.granularity_multiplier))
    return



# This is the inventory where finished goods get placed and wait until their due_date is reached, it is also named FGI.
# If an order finishes after its due_date, the order gets shipped right away and doesn't wait in the FGI
finished_goods_inventory = []

# All orders that are finished get moved here once they have reached their due dates.
shipped_orders = []

# This is the order pool where we store customer orders which have not been released to production yet.
# From this order pool, we send the orders to the first stage of production (e.g. to WIP_A and so on)
order_pool = []

# Generate WIP (work in process) inventories
# each WIP inventory is associated with one machine (and each machine with one inventory)
# when an order arrives at a machine, the order first gets placed inside the WIP inventory
# if the machine is not processing an order, it pulls one order from the WIP according to certain rules
wip_A = []
wip_B = []
wip_C = []
wip_D = []
wip_E = []
wip_F = []
list_of_all_wip_elements = [wip_A, wip_B, wip_C, wip_D, wip_E, wip_F]
list_of_inventories = [wip_A, wip_B, wip_C, wip_D, wip_E, wip_F, finished_goods_inventory, shipped_orders, order_pool]

def reset_machines():
    for machine in list_of_all_machines:
        machine.orders_inside_the_machine.clear()
    set_next_order_arrival_time()
    return


def reset_inventories():
    for inventory_element in list_of_inventories:
        inventory_element.clear()



