from gym_jobshop.envs.src import class_Machine, global_settings
import random, math

random.seed(global_settings.random_seed)

# This is the inventory where finished goods get placed and wait until their due_date is reached, it is also named FGI.
# If an order finishes after its due_date, the order gets shipped right away and doesn't wait in the FGI
finished_goods_inventory = []

# All orders that are finished get moved here once they have reached their due dates.
shipped_orders = []

# This is the order pool where we store customer orders which have not been released to production yet.
# From this order pool, we send the orders to the first stage of production (e.g. to WIP_A and so on)
order_pool = []

# instantiate machine objects with name and processing times
if global_settings.shop_type == "flow_shop":
    machine_A = class_Machine.Machine("Machine A", 30, 130, 80)
    machine_B = class_Machine.Machine("Machine B", 80, 240, 160)
    machine_C = class_Machine.Machine("Machine C", 50, 260, 155)
    machine_D = class_Machine.Machine("Machine D", 50, 370, 210)
    machine_E = class_Machine.Machine("Machine E", 200, 370, 285)
    machine_F = class_Machine.Machine("Machine F", 110, 320, 215)
    list_of_all_machines = [machine_A, machine_B, machine_C, machine_D, machine_E, machine_F]
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

elif global_settings.shop_type == "job_shop":
    machine_A = class_Machine.Machine("Machine A", 30, 130, 80)
    machine_B = class_Machine.Machine("Machine B", 30, 130, 77.5)
    machine_C = class_Machine.Machine("Machine C", 65, 125, 95)
    machine_D = None
    machine_E = None
    machine_F = None
    list_of_all_machines = [machine_A, machine_B, machine_C]
    # Generate WIP (work in process) inventories
    # each WIP inventory is associated with one machine (and each machine with one inventory)
    # when an order arrives at a machine, the order first gets placed inside the WIP inventory
    # if the machine is not processing an order, it pulls one order from the WIP according to certain rules
    wip_A = []
    wip_B = []
    wip_C = []
    wip_D = None
    wip_E = None
    wip_F = None
    list_of_all_wip_elements = [wip_A, wip_B, wip_C]
    list_of_inventories = [wip_A, wip_B, wip_C, finished_goods_inventory, shipped_orders, order_pool]

else: raise ValueError("Wrong shop_type")



def get_random_exponential_number(rate_parameter_lambda):
    """
    Take number as input and return an exponentially distributed random number based on input.
    :param rate_parameter_lambda: Rate parameter of the exponential distribution, aka λ (lambda). Must be INTEGER
    :return: an exponentially distributed random number based on λ
    This function is used because the Python built-in Random module has no similar function.
    The random.expovariate function should not be used, as it returns the inverse scale of the input parameter
    and thus returns unusable results. The numpy.random.exponential function behaves correctly and
    similar to the here presented function and can be used as a substitute.
    """
    result = round(-rate_parameter_lambda * math.log(1.0 - random.random()))
    if result == 0:  # result must not be 0 in this simulation
        result = 1
    return result


def set_next_order_arrival_time():
    """
    Set the time (in steps) at which the next order should arrive to the order pool.
    Depends on global_settings.py -> demand_distribution(String) which can be either exponential or uniform
    :return: returns nothing
    """
    if global_settings.demand_distribution == "uniform":
        # Update the time_of_next_order_arrival to a random time within the interval specified in global_settings
        global_settings.time_of_next_order_arrival = \
            global_settings.current_time + round(random.uniform(
                global_settings.next_order_arrival_lower_bound * global_settings.granularity_multiplier,
                global_settings.next_order_arrival_upper_bound * global_settings.granularity_multiplier))
    elif global_settings.demand_distribution == "exponential":
        # Update the time_of_next_order_arrival to a random time from an exponential distribution
        random_exponential_number = \
            get_random_exponential_number(global_settings.next_order_arrival_exponential_rate_parameter)
        if random_exponential_number == 0:
            random_exponential_number = 1
        global_settings.time_of_next_order_arrival = global_settings.current_time + random_exponential_number
        # print("time to next order:" + str(random_exponential_number))
    else:
        raise ValueError("global_settings.demand_distribution invalid value assigned. "
                         "Must be 'exponential' or 'uniform'")
    return


def set_new_random_processing_time(machine_object):
    """
    Set processing time for the desired machine to a random value between the lower bound and the upper bound
    (influenced by the global granularity multiplier)
    :param machine_object: instance/object of class Machine, e.g. machine_A
    :return: nothing
    """
    if global_settings.processing_time_distribution == "uniform":
        machine_object.processing_time = \
            round(random.uniform(
                machine_object.uniform_processing_time_lower_bound * global_settings.granularity_multiplier,
                machine_object.uniform_processing_time_upper_bound * global_settings.granularity_multiplier))
    elif global_settings.processing_time_distribution == "exponential":
        machine_object.processing_time = \
            get_random_exponential_number(machine_object.exponential_processing_time) \
            * global_settings.granularity_multiplier
    else:
        raise ValueError("global_settings.processing_time_distribution invalid value assigned. "
                         "Must be 'exponential' or 'uniform'")
    return





def reset_machines():
    for machine in list_of_all_machines:
        machine.orders_inside_the_machine.clear()
    set_next_order_arrival_time()
    return


def reset_inventories():
    for inventory_element in list_of_inventories:
        inventory_element.clear()
