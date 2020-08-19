from gym_jobshop.envs.src import class_Machine, global_settings
import random, math, numpy

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
    list_of_inventories = [wip_A, wip_B, wip_C, wip_D, wip_E, wip_F, finished_goods_inventory, shipped_orders,
                           order_pool]
    bottleneck_machine = machine_E

elif global_settings.shop_type == "job_shop":
    machine_A = class_Machine.Machine("Machine A", 30, 130, 80)
    machine_B = class_Machine.Machine("Machine B", 30, 130, 77.5)
    machine_C = class_Machine.Machine("Machine C", 65, 125, 95)
    list_of_all_machines = [machine_A, machine_B, machine_C]
    # Generate WIP (work in process) inventories
    # each WIP inventory is associated with one machine (and each machine with one inventory)
    # when an order arrives at a machine, the order first gets placed inside the WIP inventory
    # if the machine is not processing an order, it pulls one order from the WIP according to certain rules
    wip_A = []
    wip_B = []
    wip_C = []
    list_of_all_wip_elements = [wip_A, wip_B, wip_C]
    list_of_inventories = [wip_A, wip_B, wip_C, finished_goods_inventory, shipped_orders, order_pool]
    bottleneck_machine = machine_C

elif global_settings.shop_type == "job_shop_1_machine":
    machine_A = class_Machine.Machine("Machine A", 30, 130, 106.1999115)
    list_of_all_machines = [machine_A]
    # Generate WIP (work in process) inventories
    # each WIP inventory is associated with one machine (and each machine with one inventory)
    # when an order arrives at a machine, the order first gets placed inside the WIP inventory
    # if the machine is not processing an order, it pulls one order from the WIP according to certain rules
    wip_A = []
    list_of_all_wip_elements = [wip_A]
    list_of_inventories = [wip_A, finished_goods_inventory, shipped_orders, order_pool]
    bottleneck_machine = machine_A
else:
    raise ValueError("Wrong shop_type")


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
                global_settings.next_order_arrival_lower_bound,
                global_settings.next_order_arrival_upper_bound))
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
    :param machine_object: instance/object of class Machine, e.g. machine_A
    :return: nothing
    """
    if global_settings.processing_time_distribution == "uniform":
        machine_object.processing_time = \
            round(random.uniform(
                machine_object.uniform_processing_time_lower_bound,
                machine_object.uniform_processing_time_upper_bound))
    elif global_settings.processing_time_distribution == "exponential":
        machine_object.processing_time = \
            get_random_exponential_number(machine_object.exponential_processing_time)
    else:
        raise ValueError("global_settings.processing_time_distribution invalid value assigned. "
                         "Must be 'exponential' or 'uniform'")
    return


def get_order_amounts_by_product_type(product_type):
    """
    Retrieve the amounts of orders in each stage of the production process that are of a given product_type.
    The stages are Order pool | Work center 1 | Work center 2 | Work center 3 | FGI | Shipped goods
    :param product_type: Integer in range (1,6), this is the product_type from Class_Order.py
    :return: returns an array with six elements, each indicating the order amounts at the production steps
    """
    if product_type not in [1, 2, 3, 4, 5, 6]:
        raise ValueError("Wrong product type in environment.py -> get_order_amounts_by_product_type")

    ### Calculate amount of orders inside the order pool
    amount_in_order_pool = get_order_pool_levels_by_due_date(product_type)

    ### Calculate amount of orders inside the work centers
    amount_in_work_center_1 = 0
    for order_element in wip_A:
        if order_element.product_type == product_type:
            amount_in_work_center_1 += 1
    for order_element in machine_A.orders_inside_the_machine:
        if order_element.product_type == product_type:
            amount_in_work_center_1 += 1
    # the code hereafter only gets executed if there is more than 1 machine,
    # that is in all shop types except for "job_shop_1_machine"
    if global_settings.shop_type == "job_shop":
        amount_in_work_center_2 = 0
        for order_element in wip_B:
            if order_element.product_type == product_type:
                amount_in_work_center_2 += 1
        for order_element in machine_B.orders_inside_the_machine:
            if order_element.product_type == product_type:
                amount_in_work_center_2 += 1

        amount_in_work_center_3 = 0
        for order_element in wip_C:
            if order_element.product_type == product_type:
                amount_in_work_center_3 += 1
        for order_element in machine_C.orders_inside_the_machine:
            if order_element.product_type == product_type:
                amount_in_work_center_3 += 1
    # As a workaround, we return 0 for the states of the nonexistent machines
    if global_settings.shop_type == "job_shop_1_machine":
        amount_in_work_center_2 = 0
        amount_in_work_center_3 = 0

    ### Calculate amount of orders inside the FGI
    amount_in_fgi = get_fgi_levels_by_earliness(product_type)

    ### Calculate the amount of orders that were shipped in this period
    amount_in_shipped_goods = get_shipped_levels_by_lateness(product_type)

    ### Aggregate all amounts to one list
    order_amounts = []
    for i in amount_in_order_pool:
        order_amounts.append(i)
    order_amounts.append(amount_in_work_center_1)
    order_amounts.append(amount_in_work_center_2)
    order_amounts.append(amount_in_work_center_3)
    for i in amount_in_fgi:
        order_amounts.append(i)
    for i in amount_in_shipped_goods:
        order_amounts.append(i)
    return order_amounts


def reset_machines():
    for machine in list_of_all_machines:
        machine.orders_inside_the_machine.clear()
    set_next_order_arrival_time()
    return


def reset_inventories():
    for inventory_element in list_of_inventories:
        inventory_element.clear()


def get_order_pool_levels_by_due_date(product_type):
    """
    :param product_type: product type of orders (class Order)
    :return: a list with the amount of orders of the requested product_type sorted by due date periods
    Example: [5 3 3 1] -> the first list item refers to the amount of orders inside the order pool
    that are due in 1 period, the second item for orders due in 2 periods, the third... and the fourth item
    refers to all orders that are due in 4 or more periods
    """
    due_in_1_period = 0
    due_in_2_periods = 0
    due_in_3_periods = 0
    due_in_4_periods = 0
    due_in_5_periods = 0
    due_in_6_periods = 0
    due_in_7_periods = 0
    due_in_8_periods = 0
    due_in_9_periods = 0
    due_in_10_periods = 0
    order_pool_levels_by_due_date = []

    for order_element in order_pool:
        if order_element.product_type == product_type:
            due_in = (order_element.due_date - global_settings.current_time) / global_settings.duration_of_one_period
            if due_in <= 1: due_in_1_period += 1
            if 1 < due_in <= 2: due_in_2_periods += 1
            if 2 < due_in <= 3: due_in_3_periods += 1
            if 3 < due_in <= 4: due_in_4_periods += 1
            if 4 < due_in <= 5: due_in_5_periods += 1
            if 5 < due_in <= 6: due_in_6_periods += 1
            if 6 < due_in <= 7: due_in_7_periods += 1
            if 7 < due_in <= 8: due_in_8_periods += 1
            if 8 < due_in <= 9: due_in_9_periods += 1
            if due_in > 9: due_in_10_periods += 1

    order_pool_levels_by_due_date.extend((due_in_1_period, due_in_2_periods,
                                          due_in_3_periods, due_in_4_periods, due_in_5_periods, due_in_6_periods,
                                          due_in_7_periods, due_in_8_periods, due_in_9_periods, due_in_10_periods))
    #print("ptype ",product_type,": order pool by due date: ",order_pool_levels_by_due_date)
    return order_pool_levels_by_due_date


def get_fgi_levels_by_earliness(product_type):
    early_by_1_period = 0
    early_by_2_periods = 0
    early_by_3_periods = 0
    early_by_4_or_more_periods = 0
    fgi_levels_by_earliness = []

    for order_element in finished_goods_inventory:
        if order_element.product_type == product_type:
            early_by = (order_element.due_date - global_settings.current_time) / \
                       global_settings.duration_of_one_period
            if early_by <= 1: early_by_1_period += 1
            if early_by > 1 and early_by <= 2: early_by_2_periods += 1
            if early_by > 2 and early_by <= 3: early_by_3_periods += 1
            if early_by > 3: early_by_4_or_more_periods += 1

    fgi_levels_by_earliness.extend((early_by_1_period, early_by_2_periods,
                                    early_by_3_periods, early_by_4_or_more_periods))
    # print("ptype ",product_type,": fgi earliness: ",fgi_levels_by_earliness)

    return fgi_levels_by_earliness


def get_shipped_levels_by_lateness(product_type):
    shipped_levels_by_lateness = global_settings.shipped_orders_by_prodtype_and_lateness[int(product_type) - 1]

    # print("ptype ",product_type,": shipped lateness: ",shipped_levels_by_lateness)

    return shipped_levels_by_lateness
