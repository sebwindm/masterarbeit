# Orders are released to the production floor/shop floor periodically
# Every step of the simulation, the order release mechanism attempts to release orders according to certain rules
# The release mechanism might release any number of orders from the order pool, or none
from gym_jobshop.envs.src import environment, global_settings


def sort_order_pool_by_due_date():
    """
    CURRENTLY NOT IN USE
    """
    print("order pool element 0 due date:" + str(environment.order_pool[0].due_date))
    environment.order_pool.sort(key=lambda x: x.due_date, reverse=False)
    print("order pool sorted by due date")
    print("order pool element 0 due date:" + str(environment.order_pool[0].due_date))
    return


def release_using_periodic_release():
    """
    Periodic release policy:
    """
    # Flow shop logic: each element in the order_pool gets moved to wip_A
    if global_settings.shop_type == "flow_shop":
        temporary_list = environment.order_pool.copy()
        temp_number_of_released_orders = 0
        for order_element in temporary_list:
            order_element.order_release_date = global_settings.current_time
            environment.wip_A.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
            temp_number_of_released_orders += 1
    # Job shop logic: each element in the order_pool gets moved to its respective WIP
    if global_settings.shop_type == "job_shop":
        # iterate over order pool and release to respective inventory depending on product type
        temporary_list = environment.order_pool.copy()
        temp_number_of_released_orders = 0
        for order_element in temporary_list:
            order_element.order_release_date = global_settings.current_time
            order_element.current_production_step = 0
            if order_element.product_type in (1, 2):
                environment.wip_A.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
            elif order_element.product_type in (3, 4):
                environment.wip_B.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
            elif order_element.product_type in (5, 6):
                environment.wip_C.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
            temp_number_of_released_orders += 1
    # Same as job shop logic, but for 1 machine only
    if global_settings.shop_type == "job_shop_1_machine":
        # iterate over order pool and release to respective inventory depending on product type
        temporary_list = environment.order_pool.copy()
        temp_number_of_released_orders = 0
        for order_element in temporary_list:
            order_element.order_release_date = global_settings.current_time
            order_element.current_production_step = 0
            environment.wip_A.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
            temp_number_of_released_orders += 1
    return


def release_using_bil():
    """
    Release order using the BIL (backward infinite loading) policy
    Each generated order gets a planned release date, at which it will be released to production.
    The planned release date is defined by global_settings.planned_release_date_multiplier.
    Planned release date = (multiplier * duration of a period in minutes) + current time in minutes.
    As soon as the current time equals the planned release date, the order gets released.
    """
    # Flow shop logic: each element in the order_pool gets moved to wip_A
    if global_settings.shop_type == "flow_shop":
        temporary_list = environment.order_pool.copy()
        temp_number_of_released_orders = 0
        for order_element in temporary_list:
            if order_element.planned_release_date <= global_settings.current_time:
                order_element.order_release_date = global_settings.current_time
                environment.wip_A.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
                temp_number_of_released_orders += 1
    # Job shop logic: each element in the order_pool gets moved to its respective WIP
    if global_settings.shop_type == "job_shop":
        # iterate over order pool and release to respective inventory depending on product type
        temporary_list = environment.order_pool.copy()
        temp_number_of_released_orders = 0
        for order_element in temporary_list:
            if order_element.planned_release_date <= global_settings.current_time:
                order_element.order_release_date = global_settings.current_time
                order_element.current_production_step = 0
                if order_element.product_type in (1, 2):
                    environment.wip_A.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
                elif order_element.product_type in (3, 4):
                    environment.wip_B.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
                elif order_element.product_type in (5, 6):
                    environment.wip_C.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
                temp_number_of_released_orders += 1
    # Same as job shop logic, but for 1 machine only
    if global_settings.shop_type == "job_shop_1_machine":
        # iterate over order pool and release to respective inventory depending on product type
        temporary_list = environment.order_pool.copy()
        temp_number_of_released_orders = 0
        for order_element in temporary_list:
            if order_element.planned_release_date <= global_settings.current_time:
                order_element.order_release_date = global_settings.current_time
                order_element.current_production_step = 0
                environment.wip_A.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
                temp_number_of_released_orders += 1
    return


def release_orders():
    if global_settings.order_release_policy == "periodic":
        release_using_periodic_release()
    elif global_settings.order_release_policy == "bil":
        release_using_bil()
    return
