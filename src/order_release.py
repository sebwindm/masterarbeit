# Orders are released to the production floor/shop floor periodically
# Every step of the simulation, the order release mechanism attempts to release orders according to certain rules
# The release mechanism might release any number of orders from the order pool, or none
from src import global_settings, environment


# CURRENTLY NOT IN USE
def sort_order_pool_by_due_date():
    print("order pool element 0 due date:" + str(environment.order_pool[0].due_date))
    environment.order_pool.sort(key=lambda x: x.due_date, reverse=False)
    print("order pool sorted by due date")
    print("order pool element 0 due date:" + str(environment.order_pool[0].due_date))
    return


# Periodic release policy:
def release_using_periodic_release():
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
        # iterate over order pool and
        temporary_list = environment.order_pool.copy()
        temp_number_of_released_orders = 0
        for order_element in temporary_list:
            order_element.order_release_date = global_settings.current_time
            order_element.current_production_step = 0
            if order_element.product_type in (1,2):
                environment.wip_A.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
            elif order_element.product_type in (3,4):
                environment.wip_B.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
            elif order_element.product_type in (5,6):
                environment.wip_C.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
            temp_number_of_released_orders += 1

    # debug info:
    if temp_number_of_released_orders > 0 and global_settings.show_order_release == True:
        print("Step " + str(global_settings.current_time) + ": " + str(temp_number_of_released_orders) + " orders released. Orders in pool: " + str(len(
            environment.order_pool)))
    return


# LUMS policy
# Hybrid rule based order release model, e.g. LUMS-COR after Thürer et al. 2012:
# release periodically and pull a job forward from the order pool if work center is (about to) starve
def release_using_lums():
    # Periodic release:

    return

def release_using_bil():
    #planned_release_date
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
        # iterate over order pool and
        temporary_list = environment.order_pool.copy()
        temp_number_of_released_orders = 0
        for order_element in temporary_list:
            if order_element.planned_release_date <= global_settings.current_time:
                order_element.order_release_date = global_settings.current_time
                order_element.current_production_step = 0
                if order_element.product_type in (1,2):
                    environment.wip_A.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
                elif order_element.product_type in (3,4):
                    environment.wip_B.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
                elif order_element.product_type in (5,6):
                    environment.wip_C.append(environment.order_pool.pop(environment.order_pool.index(order_element)))
                temp_number_of_released_orders += 1

    # debug info:
    if temp_number_of_released_orders > 0 and global_settings.show_order_release == True:
        print("Step " + str(global_settings.current_time) + ": " + str(temp_number_of_released_orders) + " orders released. Orders in pool: " + str(len(
            environment.order_pool)))

    return


def release_orders():
    # sort_order_pool_by_due_date()
    if global_settings.order_release_policy == "periodic":
        release_using_periodic_release()
    elif global_settings.order_release_policy == "bil":
        release_using_bil()
    elif global_settings.order_release_policy == "lums":
        release_using_lums() # NOT YET IMPLEMENTED
    return
