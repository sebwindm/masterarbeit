from src import global_settings, environment


def verify_machines():
    # Raise error if number of orders in any machine exceeds 1
    list_of_machines = [environment.machine_A, environment.machine_B, environment.machine_C, environment.machine_D,
                        environment.machine_E, environment.machine_F]
    list_of_allowed_product_types = [
        [1, 2, 3, 4, 5, 6], [1, 2, 3], [4, 5, 6], [1, 4], [2, 5], [3, 6]
    ]
    for machine in list_of_machines:
        if len(machine.orders_inside_the_machine) > 1:
            raise ValueError("Too many orders inside machine " + str(machine.name))
        elif len(machine.orders_inside_the_machine) == 1:
            if machine.orders_inside_the_machine[0].product_type not in (
                    list_of_allowed_product_types[list_of_machines.index(machine)]):
                raise ValueError("step " + str(global_settings.current_time) +
                                 " Wrong product type in machine " + str(machine.name) +
                                 " || product type " + str(machine.orders_inside_the_machine[0].product_type))

    return


def verify_wips():  # THIS FUNCTION MIGHT BE UNNECESSARY (checking the machines is enough)
    # Raise error if a machine contains the wrong product type
    for order_element in environment.wip_B:
        if order_element.product_type in (4, 5, 6):
            raise ValueError("Wrong product type in wip ")
    for order_element in environment.wip_C:
        if order_element.product_type in (1, 2, 3):
            raise ValueError("Wrong product type in wip ")
    for order_element in environment.wip_D:
        if order_element.product_type in (2, 3, 5, 6):
            raise ValueError("Wrong product type in wip ")
    for order_element in environment.wip_E:
        if order_element.product_type in (1, 3, 4, 6):
            raise ValueError("Wrong product type in wip ")
    for order_element in environment.wip_F:
        if order_element.product_type in (1, 2, 4, 5):
            raise ValueError("Wrong product type in wip ")
    return


def verify_policies():
    # Raise error if order release policy has been entered incorrectly
    if global_settings.order_release_policy != "immediate_release":
        raise ValueError("There was a problem with the selected order release policy. "
                         "Please review order_release_policy at global_settings.py ")
    # Raise error if scheduling policy has been entered incorrectly
    if global_settings.scheduling_policy != "first_come_first_serve":
        raise ValueError("There was a problem with the selected scheduling policy. "
                         "Please review scheduling_policy at global_settings.py ")
    return

    # This runs over every order in the finished goods inventory to see if there are orders left
    # which should have been removed long ago. WARNING: takes a lot of resources, only run at the very end


def verify_fgi():
    # Raise error if there are orders in the FGI (finished goods inventory) even though they shouldn't be there
    if len(environment.finished_goods_inventory) > 0:
        for order_element in environment.finished_goods_inventory:
            if order_element.due_date < global_settings.current_time:
                raise ValueError("Programming error in finished goods inventory: "
                                 "overdue order is still inside the inventory. ")
    return


def verify_all():
    # The following checks are performed every 50 steps of the simulation
    if global_settings.current_time % 50 == 0:
        verify_machines()
    # verify_wips() --> we don't need to verify wips, as a wrong product type would be noticed in verify_machines
    # The following checks are only performed in the first step of the simulation
    if global_settings.current_time == 0:
        verify_policies()
    # The following checks are only performed in the last step of the simulation
    if global_settings.current_time == global_settings.maximum_simulation_duration - 1:
        verify_fgi()
