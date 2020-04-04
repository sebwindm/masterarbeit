    # # Move order from machine_C to WIP D/E/F, if processing_time_remaining of order is 0
    # if len(environment.machine_C.orders_inside_the_machine) == 1:
    #     if environment.machine_C.orders_inside_the_machine[0].processing_time_remaining == 0:
    #         if environment.machine_C.orders_inside_the_machine[0].product_type == (4):
    #             environment.wip_D.append(environment.machine_C.orders_inside_the_machine.pop(0))
    #             print("Step " + str(global_settings.current_time) + ": Machine_C: order finished")
    #         elif environment.machine_C.orders_inside_the_machine[0].product_type == (5):
    #             environment.wip_E.append(environment.machine_C.orders_inside_the_machine.pop(0))
    #             print("Step " + str(global_settings.current_time) + ": Machine_C: order finished")
    #         elif environment.machine_C.orders_inside_the_machine[0].product_type == (6):
    #             environment.wip_F.append(environment.machine_C.orders_inside_the_machine.pop(0))
    #             print("Step " + str(global_settings.current_time) + ": Machine_C: order finished")


    # Move order from machine_B to WIP D/E/F, if processing_time_remaining of order is 0
    # if len(environment.machine_B.orders_inside_the_machine) == 1:
    #     if environment.machine_B.orders_inside_the_machine[0].processing_time_remaining == 0:
    #         if environment.machine_B.orders_inside_the_machine[0].product_type == (1):
    #             print("Step " + str(global_settings.current_time) + ": Machine_B: order " +
    #                   str(environment.machine_B.orders_inside_the_machine[0].orderID) + " finished")
    #             environment.wip_D.append(environment.machine_B.orders_inside_the_machine.pop(0))
    #
    #         elif environment.machine_B.orders_inside_the_machine[0].product_type == (2):
    #             print("Step " + str(global_settings.current_time) + ": Machine_B: order " +
    #                   str(environment.machine_B.orders_inside_the_machine[0].orderID) + " finished")
    #             environment.wip_E.append(environment.machine_B.orders_inside_the_machine.pop(0))
    #
    #         elif environment.machine_B.orders_inside_the_machine[0].product_type == (3):
    #             print("Step " + str(global_settings.current_time) + ": Machine_B: order " +
    #                   str(environment.machine_B.orders_inside_the_machine[0].orderID) + " finished")
    #             environment.wip_F.append(environment.machine_B.orders_inside_the_machine.pop(0))

def move_orders_old():
    # This function moves orders from the inventories to the machines and from machines to the inventories
    # This function only applies to orders which have already been released, so we start with the WIPs.
    # Moving orders from the order pool to WIPs happens in order_release.py
    #
    # The logic is as follows:
    # ----- If a machine is not processing an order, an order from its WIP inventory is moved to the machine
    #       (e.g. machine_A pulls from wip_A)
    # The decision which order gets moved is made according to the selected scheduling rule from global_settings
    # ----- If a machine is processing, no order gets moved into it
    # ---- Once a machine finished processing an order, the order gets pushed to the next WIP inventory
    #
    # The default scheduling rule is "first_come_first_serve". From all orders inside a WIP inventory that get sent
    # to a machine, we send the one that has arrived first

    # Move order from machine_A to WIP_B, if processing_time_remaining of order is 0
    if len(environment.machine_A.orders_inside_the_machine) == 1:
        if environment.machine_A.orders_inside_the_machine[0].processing_time_remaining == 0:
            environment.wip_B.append(environment.machine_A.orders_inside_the_machine.pop(0))
            print("Step " + str(global_settings.current_time) + ": Machine_A: order finished")

    # Move one order from WIP_A to machine_A
    # If machine_A is empty, take first order from WIP_A, move it to machine_A
    # and set the order's processing time
    if global_settings.scheduling_policy == "first_come_first_serve":
        if len(environment.machine_A.orders_inside_the_machine) == 0 and len(environment.wip_A) > 0:
            environment.machine_A.orders_inside_the_machine.append(environment.wip_A.pop(0))
            environment.machine_A.orders_inside_the_machine[
                0].processing_time_remaining = environment.machine_A.processing_time

            # debugging info
            print("Step " + str(
                global_settings.current_time) + ": Order moved from WIP_A to M_A. Orders in WIP A: " + str(
                len(environment.wip_A)))

    # Move order from machine_B to FGI, if processing_time_remaining of order is 0
    if len(environment.machine_B.orders_inside_the_machine) == 1:
        if environment.machine_B.orders_inside_the_machine[0].processing_time_remaining == 0:
            environment.finished_goods_inventory.append(
                environment.machine_B.orders_inside_the_machine.pop(0))
            print("Step " + str(global_settings.current_time) + ": Machine_B: order finished")

    # Move one order from WIP_B to machine_B
    if global_settings.scheduling_policy == "first_come_first_serve":
        if len(environment.machine_B.orders_inside_the_machine) == 0 and len(environment.wip_B) > 0:
            environment.machine_B.orders_inside_the_machine.append(environment.wip_B.pop(0))
            environment.machine_B.orders_inside_the_machine[
                0].processing_time_remaining = environment.machine_B.processing_time

            # debugging info
            print("Step " + str(
                global_settings.current_time) + ": Order moved from WIP_B to M_B. Orders in WIP B: " + str(
                len(environment.wip_B)))

    # Move orders from FGI to shipped_orders once they have reached their due_date
    # Calculate lateness for each order
    if len(environment.finished_goods_inventory) > 0:
        for order_element in environment.finished_goods_inventory:
            if order_element.due_date <= global_settings.current_time:
                environment.shipped_orders.append(
                    environment.finished_goods_inventory.pop(
                        environment.finished_goods_inventory.index(order_element)))
                order_element.shipping_date = global_settings.current_time
                if order_element.shipping_date > order_element.due_date:
                    order_element.lateness = order_element.shipping_date - order_element.due_date
                print("Step " + str(global_settings.current_time) + ": Shipped order with due date " + str(
                    order_element.due_date) + " Lateness: " + str(order_element.lateness))
    return
