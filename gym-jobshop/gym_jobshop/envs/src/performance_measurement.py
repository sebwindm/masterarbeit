from gym_jobshop.envs.src import environment, global_settings


def get_cost_from_current_period():
    """
    Todo: documentation
    :return:
    """
    temp_wip_cost = 0
    temp_overtime_cost = 0
    temp_fgi_cost = 0
    temp_late_cost = 0
    total_cost_this_period = 0
    # Measure cost for shopfloor and overtime (machines + WIP inventories):
    for wip in environment.list_of_all_wip_elements:
        temp_wip_cost += len(wip) * global_settings.cost_per_item_in_shopfloor
    for machine in environment.list_of_all_machines:
        if len(machine.orders_inside_the_machine) > 0:
            temp_wip_cost += len(machine.orders_inside_the_machine) * \
                             global_settings.cost_per_item_in_shopfloor
        # Measure overtime cost on bottleneck machine
        if global_settings.processing_times_multiplier > 1:  # only if overtime is active in this period
            if machine.name == environment.bottleneck_machine.name:
                temp_overtime_cost += global_settings.cost_per_overtime_period * global_settings.processing_times_multiplier

    # Measure cost for finished goods inventory:
    temp_fgi_cost = len(environment.finished_goods_inventory) * global_settings.cost_per_item_in_fgi
    # Measure cost for late goods (= backorder cost):
    temp_late_cost = global_settings.temp_sum_of_late_orders_this_period * global_settings.cost_per_late_item

    # Measure total cost for this period
    total_cost_this_period = temp_wip_cost + temp_overtime_cost + temp_fgi_cost + temp_late_cost

    global_settings.temp_sum_of_late_orders_this_period = 0  # reset the count of late orders until the next period's end


    # temp_amount_wip = 0
    # for wip in environment.list_of_all_wip_elements:
    #     temp_amount_wip += len(wip)
    # for machine in environment.list_of_all_machines:
    #     temp_amount_wip += len(machine.orders_inside_the_machine)
    # print("tmp fgi cost:",temp_fgi_cost, "len fgi:",len(environment.finished_goods_inventory),
    #       "tmp wip cost:",temp_wip_cost, "len wips:",temp_amount_wip)


    return [total_cost_this_period, temp_wip_cost, temp_overtime_cost, temp_fgi_cost, temp_late_cost]


def update_total_cost():
    """
    Logic for measuring costs:
    Once at the end of every period (after orders have been released, processed and shipped) we update the cost.
    The cost that incurred in the past period will be added to the sum of the respective cost (e.g. we
        add all cost from FGI inventories from the past period to the sum of all FGI costs and so on)
    The cost is calculated by multiplying a given cost factor (see global_settings.py) with the amount of orders
    in the respective inventory (e.g. we have 5 orders in FGI and the cost factor is 4, then the cost for that period
    is 20)
    :return: return nothing
    """
    global_settings.temp_cost_this_period = 0

    all_costs_from_this_period = get_cost_from_current_period()

    # Update total cost for shopfloor (machines + WIP inventories):
    global_settings.sum_shopfloor_cost += all_costs_from_this_period[1]
    global_settings.temp_wip_cost = all_costs_from_this_period[1]

    # Update total cost for finished goods inventory:
    global_settings.sum_fgi_cost += all_costs_from_this_period[3]
    global_settings.temp_fgi_cost = all_costs_from_this_period[3]


    #print("global_settings.temp_fgi_cost: ",global_settings.temp_fgi_cost)

    # Update total cost for late goods (= backorder cost) in the last step of simulation:
    global_settings.sum_lateness_cost += all_costs_from_this_period[4]
    global_settings.temp_lateness_cost = all_costs_from_this_period[4]

    # Update total cost for overtime:
    global_settings.sum_overtime_cost += all_costs_from_this_period[2]
    global_settings.temp_overtime_cost = all_costs_from_this_period[2]

    # Update total cost:
    global_settings.total_cost += all_costs_from_this_period[0]
    global_settings.temp_cost_this_period = all_costs_from_this_period[0]
    return


def reset_all_costs():
    global_settings.total_cost = 0
    global_settings.sum_shopfloor_cost = 0
    global_settings.sum_fgi_cost = 0
    global_settings.sum_lateness_cost = 0
    global_settings.sum_overtime_cost = 0
    return


def measure_bottleneck_utilization():
    if len(environment.bottleneck_machine.orders_inside_the_machine) > 0:
        global_settings.bottleneck_utilization_per_step += 1
    return
