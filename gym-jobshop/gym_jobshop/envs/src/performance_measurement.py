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
    """
    For each step that there is an order inside the bottleneck machine,
    increase bottleneck_utilization_per_step by 1.
    Later on, this value can be divided by the total number of steps from the episode
    to compute the average utilization.
    """
    if len(environment.bottleneck_machine.orders_inside_the_machine) > 0:
        global_settings.bottleneck_utilization_per_step += 1
    return


def get_order_statistics():
    """
    Return some statistics on flow time, lateness and tardiness
    """
    late_orders = 0
    tardy_orders = 0
    sum_of_lateness = 0
    sum_of_tardiness = 0
    sum_of_flow_times = 0
    # Get total number of late/tardy orders
    # Get average lateness/tardiness of orders
    for order in environment.shipped_orders:
        sum_of_flow_times += order.flow_time
        if order.lateness > 0 and order.earliness > 0:
            raise ValueError("An order was late and early at the same time")
        if order.lateness > 1:
            late_orders += 1
            sum_of_lateness += order.lateness
        elif order.earliness > 1:
            tardy_orders += 1
            sum_of_tardiness += order.earliness

    return late_orders, tardy_orders, sum_of_lateness, sum_of_tardiness, sum_of_flow_times


def evaluate_episode():
    """
    Return performance metrics to evaluate one simulation episode (that is 8000 periods).
    Metrics: total cost, wip/shopfloor cost, fgi cost, lateness cost, overtime cost,
    bottleneck machine utilization, total amount of shipped orders, flow times
    """
    total_cost = global_settings.total_cost
    wip_cost = global_settings.sum_shopfloor_cost
    fgi_cost = global_settings.sum_fgi_cost
    lateness_cost = global_settings.sum_lateness_cost
    overtime_cost = global_settings.sum_overtime_cost
    amount_of_shipped_orders = len(environment.shipped_orders)
    bottleneck_utilization = round(
            global_settings.bottleneck_utilization_per_step / global_settings.maximum_simulation_duration,2)

    late_orders, tardy_orders, sum_of_lateness, sum_of_tardiness, sum_of_flow_times = get_order_statistics()
    average_flow_time = round(sum_of_flow_times / len(environment.shipped_orders),2)

    results = [total_cost, wip_cost, fgi_cost, lateness_cost, overtime_cost, amount_of_shipped_orders,
               bottleneck_utilization, late_orders, tardy_orders, sum_of_lateness, sum_of_tardiness,
               average_flow_time]
    return results
