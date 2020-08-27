from gym_jobshop.envs.src import environment, global_settings
import csv


def get_cost_from_current_period():
    """

    :return:
    """
    temp_wip_cost = 0
    temp_overtime_cost = 0
    temp_fgi_cost = 0
    temp_late_cost = 0
    total_cost_this_period = 0
    ################### Measure cost for shopfloor and overtime (machines + WIP inventories):
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

    ################### Measure cost for finished goods inventory:
    temp_fgi_cost = len(environment.finished_goods_inventory) * global_settings.cost_per_item_in_fgi
    print("FGI: ",len(environment.finished_goods_inventory), " orders", " Temp FGI cost: ", temp_fgi_cost)
    ################### Measure cost for late goods (= backorder cost):
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

    ################### Update total cost for shopfloor (machines + WIP inventories):
    global_settings.sum_shopfloor_cost += all_costs_from_this_period[1]
    global_settings.temp_wip_cost = all_costs_from_this_period[1]

    ################### Update total cost for finished goods inventory:
    global_settings.sum_fgi_cost += all_costs_from_this_period[3]
    global_settings.temp_fgi_cost = all_costs_from_this_period[3]
    print("global settings Temp FGI cost: ",global_settings.temp_fgi_cost)

    ################### Update total cost for late goods (= backorder cost) in the last step of simulation:
    global_settings.sum_lateness_cost += all_costs_from_this_period[4]
    global_settings.temp_lateness_cost = all_costs_from_this_period[4]

    ################### Update total cost for overtime:
    global_settings.sum_overtime_cost += all_costs_from_this_period[2]
    global_settings.temp_overtime_cost = all_costs_from_this_period[2]

    ################### Update total cost:
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


def utilization_per_step():  # this appends to the steps.csv file
    raise NotImplementedError("Function utilization_per_step() is not properly implemented")
    amount_of_active_machines = 0
    for machine in environment.list_of_all_machines:
        if len(machine.orders_inside_the_machine) > 0:
            amount_of_active_machines += 1
    utilization = amount_of_active_machines / 6
    # Append results to CSV file
    with open('../steps.csv', mode='a') as steps_CSV:
        results_writer = csv.writer(steps_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow([global_settings.current_time,
                                 len(environment.wip_A), len(environment.wip_B), len(environment.wip_C),
                                 len(environment.wip_D),
                                 len(environment.wip_E), len(environment.wip_F),
                                 len(environment.machine_A.orders_inside_the_machine),
                                 len(environment.machine_B.orders_inside_the_machine),
                                 len(environment.machine_C.orders_inside_the_machine),
                                 len(environment.machine_D.orders_inside_the_machine),
                                 len(environment.machine_E.orders_inside_the_machine),
                                 len(environment.machine_F.orders_inside_the_machine), utilization
                                 ])
        steps_CSV.close()
    return


def measure_order_flow_times():
    raise NotImplementedError("Function measure_order_flow_times() is not properly implemented")
    list_of_earliness_per_order = []
    list_of_flow_time_per_order = []
    # Create CSV file to store results after each iteration
    with open('../orders_' + str(global_settings.random_seed) + '.csv', mode='w') as orders_CSV:
        results_writer = csv.writer(orders_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(['OrderID', 'product_type', 'creation_date', 'order_release_date',
                                 'arrival_m1', 'arrival_wip_step2', 'arrival_m_step_2',
                                 'arrival_wip_step_3', 'arrival_m_step_3', 'finished_production_date',
                                 'due_date', 'shipping_date', 'lateness', 'earliness', 'flow_time'])
        for order_element in environment.shipped_orders:
            order_element.arrvival_m1 = order_element.arrival_times_m1m2m3[0]
            order_element.arrival_prodstep_2_m = order_element.arrival_times_m1m2m3[1]
            order_element.arrival_prodstep_3_m = order_element.arrival_times_m1m2m3[2]
            results_writer.writerow([
                order_element.orderID, order_element.product_type,
                order_element.creation_date, order_element.order_release_date,
                order_element.arrvival_m1, order_element.arrival_prodstep_2_wip,
                order_element.arrival_prodstep_2_m, order_element.arrival_prodstep_3_wip,
                order_element.arrival_prodstep_3_m,
                order_element.finished_production_date, order_element.due_date,
                order_element.shipping_date, order_element.lateness,
                order_element.earliness, order_element.flow_time
            ])
            list_of_earliness_per_order.append(order_element.earliness)
            list_of_flow_time_per_order.append(order_element.flow_time)

        # Append average results to the end of the CSV
        # global_settings.average_earliness_of_all_orders = statistics.mean(list_of_earliness_per_order)
        # global_settings.average_flow_time_of_all_orders = statistics.mean(list_of_flow_time_per_order)
        # results_writer.writerow(['avg_earliness','avg_lateness', 'avg_flow_time'])
        # results_writer.writerow([global_settings.average_earliness_of_all_orders,'avg_lateness', global_settings.average_flow_time_of_all_orders])
        orders_CSV.close()
    return
