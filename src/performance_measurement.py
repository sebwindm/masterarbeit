from src import global_settings, environment


def measure_cost():
    # Measure cost for WIP inventories:
    global_settings.sum_wip_cost += (len(environment.wip_A) * global_settings.cost_per_item_in_wip +
                                     len(environment.wip_B) * global_settings.cost_per_item_in_wip)
    # Measure cost for finished goods inventory:
    global_settings.sum_fgi_cost += len(environment.finished_goods_inventory) * global_settings.cost_per_item_in_fgi
    # Measure cost for late goods (= backorder cost) in the last step of simulation:
    if global_settings.current_time == global_settings.maximum_simulation_duration - 1:
        # for every order in shipped_orders, add its lateness to the sum of all lateness
        for order_element in environment.shipped_orders:
            global_settings.sum_lateness_cost += order_element.lateness * global_settings.cost_per_late_item
    # Measure total cost:
    global_settings.total_cost = global_settings.sum_wip_cost + global_settings.sum_fgi_cost \
                                 + global_settings.sum_lateness_cost
    return

def measure_lateness():
    return
