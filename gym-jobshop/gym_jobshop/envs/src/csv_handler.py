from src import global_settings, environment
import time, csv, datetime

# Generate prefix for CSVs
if global_settings.processing_time_distribution == "uniform":
    processing = "UB"
else:
    processing = "EB"
if global_settings.demand_distribution == "uniform":
    demand = "UD"
else:
    demand = "ED"
if global_settings.shop_type == "flow_shop":
    shop = "FS"
else:
    shop = "JS"
csv_prefix = str(datetime.datetime.now().strftime("%d.%m.%Y")) + "_" + shop + "_" + processing + "_" + demand

def initialize_csv_files():
    # Create CSV file to store results after each iteration
    with open(str('../' + csv_prefix) + '_simulation_results.csv', mode='w') as results_CSV:
        results_writer = csv.writer(results_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(
            ['Iteration', 'Runtime(steps)', 'Orders shipped', 'WIP cost', 'FGI cost', 'Lateness cost', 'Total cost'])
        results_CSV.close()
    # Create CSV file to track utilization of wips/machines each step
    if global_settings.create_steps_csv == True:
        with open('../steps.csv', mode='w') as steps_CSV:
            results_writer = csv.writer(steps_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(['Step', 'Orders in WIP_1', 'Orders in WIP_2', 'Orders in WIP_3',
                                     'Orders in WIP_4', 'Orders in WIP_5', 'Orders in WIP_6',
                                     'util_m1', 'util_m2', 'util_m3', 'util_m4',
                                     'util_m5', 'util_m6', 'Utilization of all machines', ])
            steps_CSV.close()
    return


def write_simulation_results():
    with open(str('../' + csv_prefix) + '_simulation_results.csv', mode='a') as results_CSV:
        results_writer = csv.writer(results_CSV, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow([global_settings.random_seed, global_settings.maximum_simulation_duration,
                                 len(environment.shipped_orders), global_settings.sum_shopfloor_cost,
                                 global_settings.sum_fgi_cost, global_settings.sum_lateness_cost,
                                 global_settings.total_cost])
        results_CSV.close()
    return
