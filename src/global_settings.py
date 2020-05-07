random_seed = 1  # Setting the random seed to a fixed value allows reproducing the results
# (meaning that all random numbers are the same every time the simulation runs)

#################### Variables for the initial setup ####################
current_time = 0  # this variable keeps track of the current time/step/tick in the simulation
duration_of_one_period = 960 # default value is 960 steps per period
number_of_periods = 1000 # Default: 8000 periods
granularity_multiplier = 1 # multiplier for all time calculations, allows to obtain larger/smaller numbers in result
maximum_simulation_duration = duration_of_one_period * number_of_periods * granularity_multiplier # maximum duration in steps
warmup_duration = 1000 * granularity_multiplier # costs are reset after warmup phase NOT YET IMPLEMENTED
repetitions = 1 # how often should the entire simulation be repeated
demand_distribution = "uniform" # must be "exponential" or "uniform". Used in enviornment.py/set_next_order_arrival_time()
processing_time_distribution = "uniform" # must be "exponential" or "uniform"
shop_type = "flow_shop" # Must be either "flow_shop" or "job_shop"

#################### Variables used during the simulation runtime ####################
order_release_policy = "periodic_release" # must be "periodic_release" or "LUMS"
scheduling_policy = "first_come_first_serve"
time_of_next_order_arrival = 0 # gets overwritten for every new order
due_date_multiplier = 9 # how many periods the due date of new orders is in the future. Default: 10 periods

    #### ONLY USED FOR UNIFORM DEMAND ###
next_order_arrival_lower_bound = 78 * granularity_multiplier # lower limit for when the next order can arrive.
next_order_arrival_upper_bound = 158 * granularity_multiplier # upper limit for when the next order can arrive

    #### ONLY USED FOR EXPONENTIAL DEMAND ###
next_order_arrival_exponential_rate_parameter = 118 * granularity_multiplier # this is the λ (lambda) or
# rate parameter of the exponential distribution

#################### Variables that are used as result metrics ####################
count_of_generated_orders = 0
cost_per_item_in_shopfloor = 1  # Cost per period for every order which is either inside a machine or wip
cost_per_item_in_fgi = 4  # Cost per period for storing one order in finished goods inventory
cost_per_late_item = 16  # Cost per period for exceeding an order's due date
total_cost = 0
sum_shopfloor_cost = 0
sum_fgi_cost = 0
sum_lateness_cost = 0
# average_earliness_of_all_orders = 0
# average_flow_time_of_all_orders = 0

create_orders_csv = False
create_steps_csv = False


# Variables to turn on certain debugging statements
show_machine_output = False
show_movements_from_wip_to_machine = False
show_order_shipping = False
show_order_release = False
show_order_generation = False


def reset_global_settings():
    global current_time
    global count_of_generated_orders
    current_time = 0
    count_of_generated_orders = 0
    return




