random_seed = 0  # Setting the random seed to a fixed value allows reproducing the results
# (meaning that all random numbers are the same every time the simulation runs)

#################### Variables for the initial setup ####################
current_time = 0  # this variable keeps track of the current time/step/tick in the simulation
duration_of_one_period = 960  # default value is 960 steps per period
number_of_periods = 8000  # Default: 8000 periods
maximum_simulation_duration = duration_of_one_period * number_of_periods  # maximum duration in steps

warmup_duration = 1000  # costs are reset after warmup phase
repetitions = 1  # how often should the entire simulation be repeated
demand_distribution = "exponential"  # must be "exponential" or "uniform". Used in
# environment.py/set_next_order_arrival_time()

processing_time_distribution = "exponential"  # must be "exponential" or "uniform"
shop_type = "job_shop"  # Must be either "flow_shop", "job_shop" or "job_shop_1_machine"

#################### Variables used during the simulation runtime ####################
order_release_policy = "bil"  # must be "periodic", "bil" or "lums"
scheduling_policy = "first_come_first_serve"
time_of_next_order_arrival = 0  # gets overwritten for every new order
due_date_multiplier = 9  # how many periods the due date of new orders is in the future. Default: 10 periods
planned_release_date_multiplier = 4  # used for the BIL order release policy.
# Planned release date is always planned_release_date_multiplier * duration_of_one_period + current_time

processing_times_multiplier = 1  # Each step, the affected machines (bottleneck machines) subtract
# 1 * processing_times_multiplier from their
# current order's remaining processing time. Thus a processing_times_multiplier > 1 means that machines subtract MORE
# than the default amount of remaining processing time from the order, which means that the capacity of the machine
# is increased. If the processing_times_multiplier were below 1, this would mean that the capacity of the machines are
# lower than the default value and thus the machines subtract less remaining processing time and take longer to
# process orders. As this is not a desired behaviour, we only consider processing_times_multiplier > 1.
# A processing_times_multiplier of 1 is the default value (subtract 1 each step).

#### ONLY USED FOR UNIFORM DEMAND ###
next_order_arrival_lower_bound = 78  # lower limit for when the next order can arrive.
next_order_arrival_upper_bound = 158  # upper limit for when the next order can arrive

#### ONLY USED FOR EXPONENTIAL DEMAND ###
next_order_arrival_exponential_rate_parameter = 118  # this is the λ (lambda) or
# rate parameter of the exponential distribution

#################### Variables that are used as result metrics ####################
count_of_generated_orders = 0
cost_per_item_in_shopfloor = 1  # Cost per period for every order which is either inside a machine or wip
cost_per_item_in_fgi = 4  # Cost per period for storing one order in finished goods inventory
cost_per_late_item = 16  # Cost per period for exceeding an order's due date
cost_per_overtime_period = 10  # Cost per period for running overtime on a bottleneck machine
total_cost = 0
sum_shopfloor_cost = 0
sum_fgi_cost = 0
sum_lateness_cost = 0
sum_overtime_cost = 0
temp_sum_of_late_orders_this_period = 0
temp_cost_this_period = 0
bottleneck_utilization_per_step = []

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
