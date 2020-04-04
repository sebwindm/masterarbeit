import datetime

random_seed = 2020  # Setting the random seed to a fixed value allows reproducing the results
# (meaning that all random numbers are the same every time the simulation runs)
random_seed1 = datetime.datetime.now() # random seed for actual random numbers


#################### Variables for the initial setup ####################

current_time = 0  # this variable keeps track of the current time/step/tick in the simulation
duration_of_one_period = 960 # default value is 960 steps per period
number_of_periods = 1000 # Default: 8000 periods
maximum_simulation_duration = duration_of_one_period * number_of_periods # maximum duration in steps
warmup_duration = 1000
amount_of_orders_to_generate_initially = 0

#################### Variables used during the simulation runtime ####################
order_release_policy = "immediate_release" # immediate_release is actually built as periodic release at the moment
scheduling_policy = "first_come_first_serve"
time_of_next_order_arrival = 0 # gets overwritten for every new order
next_order_arrival_lower_bound = 78 # lower limit for when the next order can arrive
next_order_arrival_upper_bound = 158 # upper limit for when the next order can arrive
due_date_multiplier = 1 # how many periods the due date of new orders is in the future. Default: 10 periods


#################### Variables that are used as result metrics ####################
count_of_generated_orders = 0
total_cost = 0
sum_wip_cost = 0
sum_fgi_cost = 0
sum_lateness_cost = 0
cost_per_item_in_wip = 1  # How much it costs to store one order in a WIP inventory for one step in the simulation
cost_per_item_in_fgi = 4  # Cost for storing one order in finished goods inventory for one step
cost_per_late_item = 16  # Cost for exceeding an order's due date for one step



# Variables to turn on certain debugging statements
show_machine_output = False
show_movements_from_wip_to_machine = False
show_order_shipping = False
show_order_release = False
show_order_generation = False
