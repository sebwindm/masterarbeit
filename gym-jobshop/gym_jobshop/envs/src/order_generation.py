from gym_jobshop.envs.src import environment, class_Order, global_settings
import random

random.seed(global_settings.random_seed)

def generate_order():
    "Generate new orders with random due date and random product type"
    # Orders have a fixed due date of their arrival time + 10 periods ( = 10 * 960 steps)
    product_type = random.randrange(1, 7)  # set product type to a random number between 1 and 6
    global_settings.count_of_generated_orders += 1  # Increase global counter for the total amount of orders by 1
    environment.order_pool.append(
        class_Order.Order(
        global_settings.count_of_generated_orders, global_settings.current_time,
            global_settings.current_time + (
                        global_settings.due_date_multiplier * global_settings.duration_of_one_period),
        product_type,
            global_settings.current_time +
            global_settings.planned_release_date_multiplier * global_settings.duration_of_one_period)
    )
    environment.set_next_order_arrival_time()
    # DEBUG INFO:
    if global_settings.show_order_generation == True:
        print("Step " + str(global_settings.current_time) + " Order generated. Product type: " + str(product_type)
              + " || Due Date: " + str(global_settings.current_time + (
                global_settings.due_date_multiplier * global_settings.duration_of_one_period)
                                       ) + " || orderID: " + str(global_settings.count_of_generated_orders))
    return

