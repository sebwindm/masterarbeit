from src import global_settings, environment, class_Order
import random

random.seed(global_settings.random_seed)


def generate_order():
    "Generate new orders with random due date and random product type"
    # Orders have a fixed due date of their arrival time + 10 periods ( = 10 * 960 steps)
    product_type = random.randrange(1, 7)  # set product type to a random number between 1 and 6
    global_settings.count_of_generated_orders += 1  # Increase global counter for the total amount of orders by 1
    environment.order_pool.append(class_Order.Order(global_settings.count_of_generated_orders,
                                                    global_settings.current_time + (global_settings.due_date_multiplier
                                                                                    * global_settings.duration_of_one_period),
                                                    product_type))
    if global_settings.show_order_generation == True:
        print("Step " + str(global_settings.current_time) + " Order generated. Product type: " + str(product_type))
    # Update the time_of_next_order_arrival to a random time within the interval specified in global_settings
    global_settings.time_of_next_order_arrival = \
        global_settings.current_time + random.randrange(
            global_settings.next_order_arrival_lower_bound, global_settings.next_order_arrival_upper_bound + 1)
    return
