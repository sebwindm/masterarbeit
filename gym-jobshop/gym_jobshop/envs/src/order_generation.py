from gym_jobshop.envs.src import environment, class_Order, global_settings
import random

random.seed(global_settings.random_seed)


def generate_order():
    """
    Generate one new object of class Order
    """
    # Orders have a fixed due date of their arrival time + 10 periods ( = 10 * 960 steps)
    product_type = random.randrange(1, 7)  # set product type to a random number between 1 and 6
    # Upper bound is higher by 1 since the upper bound is always exlcuded by randrange()
    global_settings.count_of_generated_orders += 1  # Increase global counter for the total amount of orders by 1

    if global_settings.due_date_slack_mode == "fixed":
        due_date = global_settings.current_time + (
                global_settings.fixed_due_date_slack * global_settings.duration_of_one_period)
    elif global_settings.due_date_slack_mode == "variable":
        new_due_date = random.randrange(global_settings.variable_due_date_lower_bound,
                                        global_settings.variable_due_date_upper_bound + 1)
        # Upper bound is higher by 1 since the upper bound is always excluded by randrange()
        due_date = global_settings.current_time + (
                new_due_date * global_settings.duration_of_one_period)
    else:
        raise ValueError("wrong value of global_settings.due_date_slack_mode")

    environment.order_pool.append(
        class_Order.Order(
            # orderID,creation_date, due_date, product_type, planned_release_date
            global_settings.count_of_generated_orders,  # orderid
            global_settings.current_time,  # creationdate
            due_date,  # due date
            product_type,  # product type
            due_date - global_settings.planned_release_date_multiplier *
            global_settings.duration_of_one_period)  # planned release date
    )
    environment.set_next_order_arrival_time()
    return
