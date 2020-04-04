from src import order_generation, global_settings, class_Machine
import random
random.seed(global_settings.random_seed)
# instantiate machine object with name and processing time per order
machine_A = class_Machine.Machine("Machine A", 1)
machine_B = class_Machine.Machine("Machine B", 1)
machine_C = class_Machine.Machine("Machine C", 1)
machine_D = class_Machine.Machine("Machine D", 1)
machine_E = class_Machine.Machine("Machine E", 1)
machine_F = class_Machine.Machine("Machine F", 1)

def set_new_processing_times():
    machine_A.processing_time = random.randrange(30,130)
    machine_B.processing_time = random.randrange(80,240)
    machine_C.processing_time = random.randrange(50,260)
    machine_D.processing_time = random.randrange(50,370)
    machine_E.processing_time = random.randrange(200,370)
    machine_F.processing_time = random.randrange(110,320)
    return

# This is the inventory where finished goods get placed and wait until their due_date is reached, it is also named FGI.
# If an order finishes after its due_date, the order gets shipped right away and doesn't wait in the FGI
finished_goods_inventory = []

# All orders that are finished get moved here once they have reached their due dates.
shipped_orders = []

# This is the order pool where we store customer orders which have not been released to production yet.
# From this order pool, we send the orders to the first stage of production (e.g. to WIP_A and so on)
order_pool = []

# Generate WIP (work in process) inventories
# each WIP inventory is associated with one machine (and each machine with one inventory)
# when an order arrives at a machine, the order first gets placed inside the WIP inventory
# if the machine is not processing an order, it pulls one order from the WIP according to certain rules
wip_A = []
wip_B = []
wip_C = []
wip_D = []
wip_E = []
wip_F = []

# Here we give an initial set of orders for the simulation.
# Over time, more orders will arrive in the order pool periodically, but that is done in order_generation.py
def setup_order_pool(amount_to_generate):
    # this function generates the initial set of orders that will be placed
    # in the order pool at the start of the simulation
    for order in range(amount_to_generate):
        order_generation.generate_order()
    return







