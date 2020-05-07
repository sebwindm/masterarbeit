class Order(object):
    """
    documentation missing
    """

    def __init__(self, orderID,creation_date, due_date, product_type):
        self.orderID = orderID
        self.due_date = due_date
        self.product_type = product_type # possible values range from 1 to 6
        self.creation_date = creation_date # when the order arrived in the system
        self.order_release_date = 0
        self.finished_production_date = 0 # = arrival in FGI
        self.shipping_date = 0 # = arrival in shipped goods
        self.processing_time_remaining = 0
        self.lateness = 0
        self.earliness = 0
        self.flow_time = 0
        self.current_production_step = None # production steps refer to the processing progress of the order.
        # Counting starts from 0 (after order has been released). Example: current_production_step = 0 means
        # the order is in the first step of production, so it's either in WIP_A or machine_A. Once it moves
        # to WIP_B, current_production_step gets increased by 1 (to a value of 2).
        # ---> current_production_step is currently only used for job_shop simulations

        self.arrival_times_m1m2m3 = [] # list of arrival times at wips/machines in order of production routing

        self.arrival_wip1 = 0 # this is equal to the order release time
        self.arrvival_m1 = 0
        self.arrival_prodstep_2_wip = 0
        self.arrival_prodstep_2_m = 0
        self.arrival_prodstep_3_wip = 0
        self.arrival_prodstep_3_m = 0





