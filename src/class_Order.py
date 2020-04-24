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

        self.arrival_times_m1m2m3 = [] # list of arrival times at wips/machines in order of production routing

        self.arrival_wip1 = 0 # this is equal to the order release time
        self.arrvival_m1 = 0
        self.arrival_prodstep_2_wip = 0
        self.arrival_prodstep_2_m = 0
        self.arrival_prodstep_3_wip = 0
        self.arrival_prodstep_3_m = 0


        #self.time_in_current_inventory = 0
        #self.time_in_all_wips = 0
        #self.time_in_fgi = 0



