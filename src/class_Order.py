class Order(object):
    """
    documentation missing
    """

    def __init__(self, orderID, due_date, product_type):
        self.orderID = orderID
        self.due_date = due_date
        self.product_type = product_type # possible values range from 1 to 6
        #self.released_to_production = 0
        #self.finished_production = 0
        self.shipping_date = 0
        self.processing_time_remaining = 0
        self.lateness = 0

        #self.time_in_current_inventory = 0
        #self.time_in_all_wips = 0
        #self.time_in_fgi = 0



