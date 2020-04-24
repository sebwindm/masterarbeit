class Machine(object):
    """
    A machine produces parts.
    A machine has a *name* and a number of *parts_made* thus far.
    """

    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.processing_time = 0 # this value is set in environment.py where the Machine objects are created
        self.orders_inside_the_machine = []
        self.processing_time_lower_bound = lower_bound
        self.processing_time_upper_bound = upper_bound



