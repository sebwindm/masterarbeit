class Machine(object):
    """

    """

    def __init__(self, name, uniform_lower_bound, uniform_upper_bound, exponential_proc_time):
        self.name = name
        self.processing_time = 0 # this value is set in environment.py where the Machine objects are created
        self.orders_inside_the_machine = []
        self.uniform_processing_time_lower_bound = uniform_lower_bound# upper/lower bound are set in environment.py
        self.uniform_processing_time_upper_bound = uniform_upper_bound
        self.exponential_processing_time = exponential_proc_time # set in environment.py. This is the λ (lambda) or
        # rate parameter of the exponential distribution



