class Machine(object):
    """
    A machine produces parts.
    A machine has a *name* and a number of *parts_made* thus far.
    """

    def __init__(self, name, processing_time):
        self.name = name
        self.processing_time = processing_time
        self.orders_inside_the_machine = []
        # Processing time A 0.08 B 0.17 C 0.16 D 0.22 E 0.3 F 0.22


