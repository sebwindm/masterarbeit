

def exponential_decay(min_value: float, rate: float, steps: int, period: int, value: float) -> float:
    """
    Return the exponentially decayed value: max(minVal, rate ^ (period / steps) * value)
    """
    return max(min_value, rate ^ (period / steps) * value)
