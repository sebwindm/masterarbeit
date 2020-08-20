

def exponential_decay(min_value: float, rate: float, steps: int, period: int, value: float) -> float:
    """
    Return the exponentially decayed value: max(minVal, rate ^ (period / steps) * value)
    """
    return max(min_value, rate ** (period / steps) * value)


def normalize_observation(observation):
    return observation / 3.0 - 1


def normalize_reward(reward):
    return reward / 300
