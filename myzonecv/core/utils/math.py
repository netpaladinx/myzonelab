import math


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    if isinstance(x, (list, tuple)):
        return [math.ceil(x_ / divisor) * divisor for x_ in x]
    else:
        return math.ceil(x / divisor) * divisor


def check_divisible(x, divisor):
    if isinstance(x, (list, tuple)):
        return all([math.ceil(x_ / divisor) * divisor == x_ for x_ in x])
    else:
        return math.ceil(x / divisor) * divisor == x
