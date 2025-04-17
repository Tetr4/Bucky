

def clamp(x: float, lower_bound: float, upper_bound: float) -> float:
    return max(min(x, upper_bound), lower_bound)


def smoothstep(x: float, edge0: float, edge1: float) -> float:
    x = clamp((x - edge0) / (edge1 - edge0), 0, 1)
    return x * x * (3.0 - 2.0 * x)


def remap(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    x = clamp(x, in_min, in_max)
    return out_min + (((x - in_min) / (in_max - in_min)) * (out_max - out_min))
