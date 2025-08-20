import numpy as np


def value_to_rgb(value, methods=None, cmap=None):
    # Map to RGB components
    if methods == "cmap":
        color_value = np.array(cmap(value)[:3]).squeeze()
    elif methods == "interpolate":
        start = np.array([1, 1, 0])
        end = np.array([0, 0, 1])
        color_value = start + value * (end - start)
    else:
        r = value
        g = 1 - value
        b = value + (1 - value) / 2
        color_value = np.array([r, g, b])
    return color_value
