import numpy as np

# TODO: time these methods to see whether class implementation is faster


def get_sideline(top_right, bottom_right):
    a = top_right[1] - bottom_right[1]
    b = bottom_right[0] - top_right[0]

    if a == b == 0:
        return None
    
    c = -(a * top_right[0] + b * top_right[1])

    return a, b, c


def get_pressed_fingers(fingers, top_right, bottom_right):
    return [finger for finger in fingers if is_pressed(finger, top_right, bottom_right, 50)]



def is_pressed(finger, top_right, bottom_right, threshold):
    x, y = finger
    sideline = get_sideline(top_right, bottom_right)

    if sideline is None:
        # assume vertical line
        dist = abs(x - top_right[0])
    else:
        a, b, c = sideline

        dist = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

    
    return dist <= threshold



