import numpy as np

# TODO: time these methods to see whether class implementation is faster

class InstrumentSide:
    def __init__(self, keypoints):
        self.top_right = keypoints[1]
        self.bottom_right = keypoints[2]


    def get_sideline(self):
        a = self.top_right[1] - self.bottom_right[1]
        b = self.bottom_right[0] - self.top_right[0]

        if a == b == 0:
            return None
        
        c = -(a * self.top_right[0] + b * self.top_right[1])

        return a, b, c


    def get_pressed_fingers(self, fingers):
        return [finger for finger in fingers if self.is_pressed(finger, 50)]



    def is_pressed(self, finger, threshold):
        x, y = finger
        sideline = self.get_sideline()

        if sideline is None:
            # assume vertical line
            dist = abs(x - self.top_right[0])
        else:
            a, b, c = sideline

            dist = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

        
        return dist <= threshold



