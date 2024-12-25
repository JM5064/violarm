from instrument_arm import InstrumentArm
import numpy as np

# TODO: time these methods to see whether class implementation is faster

class InstrumentSide(InstrumentArm):
    def __init__(self, keypoints):
        self.top_right = keypoints[1]
        self.bottom_right = keypoints[2]


    def get_sideline(self):
        return InstrumentArm.get_line(self.top_right, self.bottom_right)


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



