from instrument_arm import InstrumentArm
import numpy as np

# TODO: time these methods to see whether class implementation is faster

class InstrumentSide(InstrumentArm):
    def __init__(self, keypoints):
        self.top_right = keypoints[1]
        self.bottom_right = keypoints[2]


    def get_sideline(self):
        return self.get_line(self.top_right, self.bottom_right)


    def get_pressed_fingers(self, fingers):
        return [finger for finger in fingers if self.is_pressed(finger, 50)]


    def is_pressed(self, finger, threshold):
        sideline = self.get_sideline()

        dist = self.distance_to_line(finger, sideline)
        
        return dist <= threshold



