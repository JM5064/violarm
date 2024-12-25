from instrument_arm import InstrumentArm
import numpy as np

# TODO: time these methods to see whether class implementation is faster

class InstrumentSide(InstrumentArm):
    def __init__(self, keypoints, distance_threshold):
        self.top_right = keypoints[1]
        self.bottom_right = keypoints[2]
        self.distance_threshold = distance_threshold


    def get_pressed_fingers(self, fingers):
        return [finger for finger in fingers if self.is_pressed(finger, self.distance_threshold)]


    def is_pressed(self, finger, threshold):
        dist = self.distance_to_line(finger, self.top_right, self.bottom_right)
        
        return dist <= threshold



