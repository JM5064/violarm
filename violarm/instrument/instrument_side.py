from instrument.instrument_arm import InstrumentArm
import numpy as np

# TODO: time these methods to see whether class implementation is faster

class InstrumentSide(InstrumentArm):
    def __init__(self, keypoints, distance_threshold):
        if keypoints is not None:
            self.top_right = keypoints[1]
            self.bottom_right = keypoints[2]
        else:
            self.top_right = None
            self.bottom_right = None
        
        self.distance_threshold = distance_threshold


    def get_pressed_fingers(self, front_fingers, side_fingers):
        if len(front_fingers) != len(side_fingers):
            return []
        
        return [front_fingers[i] for i in range(len(side_fingers))
                if self.is_pressed(side_fingers[i], self.distance_threshold)]


    def is_pressed(self, finger, threshold):
        dist = self.distance_to_line(finger, self.top_right, self.bottom_right)
        return dist <= threshold



