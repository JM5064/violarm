from instrument.instrument_arm import InstrumentArm
import numpy as np

# TODO: time these methods to see whether class implementation is faster

class InstrumentSide(InstrumentArm):
    def __init__(self, keypoints, distance_threshold):
        self.keypoints = keypoints        
        self.distance_threshold = distance_threshold


    def get_pressed_fingers(self, front_fingers, side_fingers):
        if len(front_fingers) != len(side_fingers):
            return []
        
        return [front_fingers[i] for i in range(len(side_fingers))
                if self.is_pressed(side_fingers[i], self.distance_threshold)]


    def is_pressed(self, finger, threshold):
        if self.keypoints is None:
            return False
        
        top_right = self.keypoints[1]
        bottom_right = self.keypoints[2]
        
        dist = self.distance_to_line(finger, top_right, bottom_right)

        return dist <= threshold



