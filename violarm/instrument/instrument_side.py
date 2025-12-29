from instrument.instrument_arm import InstrumentArm
# import numpy as np


class InstrumentSide(InstrumentArm):
    def __init__(self, keypoint_buffer, distance_threshold):
        super().__init__(keypoint_buffer)
        self.distance_threshold = distance_threshold


    def get_pressed_fingers(self, front_fingers, side_fingers) -> list[list[float]]:
        """Matches front_fingers with side_fingers to determine which fingers are pressed
        args:
            front_fingers: list[] of [x, y] coordinates of front_fingers
            side_fingers: list[] of [x, y] coordinates of side_fingers
        
        returns:
            list[] of [x, y] coordinates of pressed fingers
        """
        if len(front_fingers) != len(side_fingers):
            return []
        
        return [front_fingers[i] for i in range(len(side_fingers))
                if self.is_pressed(side_fingers[i], self.distance_threshold)]


    def is_pressed(self, finger: list[float], threshold: int) -> bool:
        """Detects if finger is pressed based on how close it is to the side arm
        args:
            finger: [x, y] coordinates of the finger
            threshold: int, distance threshold for determining press
        
        returns:
            Boolean indicating whether the finger is pressed
        """

        if self.keypoints is None:
            return False
        
        top_right = self.keypoints[1]
        bottom_right = self.keypoints[2]
        
        dist = self.distance_to_line(finger, top_right, bottom_right)

        return dist <= threshold

