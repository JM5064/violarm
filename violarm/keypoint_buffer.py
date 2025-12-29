import numpy as np
from collections import deque
import time


class KeypointBuffer:

    def __init__(self, buffer_size):
        self.recent_keypoints = deque(maxlen=buffer_size)


    def add_to_recent_keypoints(self, keypoints):
        """Adds keypoints to list of recent keypoints
        args:
            keypoints: list[] of [x, y] arm keypoints

        returns:
            None
        """

        if keypoints is None or len(keypoints) == 0:
            return
        
        self.recent_keypoints.append(keypoints)


    def get_average_keypoint_positions(self):
        """Returns the average keypoint positions in recent_keypoints
        args:
            None

        returns:
            list[] of [x, y] arm keypoints (average over recent_keypoints)
        """
        
        if len(self.recent_keypoints) == 0:
            return []

        average_keypoints = np.zeros((len(self.recent_keypoints[0]), 2))

        for keypoints in self.recent_keypoints:
            for i in range(len(keypoints)):
                average_keypoints[i] += list(keypoints[i])

        average_keypoints /= len(self.recent_keypoints)
    
        return average_keypoints.tolist()
    
