from abc import ABC
from keypoint_buffer import KeypointBuffer


class KeypointState(ABC):
    """
    Something that inherits KeypointState should have 
    state information for keypoints
    """

    def __init__(self, keypoint_buffer: KeypointBuffer):
        self.keypoints = []
        self.keypoint_buffer = keypoint_buffer


    def set_keypoints(self, keypoints):
        self.keypoints = keypoints


    def get_average_keypoint_positions(self, keypoints):
        self.keypoint_buffer.add_to_recent_keypoints(keypoints)

        return self.keypoint_buffer.get_average_keypoint_positions()

