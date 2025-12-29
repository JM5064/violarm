from keypoint_state import KeypointState
from keypoint_buffer import KeypointBuffer


class InstrumentHand(KeypointState):

    def __init__(self, keypoint_buffer: KeypointBuffer):
        super().__init__(keypoint_buffer)
