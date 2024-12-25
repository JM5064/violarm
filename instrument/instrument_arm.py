from abc import ABC
import numpy as np

class InstrumentArm(ABC):

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    

    def distance_to_line(self, p, line):
        x, y = p

        if line is None:
            # assume vertical line
            return abs(x - self.top_right[0])
        else:
            a, b, c = line
            
            return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)


    def get_line(self, p1, p2):
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]

        if a == b == 0:
            return None
        
        c = -(a * p1[0] + b * p1[1])

        return a, b, c
