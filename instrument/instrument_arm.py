from abc import ABC
import numpy as np
import time

class InstrumentArm(ABC):

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            
    
    def get_projection_vector(self, p, line_p1, line_p2):
        """Projects vector line_p1-p onto line_p1-line_p2
        args:
            p: list[int, int]
            line_p1: list[int, int]
            line_p2: list[int, int]
        
        returns: 
            projected vector: list[int, int]
        """

        line_vector = [line_p2[0] - line_p1[0], line_p2[1] - line_p1[1]]
        to_point_vector = [p[0] - line_p1[0], p[1] - line_p1[1]]

        line_vector_norm = np.linalg.norm(line_vector)
        if line_vector_norm == 0:
            raise Exception("Invalid line points")
        
        lhs = self.dot2d(line_vector, to_point_vector) / (line_vector_norm ** 2)

        return [lhs * line_vector[0], lhs * line_vector[1]]
    

    def get_closest_point_to_line(self, p, line_p1, line_p2):
        projected_vector = self.get_projection_vector(p, line_p1, line_p2)

        return [line_p1[0] + projected_vector[0], line_p1[1] + projected_vector[1]]
    
    
    def distance_to_line(self, p, line_p1, line_p2):
        point_on_line = self.get_closest_point_to_line(p, line_p1, line_p2)

        return self.distance(p, point_on_line)


    def get_note_fraction(self, p, line_p1, line_p2):
        projected_vector = self.get_projection_vector(p, line_p1, line_p2)

        return np.linalg.norm(projected_vector) / self.distance(line_p1, line_p2)


    def dot2d(self, p1, p2):
        return p1[0] * p2[0] + p1[1] * p2[1]
    

    def in_bounds(self, p):
        pass
