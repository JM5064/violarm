from abc import ABC
import numpy as np

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


    def dot2d(self, p1, p2):
        return p1[0] * p2[0] + p1[1] * p2[1]
    

    def in_quadrilateral(self, p, a, b, c, d):

        def triangle_area(a, b, c):
            u = [b[0] - a[0], b[1] - a[1]]
            v = [c[0] - b[0], c[1] - b[1]]

            return 0.5 * abs(u[0] * v[1] - u[1] * v[0])
        
        quadrilateral_area = triangle_area(a, b, c) + triangle_area(a, d, c)
        point_triangle_sum = triangle_area(a, b, p) + triangle_area(b, p, c) + triangle_area(c, p, d) + triangle_area(d, p, a)

        return abs(quadrilateral_area - point_triangle_sum) < 0.1

