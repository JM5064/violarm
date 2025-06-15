# from instrument_arm import InstrumentArm
from instrument.instrument_arm import InstrumentArm
import numpy as np
import time

class InstrumentFront(InstrumentArm):

    def __init__(self, keypoints):
        self.keypoints = keypoints


    def get_notes(self, fingers, num_strings):
        """Gets the corresponding fractional notes given finger positions
        args:
            fingers: 
            num_strings: int

        returns:
            strings:
            notes: 
        """

        if self.keypoints is None or len(self.keypoints) != 4:
            return
        
        if num_strings <= 0:
            raise Exception("num_strings must be greater than 0")
        
        top_left, top_right, bottom_right, bottom_left = self.keypoints
        
        top_points, bottom_points = self.get_string_baseline_points(top_left, top_right, bottom_left, bottom_right, num_strings)
        
        notes = []
        strings = []
        for finger in fingers:
            if not self.in_quadrilateral(finger, top_left, top_right, bottom_right, bottom_left):
                print("not in quad")
                continue

            # Find the closest string to each finger
            min_distance = float('inf')
            for i in range(num_strings):
                distance = self.distance_to_line(finger, top_points[i], bottom_points[i])

                if distance < min_distance:
                    min_distance = distance
                    closest_string = i
                else:
                    break
            
            # Record the closest string, and calculate the closest fractional note
            strings.append(closest_string)
            notes.append(self.get_note_fraction(finger, top_points[closest_string], bottom_points[closest_string]))
            
        return strings, notes
        

    def divide_baseline(self, p1, p2, num_strings):
        points = []

        dx = (p2[0] - p1[0]) / (num_strings * 2)
        dy = (p2[1] - p1[1]) / (num_strings * 2)

        for i in range(1, num_strings * 2, 2):
            x = p1[0] + i * dx
            y = p1[1] + i * dy

            points.append([x, y])

        return points
    

    def get_string_baseline_points(self, top_left, top_right, bottom_left, bottom_right, num_strings):
        top_points = self.divide_baseline(top_left, top_right, num_strings)
        bottom_points = self.divide_baseline(bottom_left, bottom_right, num_strings)

        return top_points, bottom_points
    

    def get_note_fraction(self, p, line_p1, line_p2):
        projected_vector = self.get_projection_vector(p, line_p1, line_p2)

        return np.linalg.norm(projected_vector) / self.distance(line_p1, line_p2)

