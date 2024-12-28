# from instrument_arm import InstrumentArm
from instrument.instrument_arm import InstrumentArm
import numpy as np
import time

class InstrumentFront(InstrumentArm):

    def __init__(self, keypoints):
        if keypoints is not None:
            self.top_left = keypoints[0]
            self.top_right = keypoints[1]
            self.bottom_right = keypoints[2]
            self.bottom_left = keypoints[3]
        else:
            self.top_left = None
            self.top_right = None
            self.bottom_right = None
            self.bottom_left = None


    def get_notes(self, fingers, num_strings):
        if self.top_left is None:
            return
        if num_strings <= 0:
            raise Exception("num_strings must be greater than 0")
        
        top_points = self.divide_baseline(self.top_left, self.top_right, num_strings)
        bottom_points = self.divide_baseline(self.bottom_left, self.bottom_right, num_strings)
        
        notes = []
        strings = []
        for finger in fingers:
            if not self.in_quadrilateral(finger, self.top_left, self.top_right, self.bottom_right, self.bottom_left):
                continue

            min_distance = float('inf')
            for i in range(num_strings):
                distance = self.distance_to_line(finger, top_points[i], bottom_points[i])

                if distance < min_distance:
                    min_distance = distance
                    closest_string = i
                else:
                    break
            
            strings.append(closest_string)
            notes.append(self.get_note_fraction(finger, top_points[closest_string], bottom_points[closest_string]))
            
        return strings, notes
        

    def divide_baseline(self, p1, p2, num_strings):
        points = []

        dx = abs(p2[0] - p1[0]) / (num_strings * 2)
        dy = abs(p2[1] - p1[1]) / (num_strings * 2)

        for i in range(1, num_strings * 2, 2):
            x = p1[0] + i * dx
            y = p1[1] + i * dy

            points.append([x, y])

        return points
    

    def get_note_fraction(self, p, line_p1, line_p2):
        projected_vector = self.get_projection_vector(p, line_p1, line_p2)

        return np.linalg.norm(projected_vector) / self.distance(line_p1, line_p2)



top_left = [0, 0]
top_right = [1200,0]
bottom_right = [1600,1200]
bottom_left = [0,1200]

keypoints = [top_left, top_right, bottom_right, bottom_left]
pressed_fingers = [[432,322], [80, 1000]]

start = time.time()
instrumentFront = InstrumentFront(keypoints)

for i in range(10000):
    instrumentFront.get_notes(pressed_fingers, 4)

end = time.time()

print(f'Completed in {end - start} milliseconds')