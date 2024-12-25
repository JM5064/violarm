from instrument_arm import InstrumentArm
import numpy as np
import time
import cv2

class InstrumentFront:

    def __init__(self, keypoints):
        self.top_left = keypoints[0]
        self.top_right = keypoints[1]
        self.bottom_right = keypoints[2]
        self.bottom_left = keypoints[3]


    def get_notes(self, fingers, num_strings):
        if num_strings <= 0:
            raise Exception("num_strings must be greater than 0")
        
        top_points = self.divide_baseline(top_left, top_right, num_strings)
        bottom_points = self.divide_baseline(bottom_left, bottom_right, num_strings)



    def divide_baseline(self, p1, p2, num_strings):
        points = [self.top_left]
        dx = abs(p2[0] - p1[0]) / num_strings
        dy = abs(p2[1] - p1[1]) / num_strings
        for i in range(1, num_strings):
            x = p1[0] + i * dx
            y = p1[1] + i * dy

            points.append([x, y])

        points.append(top_right)

        return points



top_left = [444,238]
top_right = [531,250]
bottom_right = [511,617]
bottom_left = [397,623]

keypoints = [top_left, top_right, bottom_right, bottom_left]
pressed_fingers = [[432,322], [402,394]]

start = time.time()

for i in range(1):
    instrumentFront = InstrumentFront(keypoints)

    instrumentFront.get_string_lines(pressed_fingers, 4)

end = time.time()

print(f'Completed in {end - start} milliseconds')