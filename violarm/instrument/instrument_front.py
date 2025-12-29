from instrument.instrument_arm import InstrumentArm
import numpy as np


class InstrumentFront(InstrumentArm):

    def __init__(self, keypoint_buffer, fingerboard_extension_multiplier=1):
        super().__init__(keypoint_buffer)
        self.fingerboard_extension_multiplier = fingerboard_extension_multiplier


    def get_notes(self, fingers: list[list[float]], num_strings: int) -> tuple[list[int], list[float]]:
        """Gets the corresponding fractional notes and the strings they're on given finger positions
        args:
            fingers: list[] of [x, y] fingers coords
            num_strings: int

        returns:
            strings: list[int] of strings each valid finger is closest to
            notes: list[float] of fractional notes each valid finger is closest to
        """

        # Check that there are 4 arm keypoints
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
    

    def get_extended_fingerboard_keypoints(self):
        """Extends fingerboard by moving down bottom arm keypoints
        args:
            None

        returns:
            top_left, top_right, and new bottom_right, bottom_left keypoints
        """

        # Check that there are 4 arm keypoints
        if self.keypoints is None or len(self.keypoints) != 4:
            return self.keypoints
        
        top_left, top_right, bottom_right, bottom_left = self.keypoints

        # Calculate the differences in x, y between top and bottom points
        rightdx = bottom_right[0] - top_right[0]
        rightdy = bottom_right[1] - top_right[1]

        leftdx = bottom_left[0] - top_left[0]
        leftdy = bottom_left[1] - top_left[1]

        new_bottom_right  = [top_right[0] + rightdx * self.fingerboard_extension_multiplier, top_right[1] + rightdy * self.fingerboard_extension_multiplier]
        new_bottom_left = [top_left[0] + leftdx * self.fingerboard_extension_multiplier, top_left[1] + leftdy * self.fingerboard_extension_multiplier]

        return top_left, top_right, new_bottom_right, new_bottom_left


    def divide_baseline(self, p1, p2, num_strings: int) -> list[list[float]]:
        """Divides line between p1 and p2 evenly into num_strings strings
        args:
            p1: [x, y] left point of baseline
            p2: [x, y] right point of baseline
            num_strings: Number of strings to divide into

        returns:
            points: list[] of [x, y] points of representing the center of each string
        """

        points = []

        dx = (p2[0] - p1[0]) / (num_strings * 2)
        dy = (p2[1] - p1[1]) / (num_strings * 2)

        for i in range(1, num_strings * 2, 2):
            x = p1[0] + i * dx
            y = p1[1] + i * dy

            points.append([x, y])

        return points
    

    def get_string_baseline_points(self, top_left, top_right, bottom_left, bottom_right, num_strings: int) -> tuple[list[list[float]], list[list[float]]]:
        """Gets the string baseline points for the bottom and top baselines
        args:
            top_left, top_right, bottom_left, bottom_right: [x, y] coords of the respective arm keypoints
            num_strings: Number of strings to divide baseline into

        returns:
            top_points: Divided points from divide_baseline of the top points
            bottom_points: Divided points from divide_baseline of the bottom points
        """

        top_points = self.divide_baseline(top_left, top_right, num_strings)
        bottom_points = self.divide_baseline(bottom_left, bottom_right, num_strings)

        return top_points, bottom_points
    

    def get_note_fraction(self, p, line_p1, line_p2) -> float:
        """Gets the fraction of point p of the line from line_p1 to line_p2
        args:
            p: [x, y] point
            line_p1: [x, y] point representing the start of a line
            line_p2: [x, y] point representing the end of a line

        returns:
            float, fraction of how far down the line p is / the total length of the line
        """

        projected_vector = self.get_projection_vector(p, line_p1, line_p2)

        return np.linalg.norm(projected_vector) / self.distance(line_p1, line_p2)


    def set_keypoints(self, keypoints):
        super().set_keypoints(keypoints)

        self.keypoints = self.get_extended_fingerboard_keypoints()
    