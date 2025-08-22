from instrument.instrument_front import InstrumentFront

import cv2
import numpy as np


def draw_arm_outline(frame, arm_keypoints):
    """Draws arm outline on frame
    args:
        frame: cv2 frame
        arm_keypoints: list[] of [x, y] arm keypoints

    returns:
        frame: cv2 frame with arm keypoints drawn
    """

    corners = []
    for point in arm_keypoints:
        x, y = int(point[0]), int(point[1])
        corners.append([x, y])
        cv2.circle(frame, (x, y), 3, (255, 0, 0), 3)

    corners = np.array(corners)
    corners = corners.reshape((-1, 1, 2))
    cv2.polylines(frame, [corners], isClosed=True, color=(255, 0, 0), thickness=2)

    return frame


def draw_hand_points(frame, hand_keypoints):
    """Draws hand keypoints on frame
    args:
        frame: cv2 frame
        hand_keypoints: list[] of [x, y] hand keypoints

    returns:
        frame: cv2 frame with hand keypoints drawn
    """

    for point in hand_keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 3, (0, 0, 255), 3)

    return frame


def draw_strings(frame, arm_keypoints, num_strings: int, instrument_front: InstrumentFront):
    """Splits arm frame and draws num_strings strings
    args:
        frame: cv2 frame to draw on
        arm_keypoints: list[] of [x, y] arm keypoints
        num_strings: int
        instrument_front: InstrumentFront for calculation
    
    returns:
        frame: cv2 frame with strings drawn
    """
    
    if len(arm_keypoints) != 4:
        return frame
    
    top_left, top_right, bottom_right, bottom_left = arm_keypoints

    top_points, bottom_points = instrument_front.get_string_baseline_points(
        top_left, top_right, bottom_left, bottom_right, num_strings)
    
    for i in range(len(top_points)):
        top_x, top_y = int(top_points[i][0]), int(top_points[i][1])
        bottom_x, bottom_y = int(bottom_points[i][0]), int(bottom_points[i][1])
        cv2.line(frame, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 2)
    
    return frame


def draw_frets(frame, arm_keypoints, fret_fractions: list[float]):
    """Draws frets as specified by fret_fractions
    args:
        frame: cv2 frame to draw on
        arm_keypoints: list[] of [x, y] arm keypoints
        fret_fractions: list[float] of fractional notes

    returns:
        frame: cv2 frame with frets drawn
    """

    if arm_keypoints is None or len(arm_keypoints) != 4:
        return frame
    
    top_left, top_right, bottom_right, bottom_left = arm_keypoints

    left_dx = bottom_left[0] - top_left[0]
    left_dy = bottom_left[1] - top_left[1]
    right_dx = bottom_right[0] - top_right[0]
    right_dy = bottom_right[1] - top_right[1]
    for fraction in fret_fractions:
        left_x = int(left_dx * fraction + top_left[0])
        left_y = int(left_dy * fraction + top_left[1])
        right_x = int(right_dx * fraction + top_right[0])
        right_y = int(right_dy * fraction + top_right[1])

        cv2.line(frame, (left_x, left_y), (right_x, right_y), (255, 255, 0), 1)
    
    return frame
