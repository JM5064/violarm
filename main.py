import cv2
import numpy as np
import mediapipe as mp
import arm
import utils
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

arm = arm.Arm()


cap = cv2.VideoCapture(0)

wrist_start = arm.wrist_start
wrist_end = arm.wrist_end

elbow_start = arm.elbow_start
elbow_end = arm.elbow_end

total_time = 0
total_frames = 0

def get_corners(frame, start, end):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # crop wrist
    crop, cropped_x, cropped_y = arm.crop_image(grayscale_frame, start, end, padding=150)

    # get keypoints of cropped wrist
    curr_keypoints, curr_descriptors = arm.get_keypoints(crop)

    # get matches and center of wrist
    matches = arm.get_matches(arm.wrist_descriptors, curr_descriptors, threshold=300)
    center_x, center_y = arm.find_center(matches, curr_keypoints, distance_threshold=50)

    # find corners of wrist
    large_contours = arm.filter_small_contours(crop, size_threshold=300)
    x1, y1, x2, y2 = arm.find_corners(crop, (center_x, center_y), large_contours)

    x1 += cropped_x
    x2 += cropped_x
    y1 += cropped_y
    y2 += cropped_y

    start, end = arm.calculate_new_start_end((center_x + cropped_x, center_y + cropped_y), padding=150)

    return x1, y1, x2, y2, start, end


while True:
    start = time.time()
    ret, frame = cap.read()

    wrist_x1, wrist_y1, wrist_x2, wrist_y2, new_wrist_start, new_wrist_end = get_corners(frame, wrist_start, wrist_end)
    wrist_start, wrist_end = new_wrist_start, new_wrist_end

    # elbow_x1, elbow_y1, elbow_x2, elbow_y2, new_elbow_start, new_elbow_end = get_corners(frame, elbow_start, elbow_end)
    # elbow_start, elbow_end = new_elbow_start, new_elbow_end

    # corners = np.array([[wrist_x1, wrist_y1], [wrist_x2, wrist_y2], [elbow_x2, elbow_y2], [elbow_x1, elbow_y1]])
    # corners = corners.reshape((-1, 1, 2))

    # testing
    corner_image = cv2.circle(frame, (wrist_x1, wrist_y1), 3, (0, 0, 255), 5)
    cv2.circle(frame, (wrist_x2, wrist_y2), 3, (0, 0, 255), 5)

    # cv2.circle(frame, (elbow_x1, elbow_y1), 3, (0, 0, 255), 5)
    # cv2.circle(frame, (elbow_x2, elbow_y2), 3, (0, 0, 255), 5)

    # cv2.polylines(frame, [corners], isClosed=True, color=(255, 0, 0), thickness=2)


    cv2.imshow("Frame", frame)

    end = time.time()
    total_time += end - start
    total_frames += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

default_time = 13.8 # milliseconds
average_time = 1000 * (total_time / total_frames)
print("Average time per frame: ", average_time, "ms")
print(1000 / average_time, "fps")
print("Average additional time per frame: ", average_time - default_time, "ms")

cap.release()
cv2.destroyAllWindows()

