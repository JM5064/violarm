import cv2
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
print(wrist_start)
print(wrist_end)

total_time = 0
total_frames = 0

while True:
    start = time.time()
    ret, frame = cap.read()

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # crop wrist
    wrist_crop, cropped_wrist_x, cropped_wrist_y = arm.crop_image(grayscale_frame, wrist_start, wrist_end, padding=150)

    # get keypoints of cropped wrist
    curr_wrist_keypoints, curr_wrist_descriptors = arm.get_keypoints(wrist_crop)

    # get matches and center of wrist
    wrist_matches = arm.get_matches(arm.wrist_descriptors, curr_wrist_descriptors, threshold=300)
    wrist_center_x, wrist_center_y = arm.find_center(wrist_matches, curr_wrist_keypoints, distance_threshold=50)

    # find corners of wrist
    large_wrist_contours = arm.filter_small_contours(wrist_crop, size_threshold=300)
    wrist_x1, wrist_y1, wrist_x2, wrist_y2 = arm.find_corners(wrist_crop, (wrist_center_x, wrist_center_y), large_wrist_contours)

    wrist_x1 += cropped_wrist_x
    wrist_x2 += cropped_wrist_x
    wrist_y1 += cropped_wrist_y
    wrist_y2 += cropped_wrist_y

    wrist_start, wrist_end = arm.calculate_new_start_end((wrist_center_x + cropped_wrist_x, wrist_center_y + cropped_wrist_y), padding=150)

    # testing
    # corner_image = cv2.circle(frame, (wrist_x1, wrist_y1), 3, (0, 0, 255), 5)
    # cv2.circle(frame, (wrist_x2, wrist_y2), 3, (0, 0, 255), 5)

    # cv2.circle(frame, (wrist_center_x + cropped_wrist_x, wrist_center_y + cropped_wrist_y), 3, (255, 0, 0), 5)

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

