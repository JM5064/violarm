import cv2
import mediapipe as mp
import arm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

arm = arm.Arm()


cap = cv2.VideoCapture(0)

wrist_start = arm.wrist_start
wrist_end = arm.wrist_end

while True:
    ret, frame = cap.read()
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # crop wrist
    padding = 200
    wrist_crop, cropped_x, cropped_y = arm.crop_image(grayscale_frame, wrist_start, wrist_end, padding)

    # get keypoints of cropped wrist
    curr_wrist_keypoints, curr_wrist_descriptors = arm.get_keypoints(wrist_crop)

    # get matches and center of wrist
    wrist_matches = arm.get_matches(arm.wrist_descriptors, curr_wrist_descriptors, 300)
    wrist_center_x, wrist_center_y = arm.find_center(wrist_matches, curr_wrist_keypoints, 50)

    # find corners of wrist
    large_wrist_contours = arm.filter_small_contours(wrist_crop, 300)
    wrist_x1, wrist_y1, wrist_x2, wrist_y2 = arm.find_corners(wrist_crop, (wrist_center_x, wrist_center_y), large_wrist_contours)

    wrist_x1 += cropped_x
    wrist_x2 += cropped_x
    wrist_y1 += cropped_y
    wrist_y2 += cropped_y

    corner_image = cv2.circle(frame, (wrist_x1, wrist_y1), 3, (0, 0, 255), 5)
    cv2.circle(frame, (wrist_x2, wrist_y2), 3, (0, 0, 255), 5)


    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

