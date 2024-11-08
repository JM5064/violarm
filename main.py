import cv2
import mediapipe as mp
import arm
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

arm = arm.Arm()

def display_image(image, window_message):
    cv2.imshow(window_message, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows


def draw_matches(grayscale_image1, grayscale_image2, keypoints1, keypoints2, matches):
        matched_image = cv2.drawMatches(grayscale_image1, keypoints1, grayscale_image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        display_image(matched_image, "Matches")


def display_contours(image, contours):
        image_copy = image.copy()

        for contour in contours:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(image_copy, [contour], -1, color, 2)

        display_image(image_copy, "Contours")

        return image_copy


cap = cv2.VideoCapture(0)

wrist_start = arm.wrist_start
wrist_end = arm.wrist_end

while True:
    ret, frame = cap.read()
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # crop wrist
    wrist_crop, cropped_wrist_x, cropped_wrist_y = arm.crop_image(grayscale_frame, wrist_start, wrist_end, padding=200)

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

    corner_image = cv2.circle(frame, (wrist_x1, wrist_y1), 3, (0, 0, 255), 5)
    cv2.circle(frame, (wrist_x2, wrist_y2), 3, (0, 0, 255), 5)


    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

