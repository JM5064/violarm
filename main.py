import cv2
import mediapipe as mp
import numpy as np
import random


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)



def get_still_image():
    # press q when ready to draw
    while True:
        ret, frame = cap.read()

        cv2.imshow("Press 'q' when ready", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return frame


still_image = get_still_image()
width = still_image.shape[1]
height = still_image.shape[0]


def prompt_selection(frame, message):
    initial_frame_copy = frame.copy()

    drawing = False
    finished = False

    start_point = None
    end_point = None

    def draw_rectangle(event, x, y, flags, param):
        nonlocal frame, initial_frame_copy, start_point, end_point, drawing, finished
        
        # undo drawing
        if event == cv2.EVENT_RBUTTONDOWN:
            finished = False
            start_point = None
            end_point = None
            
            frame = initial_frame_copy
            initial_frame_copy = frame.copy()

        if finished:
            return

        # draw rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            finished = True
            

    cv2.namedWindow(message)
    cv2.setMouseCallback(message, draw_rectangle)

    while True:
        temp_frame = frame.copy()
        if drawing and end_point is not None:
            cv2.rectangle(temp_frame, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow(message, temp_frame)

        if cv2.waitKey(1) & 0xFF == ord('q') and start_point is not None and end_point is not None:
            if start_point[0] < end_point[0] and start_point[1] < end_point[1]:
                # TODO
                # only allowed to draw from top left to bottom right. add other case
                cv2.destroyAllWindows()
                
                return start_point, end_point
            

def display_contours(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(grayscale_image, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    image_copy = image.copy()

    for contour in contours:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(image_copy, [contour], -1, color, 2)

    display_image(image_copy, "Contours")

    return image_copy


def display_image(image, window_message):
    cv2.imshow(window_message, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows
            

def get_keypoints(grayscale_image):
    detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(grayscale_image, None)

    return keypoints, descriptors


def get_matches(descriptors1, descriptors2, threshold):
    # match the keypoints of the two images
    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors1, descriptors2)

    # threshold matches
    return [m for m in matches if m.distance < threshold]


def draw_matches(grayscale_image1, grayscale_image2, keypoints1, keypoints2, matches):
    matched_image = cv2.drawMatches(grayscale_image1, keypoints1, grayscale_image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    display_image(matched_image, "Matches")


def crop_image(image, top_left_point, bottom_right_point, padding):
    x1, y1 = top_left_point
    x2, y2 = bottom_right_point

    x1 -= padding
    y1 -= padding

    x2 += padding
    y2 += padding

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > width:
        x2 = width
    if y2 > height:
        y2 = height

    return image[y1:y2, x1:x2]


grayscale_still_image = cv2.cvtColor(still_image, cv2.COLOR_RGB2GRAY)

still_image_copy = still_image.copy()
wrist_start, wrist_end = prompt_selection(still_image_copy, "Draw Selection Around Wrist")

wrist_crop = crop_image(grayscale_still_image, wrist_start, wrist_end, 0)


# get second image
still_image2 = get_still_image()
grayscale_still_image2 = cv2.cvtColor(still_image2, cv2.COLOR_RGB2GRAY)

# crop from second image
wrist_crop_larger = crop_image(grayscale_still_image2, wrist_start, wrist_end, 150)

# match first crop to second crop
keypoints1, descriptors1 = get_keypoints(wrist_crop_larger)
keypoints2, descriptors2 = get_keypoints(wrist_crop)

matches = get_matches(descriptors1, descriptors2, 200)

draw_matches(wrist_crop_larger, wrist_crop, keypoints1, keypoints2, matches)


cap.release()
cv2.destroyAllWindows()
