import cv2
import mediapipe as mp
import numpy as np


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

    cv2.imshow("Matches", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


still_image = get_still_image()
grayscale_still_image = cv2.cvtColor(still_image, cv2.COLOR_RGB2GRAY)
still_image_copy = still_image.copy()
wrist_start, wrist_end = prompt_selection(still_image_copy, "Draw Selection Around Wrist")

x1, y1 = wrist_start
x2, y2 = wrist_end

wrist_crop = grayscale_still_image[y1:y2, x1:x2]

keypoints1, descriptors1 = get_keypoints(grayscale_still_image)
keypoints2, descriptors2 = get_keypoints(wrist_crop)

matches = get_matches(descriptors1, descriptors2, 100)

draw_matches(grayscale_still_image, wrist_crop, keypoints1, keypoints2, matches)



cap.release()
cv2.destroyAllWindows()
