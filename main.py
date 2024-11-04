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


still_image = get_still_image()
still_image_copy = still_image.copy()
wrist_start, wrist_end = prompt_selection(still_image_copy, "Draw Selection Around Wrist")
# elbow_start, elbow_end = prompt_selection(still_image_copy, "Draw Selection Around Elbow")

x1, y1 = wrist_start
x2, y2 = wrist_end

wrist_crop = still_image[y1:y2, x1:x2]

cv2.imshow('Wrist Crop', wrist_crop)
cv2.waitKey(0)






cap.release()
cv2.destroyAllWindows()
