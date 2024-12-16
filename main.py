import cv2
import numpy as np
import mediapipe as mp
import arm
import utils
import time
import torch
from torchvision import transforms
from ultralytics import YOLO

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

model = YOLO("runs/pose/train2/weights/best.pt")
cap = cv2.VideoCapture(0)

total_time = 0
total_frames = 0


def predict(frame):
    results = model(frame)

    return results


while True:
    start = time.time()
    ret, frame = cap.read()

    results = predict(frame)

    keypoints = results[0].keypoints.xy
    if keypoints.shape == torch.Size([1, 4, 2]):
        points = keypoints.numpy()[0]

        corners = []
        for point in points:
            x, y = int(point[0]), int(point[1])
            corners.append([x, y])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), 3)

        corners[-1], corners[-2] = corners[-2], corners[-1]
        corners = np.array(corners)
        corners = corners.reshape((-1, 1, 2))
        cv2.polylines(frame, [corners], isClosed=True, color=(255, 0, 0), thickness=2)

    # hand_points = hands.process(frame)
    # if hand_points.multi_hand_landmarks:
    #     for hand_landmarks in hand_points.multi_hand_landmarks:
    #         mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


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

