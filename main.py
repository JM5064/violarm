import cv2
import numpy as np
import arm
import utils
import time
import torch
from torchvision import transforms
from ultralytics import YOLO

model = YOLO("runs/pose/yolov11/weights/best.pt")
main_cap = cv2.VideoCapture(0)
url = "camera_ip"
side_cap = cv2.VideoCapture(url)

total_time = 0
total_frames = 0


def predict(frame):
    results = model(frame, verbose=False)

    return results


def draw_arm_outline(frame, keypoints):
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

    return frame


while True:
    start = time.time()

    if main_cap.isOpened():
        ret_main, frame_main = main_cap.read()

        results = predict(frame_main)

        keypoints = results[0].keypoints.xy
        if keypoints.shape == (1, 4, 2):
            frame_main = draw_arm_outline(frame_main, keypoints)

        cv2.imshow("Main Frame", frame_main)

    if side_cap.isOpened():
        ret_side, frame_side = side_cap.read()

        results = predict(frame_side)

        keypoints = results[0].keypoints.xy
        if keypoints.shape == (1, 4, 2):
            frame_side = draw_arm_outline(frame_side, keypoints)
            
        cv2.imshow("Side Frame", frame_side)


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

main_cap.release()
cv2.destroyAllWindows()

