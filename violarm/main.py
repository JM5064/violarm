import cv2
import numpy as np
from ultralytics import YOLO
import time

import video


model = YOLO("/Users/justinmao/Documents/GitHub/violarm/best_hand_arm.pt")
front_cap = video.Video(0)
url = "http://192.168.0.57:4747/video"
side_cap = video.Video(url)

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

    corners = np.array(corners)
    corners = corners.reshape((-1, 1, 2))
    cv2.polylines(frame, [corners], isClosed=True, color=(255, 0, 0), thickness=2)

    if len(keypoints.numpy()) > 1:
        points = keypoints.numpy()[1]
        for point in points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), 3)

    return frame


while True:
    start = time.time()

    if front_cap.isOpened():
        front_frame = front_cap.read()

        if front_frame is None:
            continue

        front_results = predict(front_frame)

        front_keypoints = front_results[0].keypoints.xy
        front_frame = draw_arm_outline(front_frame, front_keypoints)

        cv2.imshow("Front Frame", front_frame)

    if side_cap.isOpened():
        side_frame = side_cap.read()

        if side_frame is None:
            continue
        
        side_frame = cv2.resize(side_frame, (960, 720))

        side_results = predict(side_frame)

        side_keypoints = side_results[0].keypoints.xy
        side_frame = draw_arm_outline(side_frame, side_keypoints)
            
        cv2.imshow("Side Frame", side_frame)


    end = time.time()
    total_time += end - start
    total_frames += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

default_time = 0.97 # milliseconds
average_time = 1000 * (total_time / total_frames)
print("Average time per frame: ", average_time, "ms")
print(1000 / average_time, "fps")
print("Average additional time per frame: ", average_time - default_time, "ms")

front_cap.release()
side_cap.release()
cv2.destroyAllWindows()

