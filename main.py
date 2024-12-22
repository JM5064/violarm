import cv2
import numpy as np
import arm
import video
import utils
import time
from ultralytics import YOLO

model = YOLO("runs/pose/yolov11/weights/best.pt")
main_cap = video.Video(0)
url = "camera_url"
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

    corners[-1], corners[-2] = corners[-2], corners[-1]
    corners = np.array(corners)
    corners = corners.reshape((-1, 1, 2))
    cv2.polylines(frame, [corners], isClosed=True, color=(255, 0, 0), thickness=2)

    return frame


while True:
    start = time.time()

    if main_cap.isOpened():
        main_frame = main_cap.read()

        if main_frame is None:
            continue

        main_results = predict(main_frame)

        main_keypoints = main_results[0].keypoints.xy
        if main_keypoints.shape == (1, 4, 2):
            main_frame = draw_arm_outline(main_frame, main_keypoints)

        cv2.imshow("Main Frame", main_frame)

    if side_cap.isOpened():
        side_frame = side_cap.read()

        if side_frame is None:
            continue

        side_results = predict(side_frame)

        side_keypoints = side_results[0].keypoints.xy
        if side_keypoints.shape == (1, 4, 2):
            side_frame = draw_arm_outline(side_frame, side_keypoints)
            
        cv2.imshow("Side Frame", side_frame)


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
side_cap.release()
cv2.destroyAllWindows()

