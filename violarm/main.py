import cv2
import numpy as np
from ultralytics import YOLO
import time

import video
from instrument.instrument import Instrument
from instrument.instrument_string import InstrumentString
from instrument.instrument_front import InstrumentFront
from instrument.instrument_side import InstrumentSide


model = YOLO("/Users/justinmao/Documents/GitHub/violarm/best_hand_arm.pt")
front_cap = video.Video(0)
url = "http://192.168.0.57:4747/video"
side_cap = video.Video(url)

total_time = 0
total_frames = 0


instrument_side = InstrumentSide(None, 20)
instrument_front = InstrumentFront(None)

def initialize_instrument():
    global instrument_side, instrument_front

    a_string = InstrumentString(440, 880)
    
    violin = Instrument([a_string])


initialize_instrument()


def predict(frame):
    results = model(frame, verbose=False)

    return results


def draw_arm_outline(frame, arm_keypoints):
    corners = []
    for point in arm_keypoints:
        x, y = int(point[0]), int(point[1])
        corners.append([x, y])
        cv2.circle(frame, (x, y), 3, (255, 0, 0), 3)

    corners = np.array(corners)
    corners = corners.reshape((-1, 1, 2))
    cv2.polylines(frame, [corners], isClosed=True, color=(255, 0, 0), thickness=2)

    return frame


def draw_hand_points(frame, hand_keypoints):
    for point in hand_keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 3, (0, 0, 255), 3)

    return frame


def process_frame(frame):
    frame_results = predict(frame)
    frame_keypoints = frame_results[0].keypoints.xy

    arm_keypoints = []
    hand_keypoints = []

    if len(frame_keypoints[0]) > 0:
        for keypoint in frame_keypoints[0]:
            arm_keypoints.append(keypoint)

    if len(frame_keypoints) > 1:
        for keypoint in frame_keypoints[1]:
            hand_keypoints.append(keypoint)

    return arm_keypoints, hand_keypoints


while True:
    start = time.time()

    if front_cap.isOpened():
        front_frame = front_cap.read()

        if front_frame is None:
            continue

        front_arm_keypoints, front_hand_keypoints = process_frame(front_frame)

        front_frame = draw_arm_outline(front_frame, front_arm_keypoints)
        front_frame = draw_hand_points(front_frame, front_hand_keypoints)

        cv2.imshow("Front Frame", front_frame)

    if side_cap.isOpened():
        side_frame = side_cap.read()

        if side_frame is None:
            continue
        
        side_frame = cv2.resize(side_frame, (960, 720))

        side_arm_keypoints, side_hand_keypoints = process_frame(side_frame)

        side_frame = draw_arm_outline(side_frame, side_arm_keypoints)
        side_frame = draw_hand_points(side_frame, side_hand_keypoints)

        cv2.imshow("Side Frame", side_frame)

    if (front_cap.isOpened() and side_cap.isOpened() and
        len(front_arm_keypoints) > 0 and len(front_hand_keypoints) > 0 and
        len(side_arm_keypoints) > 0 and len(side_hand_keypoints) > 0):


        instrument_front.keypoints = front_arm_keypoints
        instrument_side.keypoints = side_arm_keypoints

        pressed_fingers = instrument_side.get_pressed_fingers(front_hand_keypoints, side_hand_keypoints)
        
        string, note = instrument_front.get_notes(pressed_fingers, 2)
        print(f'Note value(s) {note} played on string(s) {string}')
        print("---------")


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

