import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import multiprocessing
import os
import time

import video
from instrument.instrument import Instrument
from instrument.instrument_string import InstrumentString
from instrument.instrument_front import InstrumentFront
from instrument.instrument_side import InstrumentSide

load_dotenv()

model = YOLO("models/best_maybe2.pt")

front_cap = video.Video(0)
ip = os.environ.get('IP')
port = os.environ.get('PORT')
url = f"http://{ip}:{port}/video"
side_cap = video.Video(url)

instrument_side = InstrumentSide(None, 20)
instrument_front = InstrumentFront(None)

g_string = InstrumentString(196, 784)
d_string = InstrumentString(293, 1175)
a_string = InstrumentString(440, 1760)
e_string = InstrumentString(659, 2637)

violin_strings = [g_string, d_string, a_string, e_string]
violin = Instrument(violin_strings)
fret_fractions = a_string.calculate_fret_fractions()

violin.start()



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


def draw_strings(frame, arm_keypoints, num_strings):
    if len(arm_keypoints) != 4:
        return frame
    
    top_left, top_right, bottom_right, bottom_left = arm_keypoints

    top_points, bottom_points = instrument_front.get_string_baseline_points(
        top_left, top_right, bottom_left, bottom_right, num_strings)
    
    for i in range(len(top_points)):
        top_x, top_y = int(top_points[i][0]), int(top_points[i][1])
        bottom_x, bottom_y = int(bottom_points[i][0]), int(bottom_points[i][1])
        cv2.line(frame, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 2)
    
    return frame


def draw_frets(frame, arm_keypoints):
    if len(arm_keypoints) != 4:
        return frame
    
    top_left, top_right, bottom_right, bottom_left = arm_keypoints

    left_dx = bottom_left[0] - top_left[0]
    left_dy = bottom_left[1] - top_left[1]
    right_dx = bottom_right[0] - top_right[0]
    right_dy = bottom_right[1] - top_right[1]
    for fraction in fret_fractions:
        left_x = int(left_dx * fraction + top_left[0])
        left_y = int(left_dy * fraction + top_left[1])
        right_x = int(right_dx * fraction + top_right[0])
        right_y = int(right_dy * fraction + top_right[1])

        cv2.line(frame, (left_x, left_y), (right_x, right_y), (255, 255, 0), 1)
    
    return frame


def process_frame(frame):
    frame_results = predict(frame)
    frame_keypoints = frame_results[0].keypoints.xy
    classes = frame_results[0].boxes.cls

    arm_keypoints = []
    hand_keypoints = []

    for i in range(len(classes)):
        if classes[i] == 0:
            # hand
            if len(hand_keypoints) == 0:
                hand_keypoints.extend(frame_keypoints[i])
        else:
            # arm
            if len(arm_keypoints) == 0:
                arm_keypoints.extend(frame_keypoints[i])

    return arm_keypoints, hand_keypoints


def get_playing_notes(front_hand_keypoints, side_hand_keypoints):
    # get notes pressed
    pressed_fingers = instrument_side.get_pressed_fingers(front_hand_keypoints, side_hand_keypoints)
    strings, notes = instrument_front.get_notes(pressed_fingers, violin.num_strings)

    string_notes = [[] for _ in range(violin.num_strings)]
    for i in range(len(strings)):
        string_notes[strings[i]].append(notes[i])

    # get highest notes for each string and convert into frequencies
    string_note_freqs = [InstrumentString.get_playing_note(notes) for notes in string_notes]
    for i in range(violin.num_strings):
        if string_note_freqs[i] is None:
            continue

        string_note_freqs[i] = violin_strings[i].to_frequency(string_note_freqs[i].item())

    return string_note_freqs


def play_notes(string_note_freqs):
    for i in range(violin.num_strings):
        if violin.is_playing(i):
            if string_note_freqs[i] is None:
                violin.remove_note(i)
            else:
                violin.update_note(i, string_note_freqs[i])
        else:
            if string_note_freqs[i] is not None:
                violin.add_note(i, string_note_freqs[i])


if __name__ == '__main__':
    total_time = 0
    total_frames = 0

    while True:
        start = time.time()

        if front_cap.isOpened():
            front_frame = front_cap.read()

            if front_frame is None:
                continue

            front_arm_keypoints, front_hand_keypoints = process_frame(front_frame)

            front_frame = draw_frets(front_frame, front_arm_keypoints)
            front_frame = draw_arm_outline(front_frame, front_arm_keypoints)
            front_frame = draw_strings(front_frame, front_arm_keypoints, violin.num_strings)
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

            # update object keypoints
            instrument_front.keypoints = front_arm_keypoints
            instrument_side.keypoints = side_arm_keypoints

            # get playing notes
            string_note_freqs = get_playing_notes(front_hand_keypoints, side_hand_keypoints)

            # play notes
            play_notes(string_note_freqs)

            print(f'Notes played: {string_note_freqs}')

        else:
            # remove all notes if missing keypoints
            for i in range(violin.num_strings):
                violin.remove_note(i)


        end = time.time()
        total_time += end - start
        total_frames += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    average_time = 1000 * (total_time / total_frames)
    print("Average time per frame: ", average_time, "ms")
    print(1000 / average_time, "fps")

    violin.stop()
    front_cap.release()
    side_cap.release()
    cv2.destroyAllWindows()

