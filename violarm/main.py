import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from dotenv import load_dotenv
import platform
import os
import time

import video
from instrument.instrument import Instrument
from instrument.instrument_string import InstrumentString
from instrument.instrument_front import InstrumentFront
from instrument.instrument_side import InstrumentSide


def load_model():
    """Loads the corresponding model based on operating system"""

    os_name = platform.system()

    if os_name == "Darwin":
        print("Using coreml")
        return YOLO("models/arm20.mlpackage", task="pose")
    
    print("Using pytorch")
    return YOLO("models/arm20.pt", task="pose") 


def initialize_mediapipe_hands(num_frames: int):
    """Initializes mediapipe hands models
    args:
        num_frames: Number of hands models to initialize (one for each frame)

    returns:
        mp_hands: list[] of mediapipe hands
    """

    if num_frames > 2:
        raise ValueError("Maximum 2 frames permitted")
    
    mp_hands = mp.solutions.hands
    
    hands = []
    for _ in range(num_frames):
        hand = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        hands.append(hand)

    return hands


def process_frame(arm_model, hand_model, frame):
    """Processes a frame using Arm and Hand models
    args:
        arm_model: Model for processing arm keypoints
        hand_model: Model for processing hand keypoints
        frame: Frame to process

    returns:
        arm_keypoints: list[] of [x, y] arm keypoints 
        hand_keypoints: list[] of [x, y] hand keypoints
    """

    # Run arm model
    arm_results = arm_model.predict(frame, verbose=False)

    frame_keypoints = arm_results[0].keypoints.xy
    classes = arm_results[0].boxes.cls

    arm_keypoints = []
    # Only pick the first arm detected
    if len(classes) > 0:
        arm_keypoints = frame_keypoints[0]

    # Run hand model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hand_model.process(rgb_frame)
    height, width = frame.shape[:2]

    hand_keypoints = []
    if hand_results.multi_hand_landmarks:
        # Only pick the first hand detected
        hand_landmarks = hand_results.multi_hand_landmarks[0]

        index = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]
        ring = hand_landmarks.landmark[16]
        pinky = hand_landmarks.landmark[20]
        
        # Unnormalize hand points
        hand_keypoints.append([index.x * width, index.y * height])
        hand_keypoints.append([middle.x * width, middle.y * height])
        hand_keypoints.append([ring.x * width, ring.y * height])
        hand_keypoints.append([pinky.x * width, pinky.y * height])
    
    return arm_keypoints, hand_keypoints


def draw_arm_outline(frame, arm_keypoints):
    """Draws arm outline on frame
    args:
        frame: cv2 frame
        arm_keypoints: list[] of [x, y] arm keypoints

    returns:
        frame: cv2 frame with arm keypoints drawn
    """

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
    """Draws hand keypoints on frame
    args:
        frame: cv2 frame
        hand_keypoints: list[] of [x, y] hand keypoints

    returns:
        frame: cv2 frame with hand keypoints drawn
    """

    for point in hand_keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 3, (0, 0, 255), 3)

    return frame


def draw_strings(frame, arm_keypoints, num_strings: int, instrument_front: InstrumentFront):
    """Splits arm frame and draws num_strings strings
    args:
        frame: cv2 frame to draw on
        arm_keypoints: list[] of [x, y] arm keypoints
        num_strings: int
        instrument_front: InstrumentFront for calculation
    
    returns:
        frame: cv2 frame with strings drawn
    """
    
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


def draw_frets(frame, arm_keypoints, fret_fractions: list[float]):
    """Draws frets as specified by fret_fractions
    args:
        frame: cv2 frame to draw on
        arm_keypoints: list[] of [x, y] arm keypoints
        fret_fractions: list[float] of fractional notes

    returns:
        frame: cv2 frame with frets drawn
    """

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


def get_playing_notes(
        instrument_front: InstrumentFront, 
        instrument_side: InstrumentSide, 
        violin_strings: list[InstrumentString], 
        front_hand_keypoints: list[list[float]], 
        side_hand_keypoints: list[list[float]]
    ) -> list[int | None]:
    """Calculate notes being played on each string
    args:
        instrument_front: InstrumentFront, for getting notes pressed
        instrument_side: InstrumentSide, for getting notes pressed
        violin_strings: list[InstrumentString], for getting note frequencies
        front_hand_keypoints: list[list[float]], for getting notes pressed
        side_hand_keypoints: list[list[float]], for getting notes pressed

    returns:
        string_note_freqs: list[int | None] of frequencies being played on each string
    """

    # get notes pressed
    pressed_fingers = instrument_side.get_pressed_fingers(front_hand_keypoints, side_hand_keypoints)
    strings, notes = instrument_front.get_notes(pressed_fingers, len(violin_strings))

    string_notes = [[] for _ in range(len(violin_strings))]
    for i in range(len(strings)):
        string_notes[strings[i]].append(notes[i])

    # get highest notes for each string and convert into frequencies
    string_note_freqs = [InstrumentString.get_playing_note(notes) for notes in string_notes]
    for i in range(len(violin_strings)):
        if string_note_freqs[i] is None:
            continue

        string_note_freqs[i] = violin_strings[i].to_frequency(string_note_freqs[i].item())

    return string_note_freqs


def play_notes(violin: Instrument, string_note_freqs: list[int | None]) -> None:
    """Plays notes on instrument
    args:
        violin: Intrument
        string_note_freqs: list[int | None] of the current frequency of each string

    returns:
        None
    """

    for i in range(violin.num_strings):
        if violin.is_playing(i):
            if string_note_freqs[i] is None:
                violin.remove_note(i)
            else:
                violin.update_note(i, string_note_freqs[i])
        else:
            if string_note_freqs[i] is not None:
                violin.add_note(i, string_note_freqs[i])


def main():
    load_dotenv()

    model = load_model()
    hands_front, hands_side = initialize_mediapipe_hands(2)

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
    violin = Instrument(violin_strings, "violarm/instrument/violin.sf2")
    fret_fractions = a_string.calculate_fret_fractions()

    violin.start()

    total_time = 0
    total_frames = 0

    while True:
        start = time.time()

        if front_cap.isOpened():
            front_frame = front_cap.read()

            if front_frame is None:
                continue

            front_arm_keypoints, front_hand_keypoints = process_frame(model, hands_front, front_frame)

            front_frame = draw_frets(front_frame, front_arm_keypoints, fret_fractions)
            front_frame = draw_arm_outline(front_frame, front_arm_keypoints)
            front_frame = draw_strings(front_frame, front_arm_keypoints, violin.num_strings, instrument_front)
            front_frame = draw_hand_points(front_frame, front_hand_keypoints)

            cv2.imshow("Front Frame", front_frame)

        if side_cap.isOpened():
            side_frame = side_cap.read()

            if side_frame is None:
                continue
            
            side_frame = cv2.resize(side_frame, (960, 540))

            side_arm_keypoints, side_hand_keypoints = process_frame(model, hands_side, side_frame)

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
            string_note_freqs = get_playing_notes(instrument_front, instrument_side, violin_strings, front_hand_keypoints, side_hand_keypoints)

            # play notes
            play_notes(violin, string_note_freqs)

            print(f'Notes played: {string_note_freqs}')

        else:
            # remove all notes if missing keypoints
            for i in range(violin.num_strings):
                violin.remove_note(i)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end = time.time()
        # print((end-start) * 1000)
        if end - start < 1:
            total_time += end - start
            total_frames += 1

    average_time = 1000 * (total_time / total_frames)
    print("Average time per frame: ", average_time, "ms")
    print(1000 / average_time, "fps")

    violin.stop()
    front_cap.release()
    side_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
