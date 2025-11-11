import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from dotenv import load_dotenv
import platform
import os
import time

import video
from drawing import draw_arm_outline, draw_hand_points, draw_strings, draw_frets
from piece.piece_loader import PieceLoader 
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
        string_note_midis: list[int | None] of midi values being played on each string
    """

    # get notes pressed
    pressed_fingers = instrument_side.get_pressed_fingers(front_hand_keypoints, side_hand_keypoints)
    strings, notes = instrument_front.get_notes(pressed_fingers, len(violin_strings))

    string_notes = [[] for _ in range(len(violin_strings))]
    for i in range(len(strings)):
        string_notes[strings[i]].append(notes[i])

    # get highest notes for each string and convert into frequencies
    string_note_midis = [InstrumentString.get_playing_note(notes) for notes in string_notes]
    for i in range(len(violin_strings)):
        if string_note_midis[i] is None:
            continue

        string_note_midis[i] = violin_strings[i].fraction_to_midi(string_note_midis[i].item())

    return string_note_midis


def play_notes(violin: Instrument, string_note_midis: list[int | None]) -> None:
    """Plays notes on instrument
    args:
        violin: Intrument
        string_note_midis: list[int | None] of the current midi values being played on each string

    returns:
        None
    """

    for i in range(violin.num_strings):
        if violin.is_playing(i):
            if string_note_midis[i] is None:
                violin.remove_note(i)
            else:
                violin.update_note(i, string_note_midis[i])
        else:
            if string_note_midis[i] is not None:
                violin.add_note(i, string_note_midis[i])


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
    instrument_front = InstrumentFront(None, 1.39)

    g_string = InstrumentString(196, 784)
    d_string = InstrumentString(293, 1175)
    a_string = InstrumentString(440, 1760)
    e_string = InstrumentString(659, 2637)

    violin_strings = [g_string, d_string, a_string, e_string]
    violin = Instrument(violin_strings, "violarm/instrument/Sonatina_Symphonic_Orchestra.sf2", preset=12, volume=120)
    fret_fractions = a_string.calculate_fret_fractions()

    violin.start()

    piece_loader = PieceLoader(violin)
    piece = piece_loader.load_piece("Maple Leaf Rag", "Scott Joplin")
    piece.start()
    print("CURRENT NOTE:", piece.current_note.midi_note)

    total_time = 0
    total_frames = 0

    while True:
        start = time.time()

        if front_cap.isOpened():
            front_frame = front_cap.read()

            if front_frame is None:
                continue

            # Get arm and hand keypoint predictions
            front_arm_keypoints, front_hand_keypoints = process_frame(model, hands_front, front_frame)

            # Add to recent keypoints pool and use average set of keypoints
            instrument_front.add_to_recent_keypoints(front_arm_keypoints, 5)
            average_keypoints = instrument_front.get_average_keypoint_positions(front_arm_keypoints)
            instrument_front.set_keypoints(average_keypoints)

            front_frame = draw_frets(front_frame, instrument_front.keypoints, fret_fractions)
            front_frame = draw_arm_outline(front_frame, average_keypoints)
            front_frame = draw_strings(front_frame, average_keypoints, violin.num_strings, instrument_front)
            front_frame = draw_hand_points(front_frame, front_hand_keypoints)

            cv2.imshow("Front Frame", front_frame)

        if side_cap.isOpened():
            side_frame = side_cap.read()

            if side_frame is None:
                continue
            
            side_frame = cv2.resize(side_frame, (960, 540))

            # Get arm and hand keypoint predictions
            side_arm_keypoints, side_hand_keypoints = process_frame(model, hands_side, side_frame)

            # Add to recent keypoints pool and use average set of keypoints
            instrument_side.add_to_recent_keypoints(side_arm_keypoints, 3)
            average_keypoints = instrument_side.get_average_keypoint_positions(side_arm_keypoints)
            instrument_side.set_keypoints(average_keypoints)

            side_frame = draw_arm_outline(side_frame, side_arm_keypoints)
            side_frame = draw_hand_points(side_frame, side_hand_keypoints)

            cv2.imshow("Side Frame", side_frame)

        if (front_cap.isOpened() and side_cap.isOpened() and
            len(front_arm_keypoints) > 0 and len(front_hand_keypoints) > 0 and
            len(side_arm_keypoints) > 0 and len(side_hand_keypoints) > 0):

            # get playing notes
            string_note_midis = get_playing_notes(instrument_front, instrument_side, violin_strings, front_hand_keypoints, side_hand_keypoints)

            # play notes
            play_notes(violin, string_note_midis)

            # maybe here, method to check if playing note is the current target note. if so, go to next note
            # for drawing: midi -> freq -> fraction
            if piece.current_note and piece.current_note.midi_note in string_note_midis:
                next_note = piece.next_note()
                print("CURRENT NOTE SWITCHED TO ", next_note.midi_note)

            print(f'Notes played: {string_note_midis}')

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
