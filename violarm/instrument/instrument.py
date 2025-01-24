import numpy as np
import time
import pyaudio

from instrument.instrument_string import InstrumentString


class Instrument:

    def __init__(self, strings: list[InstrumentString], amplitude=0.5):
        self.strings = strings
        self.num_strings = len(strings)
        self.notes = [None] * self.num_strings

        self.SAMPLE_RATE = 44100
        self.amplitude = amplitude

        self.p = pyaudio.PyAudio()
        self.stream = None


    def audio_callback(self, in_data, frame_count, time_info, status):
        # np array representing times
        t = np.arange(frame_count) / self.SAMPLE_RATE
        audio = np.zeros(frame_count, dtype=np.float32)

        for i in range(len(self.notes)):
            if self.notes[i] is None:
                continue

            frequency, phase = self.notes[i]

            wave = self.amplitude * np.sin(2 * np.pi * frequency * (t + phase / self.SAMPLE_RATE))
            audio += wave

            self.notes[i] = (frequency, (phase + frame_count) % self.SAMPLE_RATE)

        return (audio.tobytes(), pyaudio.paContinue)

    
    def start(self) -> None:
        if self.stream is not None:
            return
        
        # open pyaudio stream
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=self.SAMPLE_RATE,
                                  output=True,
                                  stream_callback=self.audio_callback)
        
        self.stream.start_stream()

    
    def stop(self) -> None:
        if self.stream is None:
            return
        
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None


    def add_note(self, string_num: int, frequency: int) -> None:
        self.notes[string_num] = (frequency, 0)

    
    def remove_note(self, string_num: int) -> None:
        self.notes[string_num] = None

    
    def update_note(self, string_num: int, frequency: int) -> None:
        if self.notes[string_num] is None:
            return
        
        _, phase = self.notes[string_num]
        self.notes[string_num] = (frequency, phase)


