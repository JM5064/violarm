from instrument.instrument_string import InstrumentString


class Note:
    def __init__(self, midi_note: int, string: InstrumentString):
        self.midi_note = midi_note
        self.string = string
