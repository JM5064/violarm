import fluidsynth

from instrument.instrument_string import InstrumentString


class Instrument:

    def __init__(self, strings: list[InstrumentString], soundfont_path: str):
        self.strings = strings
        self.num_strings = len(strings)
        self.notes = [None] * self.num_strings
        self.soundfont_path = soundfont_path

        self.fs = fluidsynth.Synth()

    
    def start(self) -> None:
        self.fs.start()

        sfid = self.fs.sfload(self.soundfont_path)
        self.fs.program_select(0, sfid, 0, 0)

    
    def stop(self) -> None:
        self.fs.delete()


    def add_note(self, string_num: int, frequency: int) -> None:
        self.notes[string_num] = frequency

        midi_note = InstrumentString.freq_to_midi(frequency)
        self.fs.noteon(0, midi_note, 30)

    
    def remove_note(self, string_num: int) -> None:
        if self.notes[string_num] is None:
            return
        
        current_note = self.notes[string_num]
        midi_note = InstrumentString.freq_to_midi(current_note)
        self.fs.noteoff(0, midi_note)
        
        self.notes[string_num] = None

    
    def update_note(self, string_num: int, frequency: int) -> None:
        if self.notes[string_num] is None:
            return
        
        if InstrumentString.freq_to_midi(self.notes[string_num]) == InstrumentString.freq_to_midi(frequency):
            return
        
        self.remove_note(string_num)
        self.add_note(string_num, frequency)


    def is_playing(self, string_num: int) -> bool:
        return self.notes[string_num] is not None

