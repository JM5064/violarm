import fluidsynth

from instrument.instrument_string import InstrumentString


class Instrument:

    def __init__(self, strings: list[InstrumentString], soundfont_path: str, preset=0, volume=30):
        self.strings = strings
        self.num_strings = len(strings)
        self.notes = [None] * self.num_strings

        self.fs = fluidsynth.Synth()
        self.soundfont_path = soundfont_path
        self.preset = preset
        self.volume = volume

    
    def start(self) -> None:
        self.fs.setting("synth.gain", 2.0)
        self.fs.start()

        sfid = self.fs.sfload(self.soundfont_path)
        self.fs.program_select(0, sfid, 0, self.preset)

    
    def stop(self) -> None:
        self.fs.delete()


    def add_note(self, string_num: int, midi_note: int) -> None:
        self.notes[string_num] = midi_note

        self.fs.noteon(0, midi_note, self.volume)

    
    def remove_note(self, string_num: int) -> None:
        if self.notes[string_num] is None:
            return
        
        current_note = self.notes[string_num]
        self.fs.noteoff(0, current_note)
        
        self.notes[string_num] = None

    
    def update_note(self, string_num: int, midi_note: int) -> None:
        if self.notes[string_num] is None:
            return
        
        if self.notes[string_num] == midi_note:
            return
        
        self.remove_note(string_num)
        self.add_note(string_num, midi_note)


    def is_playing(self, string_num: int) -> bool:
        return self.notes[string_num] is not None

