from instrument.instrument import Instrument
from instrument.instrument_string import InstrumentString
from piece.note import Note

import math


class FingeringManager:

    def __init__(self, instrument: Instrument):
        self.instrument = instrument

    
    def validate_notes(self, notes: list[int]) -> None:
        """Checks whether the piece is playable
        args:
            notes: list[int], list of notes of the piece in midi format

        returns:
            None, raises errors
        """
        
        n = len(notes)
        num_strings = len(self.instrument.strings)
        if n == 0 or num_strings == 0:
            raise ValueError(f'Invalid notes or strings: Number of notes: {n}, Number of strings: {num_strings}')

        # Check that the piece is playable. Assume strings are ordered lowest to highest
        lowest_string =  self.instrument.strings[0]
        for note in notes:
            note_freq = InstrumentString.midi_to_freq(note)
            if note_freq < lowest_string.min_freq:
                print(note_freq, lowest_string.min_freq)
                raise ValueError(f'Piece is impossible to play. Note {note} is lower than lowest string')

    
    def get_fingering(self, notes: list[int]) -> list[Note]:
        """Returns the string of the current note
        args:
            curr_note: Current note in midi format
            prev_note: Previous note in midi format
            prev_note_string: InstrumentString which the previous note was on

        returns:
            InstrumentString of the best string to play curr_note on
        """

        # Check that piece is playable
        self.validate_notes(notes)

        n = len(notes)
        num_strings = len(self.instrument.strings)

        # Create graph, with each note's possible string being a vertex
        # Initialize notes
        possible_notes: list[list[Note]] = [[] for _ in range(n)]
        notes_map = {}
        for i in range(n):
            note_freq = InstrumentString.midi_to_freq(notes[i])

            # Find the highest two (or one) string(s) that can play the note
            for j in range(num_strings-1, -1, -1):
                if note_freq < self.instrument.strings[j].min_freq:
                    continue
                
                note = Note(notes[i], self.instrument.strings[j])

                possible_notes[i].append(note)
                notes_map[note] = {
                    "cost": float('inf'),
                    "best_prev": None
                }

                if len(possible_notes[i]) == 2:
                    break

        # Update starting note costs to 0 or 1 depending on the string
        for i in range(len(possible_notes[0])):
            notes_map[possible_notes[0][i]]["cost"] = i

        # DAG DP
        for i in range(1, n):
            curr_notes = possible_notes[i]
            prev_notes = possible_notes[i-1]

            # For each current note, update its cost by finding the most optimal previous note
            for curr_note in curr_notes:
                for prev_note in prev_notes:
                    cost = notes_map[prev_note]["cost"] + self.transition_cost(prev_note, curr_note, curr_notes)
                    if cost < notes_map[curr_note]["cost"]:
                        notes_map[curr_note]["cost"] = cost
                        notes_map[curr_note]["best_prev"] = prev_note


        # Debug print costs for each of the last notes
        for note in possible_notes[-1]:
            print(notes_map[note]["cost"])

        # Get the best fingering path
        best_last_note = min(possible_notes[-1], key=lambda note: notes_map[note]["cost"])
        best_fingering: list[Note] = []
        curr = best_last_note
        while curr is not None:
            best_fingering.append(curr)
            curr = notes_map[curr]["best_prev"]

        best_fingering.reverse()

        return best_fingering


    def transition_cost(self, prev_note: Note, curr_note: Note, curr_notes: list[Note]) -> float:
        """Defines a heuristic for transitioning between two notes
        args:
            prev_note, curr_note: the previous and current note
            curr_notes: list[Note] of possible ways to play the current note

        returns:
            float cost
        """

        cost = 0

        # Cost: 1 if not on same string
        if prev_note.string != curr_note.string:
            cost += 1

        # Cost: 1 if current note is on the lower string of the two strings available
        cost += curr_notes.index(curr_note)

        # Cost: 1 if "open string"
        if curr_note.midi_note == InstrumentString.freq_to_midi(curr_note.string.min_freq):
            cost += 1

        # Cost: Vertical distance between notes
        normalized_prev_note = prev_note.midi_note - InstrumentString.freq_to_midi(prev_note.string.min_freq)
        normalized_curr_note = curr_note.midi_note - InstrumentString.freq_to_midi(curr_note.string.min_freq)

        distance = abs(normalized_curr_note - normalized_prev_note)

        # Hyperparameter for determining how much distance should impact cost
        a = 0.25
        cost += a * distance

        return cost



g_string = InstrumentString(196, 784)
d_string = InstrumentString(293, 1175)
a_string = InstrumentString(440, 1760)
e_string = InstrumentString(659, 2637)

violin = Instrument([g_string, d_string, a_string, e_string], "")
fm = FingeringManager(violin)


notes =  [67, 69, 71, 69, 74, 72, 72, 71, 69, 67, 66, 62, 66, 67, 69, 71, 69, 74, 72, 62, 72, 71, 76, 74, 72, 71, 76, 74, 69, 67, 79, 81, 83, 81, 86, 84, 84, 83, 81, 79, 78, 74, 78, 79, 81, 83, 81, 86, 84, 86, 84, 83, 79, 83, 88, 79, 88, 86, 74, 86, 84, 78, 84, 83, 79, 83, 88, 79, 88, 86, 74, 86, 84, 81, 78, 79, 76, 79, 76, 74, 79, 74, 72, 62, 72, 71, 72, 74, 76, 79, 76, 74, 79, 74, 72, 62, 72, 71, 83, 83, 83, 74, 83, 74, 83, 83, 83, 74, 83, 84, 81, 78, 74, 69, 62, 79, 79, 71, 79, 79, 79, 71, 79, 81, 78, 74, 69, 66, 64, 62, 64, 67, 64, 67, 64, 64, 55, 64, 62, 67, 62, 67, 62, 62, 55, 62, 60, 72, 69, 66, 64, 62, 66, 72, 71, 74, 71, 67, 62, 55, 59, 62, 64, 67, 64, 67, 64, 64, 55, 64, 62, 67, 62, 67, 62, 62, 55, 62, 72, 72, 62, 72, 72, 72, 62, 72, 71, 71, 62, 71, 71, 71, 62, 71, 76, 79, 76, 79, 76, 76, 79, 76, 74, 79, 74, 79, 74, 74, 79, 74, 72, 76, 72, 69, 69, 62, 69, 71, 74, 71, 67, 67, 71, 74, 76, 79, 76, 79, 76, 76, 79, 76, 74, 79, 74, 79, 74, 74, 79, 74, 72, 62, 72, 72, 62, 72, 71, 62, 71, 71, 62, 71, 64, 64, 64, 55, 64, 62, 62, 62, 55, 62, 66, 66, 66, 66, 64, 66, 67, 67, 67, 67, 66, 64, 64, 55, 55, 55, 64, 64, 62, 55, 55, 55, 62, 62, 66, 57, 57, 57, 66, 66, 67, 74, 71, 67, 62, 55, 86, 83, 79, 74, 71, 67, 84, 81, 78, 74, 69, 66, 79, 76, 71, 67, 62, 79, 78, 74, 69, 66, 64, 62, 76, 72, 67, 64, 60, 55, 59, 55, 62, 59, 67, 62, 84, 81, 78, 74, 69, 66, 67, 62, 83, 79, 55, 79, 83, 79, 83, 79, 78, 81, 78, 76, 79, 76, 79, 76, 74, 78, 74, 72, 76, 72, 76, 72, 71, 74, 79, 78, 84, 78, 79, 62, 59, 55, 79, 79, 78, 69, 66, 62, 76, 76, 74, 62, 59, 55, 72, 72, 71, 74, 79, 78, 84, 78, 79, 62, 59, 55, 64, 62, 66, 67, 64, 62, 78, 79]
fingering = fm.get_fingering(notes)

for note in fingering:
    print(note.midi_note, "on string", note.string.min_freq)


                

    
