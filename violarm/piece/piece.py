from piece.fingering_manager import FingeringManager


class Piece:

    def __init__(self, notes):
        self.notes = notes
        self.current_note = 0


    def next_note(self):
        if self.current_note == len(self.notes):
            # Reset current note to 0
            self.current_note = 0

            return None
        
        self.current_note += 1

        return self.notes[self.current_note]

