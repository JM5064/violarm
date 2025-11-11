from piece.fingering_manager import FingeringManager
from piece.note import Note


class Piece:

    def __init__(self, notes: list[Note]):
        self.notes = notes   
        self.current_note = None
        self.initialized = False

        self.gen = self.note_gen() 


    def note_gen(self):
        while True:
            for note in self.notes:
                yield note

            yield None

        
    def start(self):
        self.initialized = True
        self.next_note()


    def next_note(self):
        """Returns the next note"""

        self.current_note = next(self.gen)

        return self.current_note
    

    def reset_piece(self):
        """Resets note generator"""
        
        self.gen = self.note_gen()
        self.current_note = None

