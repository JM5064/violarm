from piece_adder import PieceAdder
import json

# add piece (midi -> list of notes?)
# load piece (set current piece to some Piece)
# piece chooser


class PieceManager:

    def __init__(self):
        self.current_piece = None

    
    def add_piece(self):
        title = input("Enter the title of the piece: ")
        composer = input("Enter the composer's full name: ")
        path = input("Enter the path to the midi file: ")

        adder = PieceAdder(title, composer, path)

        midi_notes = adder.extract_notes_from_midi()

        piece_data = {
            'title': title,
            'composer': composer,
            'midi_notes': midi_notes
        }

        adder.save_data(piece_data)


    def load_piece(title: str, composer: str):
        """Returns 
        """

        # in main: target note: if curr playing note = target note, target note = next note, switch visual indicator

        pieces_data_path = "violarm/piece/pieces.json"

        
        


manager = PieceManager()
manager.add_piece()

