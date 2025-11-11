from .piece_adder import PieceAdder
from .piece import FingeringManager
from .piece import Piece
from .piece import Note
from instrument.instrument import Instrument
from instrument.instrument_string import InstrumentString
import json


class PieceLoader:

    def __init__(self, instrument: Instrument):
        self.instrument = instrument
        self.fingering_manager = FingeringManager(instrument)


    def load_piece(self, title: str, composer: str) -> Piece:
        """Returns the piece
        args:
            title: title of the piece
            composer: composer of the piece

        returns:
            piece: Piece object of the given piece, or raises exception
        """

        # in main: target note: if curr playing note = target note, target note = next note, switch visual indicator

        pieces_data_path = "piece/pieces.json"

        pieces_data = json.load(open(pieces_data_path, "r"))

        piece_data = None
        for piece in pieces_data:
            if piece['title'].lower() == title.lower() and piece['composer'].lower() == composer.lower():
                piece_data = piece
                break

        if not piece_data:
            raise Exception(f"Piece with title '{title}' and composer '{composer}' not found")
        
        fingering = self.fingering_manager.get_fingering(piece_data['midi_notes'])

        piece = Piece(fingering)

        return piece


if __name__ == "__main__":
    g_string = InstrumentString(196, 784)
    d_string = InstrumentString(293, 1175)
    a_string = InstrumentString(440, 1760)
    e_string = InstrumentString(659, 2637)

    violin_strings = [g_string, d_string, a_string, e_string]
    violin = Instrument(violin_strings, "instrument/Sonatina_Symphonic_Orchestra.sf2", preset=12, volume=120)

    loader = PieceLoader(violin)
    loader.load_piece("Maple Leaf Rag", "Scott Joplin")
