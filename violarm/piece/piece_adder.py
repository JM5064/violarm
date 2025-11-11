from music21 import converter, note, chord
import jsonyx as json
    
    
def add_piece():
    title = input("Enter the title of the piece: ")
    composer = input("Enter the composer's full name: ")
    path = input("Enter the path to the midi file: ")

    midi_notes = extract_notes_from_midi(path)

    piece_data = {
        'title': title,
        'composer': composer,
        'midi_notes': midi_notes
    }

    save_data(piece_data)


def extract_notes_from_midi(recording_path):
    """Takes a midi file and returns the topmost notes of the first part in midi format
    args:
        None

    returns:
        list[int] of midi notes
    """

    score = converter.parse(recording_path)
    notes = []

    # Get the measures of the first part of the score
    measures = score.parts[0].getElementsByClass('Stream')


    # Add each note to notes
    for measure in measures:
        for item in measure:
            if isinstance(item, note.Note):
                notes.append(item)

            # For chords, append only the top note
            if isinstance(item, chord.Chord):
                notes.append(max(item.notes))

    midi_notes = []
    # Remove extra notes from ties and convert to midi
    i = 0
    while i < len(notes):
        # Append notes which are not a tie, or are the last note of a tie
        if notes[i].tie is None or i == len(notes) - 1 or notes[i+1].tie is None:
            midi_notes.append(notes[i].pitch.midi)

        i += 1

    return midi_notes


def save_data(data):
    """Saves data json to pieces data json file"""
    
    pieces_data_path = "violarm/piece/pieces.json"

    pieces_data = json.load(open(pieces_data_path, "r"))
    pieces_data.append(data)
    
    with open(pieces_data_path, 'w') as file:
        json.dump(pieces_data, file, indent=2, indent_leaves=False)


if __name__ == "__main__":
    add_piece()