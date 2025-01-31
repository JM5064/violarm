
class InstrumentString:
    
    def __init__(self, min_freq: float, max_freq=2637, string_length=0.325, equal_tuning=False):
        self.min_freq = min_freq
        self.max_freq = max_freq
        if self.max_freq <= self.min_freq:
            self.max_freq = self.min_freq + 1

        self.string_length = string_length
        self.velocity = min_freq * 2 * string_length

        self.equal_tuning = equal_tuning


    @staticmethod
    def get_playing_note(notes: list[float]) -> float | None:
        '''Gets the highest note being pressed on the string
        args:
            notes: list[float] of note fractions or frequencies

        returns:
            highest_note: float
        '''
        
        if len(notes) == 0:
            return None
    
        return max(notes)


    def to_frequency(self, note_fraction: float) -> float:
        """Converts fractional note form to frequency
        args: 
            note_fraction: float

        returns:
            frequency: float
        """

        if self.equal_tuning:
            return round(self.min_freq + note_fraction * (self.max_freq - self.min_freq))
        
        new_length = self.string_length - self.string_length * note_fraction
        if new_length == 0:
            return self.max_freq
        
        # Standing wave equation
        frequency = round(self.velocity / (2 * new_length))

        # Threshold frequency to make sure it's not too high
        if frequency > self.max_freq:
            return self.max_freq

        return frequency

        
    def to_fraction(self, note_frequency: float) -> float:
        """Converts frequency to fractional note form
        args:
            note_frequency: float

        returns:
            fraction: float
        """

        if self.equal_tuning:
            return (note_frequency - self.min_freq) / (self.max_freq - self.min_freq)
                
        if note_frequency == 0:
            return 1
        
        new_length = self.velocity / (2 * note_frequency)
        
        fraction = (self.string_length - new_length) / self.string_length

        return fraction
    

    def calculate_fret_fractions(self) -> list[float]:
        frequencies = {
            "A4": 440,
            "A#4": 466.16,
            "B4": 493.88,
            "C5": 523.25,
            "C#5": 554.37,
            "D5": 587.33,
            "D#5": 622.25,
            "E5": 659.25,
            "F5": 698.46,
            "F#5": 739.99,
            "G5": 783.99, 
            "G#5": 830.61, 
            "A5": 880, 
            "A#5": 932.33,
            "B5:": 987.77,
            "C6": 1046.5, 
            "C#6": 1108.73, 
            "D6": 1174.66, 
            "D#6": 1244.51, 
            "E6": 1318.51, 
            "F6": 1396.91, 
            "F#6": 1479.98, 
            "G6": 1567.98, 
            "G#6": 1661.22, 
            "A6": 1760, 
            "A#6": 1864.66, 
            "B6": 1975.53
        }

        fractions = [self.to_fraction(freq) for freq in frequencies.values()]

        return fractions
