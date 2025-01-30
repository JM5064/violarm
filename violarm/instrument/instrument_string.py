
class InstrumentString:
    
    def __init__(self, min_freq: int, max_freq=2637, string_length=0.325, equal_tuning=False):
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


    def to_frequency(self, note_fraction: float) -> int:
        """Converts fractional note form to frequency
        args: 
            note_fraction: float

        returns:
            frequency: int
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

        
    def to_fraction(self, note_frequency: int) -> float:
        """Converts frequency to fractional note form
        args:
            note_frequency: int

        returns:
            fraction: float
        """

        if self.equal_tuning:
            return (note_frequency - self.min_freq) / (self.max_freq - self.min_freq)
        
        if 2 * note_frequency * self.string_length == 0:
            return None #
        
        if note_frequency == 0:
            return 1
        
        new_length = self.velocity / (2 * note_frequency)
        
        fraction = (self.string_length - new_length) / self.string_length

        return fraction

