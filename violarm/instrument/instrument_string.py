
class InstrumentString:
    
    def __init__(self, min_freq: int, max_freq=2637, string_length=0.325, equal_tuning=False):
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.string_length = string_length
        self.velocity = min_freq * 2 * string_length

        self.equal_tuning = equal_tuning


    def get_playing_note(self, notes: list[int]) -> int:
        '''Gets the highest note being pressed on the string
        args:
            notes: list[int] of note fractions or frequencies

        returns:
            highest_note: int 
        '''
        
        if len(notes) == 0:
            return
    
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
        
        frequency = round(self.velocity / (2 * new_length))

        if frequency > self.max_freq:
            return self.max_freq

        return frequency

        
