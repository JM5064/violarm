import numpy as np


class Instrument:

    def __init__(self, strings):
        self.strings = strings
        self.num_strings = len(strings)


    def play_note(self, string, note):
        print(f'{note} played on {string} string')

