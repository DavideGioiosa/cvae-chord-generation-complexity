import numpy as np
import re


#########
# Defs file used for chord voicings in midi file creation
# from Di Giorgi et al. : A Data-Driven Model of Tonal Chord Sequence Complexity
# #######

# ***********
# definitions
# ***********

NATURALS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

PITCHES_DICT = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
              'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}

PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALE_SHORTHANDS = {'ionian': [0, 2, 4, 5, 7, 9, 11],
                    'dorian': [0, 2, 3, 5, 7, 9, 10],
                    'frigian': [0, 1, 3, 5, 7, 8, 10],
                    'lydian': [0, 2, 4, 6, 7, 9, 11],
                    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
                    'aeolian': [0, 2, 3, 5, 7, 8, 10],
                    'locrian': [0, 1, 3, 5, 6, 8, 10],
                    }

CHORD_SHORTHANDS = {'maj': [0, 4, 7],
                    'min': [0, 3, 7],
                    'dim': [0, 3, 6],
                    'aug': [0, 4, 8],
                    # sevenths
                    'maj7': [0, 4, 7, 11],
                    'min7': [0, 3, 7, 10],
                    '7': [0, 4, 7, 10],
                    'dim7': [0, 3, 6, 9],
                    'hdim7': [0, 3, 6, 10],
                    'minmaj7': [0, 3, 7, 11],
                    # sixths
                    'maj6': [0, 4, 7, 9],
                    'min6': [0, 3, 7, 9],
                    # ninth
                    '9': [0, 4, 7, 10, 14],
                    'maj9': [0, 4, 7, 11, 14],
                    'min9': [0, 3, 7, 10, 14],
                    # sus
                    'sus4': [0, 5, 7],
                    'sus2': [0, 2, 7],

                    '':      [0, 4, 7],
                    'm':     [0, 3, 7],
                    '7':     [0, 4, 7, 10],
                    '5':     [0, 7],
                    'm7':    [0, 3, 7, 10],
                    'maj7':  [0, 4, 7, 11],
                    'sus4':  [0, 5, 7],
                    'dim':   [0, 3, 6],
                    'aug':   [0, 4, 8],
                    '7sus4': [0, 5, 7, 10],
                    }

# *****
# Utils
# *****


def parse_note(note):
    """Parse a string and return the number of the pitch class [0,11] (C=0)."""
    if not isinstance(note, str):
        return ValueError('not a string')
    note_nat = note.replace('b', '').replace('#', '')
    if (note_nat not in NATURALS):
        raise ValueError('unrecognized natural' + note_nat)
    note = NATURALS[note_nat] + note.count('#') - note.count('b')
    note = note % 12
    return note


def split_note(string):
    """
    Split a string containing a note.

    Useful for
    - Pitches for splitting note/octave
    - Chords for splitting root/chord_shorthand
    - Scale for splitting root/scale_name
    """
    m = re.match('([A-G]#*b*)', string)
    note = m.group()
    remaining = string.replace(note, '').lstrip(' ').rstrip(' ')
    return note, remaining


def is_pitch_class_list(chord):
        is_list = isinstance(chord, list)
        is_nparray = isinstance(chord, np.ndarray)
        if(is_list or is_nparray):
            len_12 = (len(chord) < 12)
            range_12 = all([((i < 12) and (i >= 0)) for i in chord])
            if(len_12 and range_12):
                return True
        return False


def is_binary_pitch_class_list(chord):
    is_list = isinstance(chord, list)
    is_nparray = isinstance(chord, np.ndarray) and np.issubdtype(chord[0], int)
    if(is_list or is_nparray):
        len_12 = (len(chord) == 12)
        range_01 = all([i in [0, 1] for i in chord])
        if(len_12 and range_01):
            return True
    return False


def binary_pitch_class_list(pitch_class_list):
    if not is_pitch_class_list(pitch_class_list):
        raise ValueError("not a pitch class list")
    tones = [0 for i in range(12)]
    for i in pitch_class_list:
        tones[i] = 1
    return tones

# *******
# Classes
# *******


class Pitch:
    def __init__(self, pitch, A440=440):
        if hasattr(pitch, 'pitch_number'):
            self.initialize_from_pitch_number(pitch.pitch_number, A440=A440)
        elif(isinstance(pitch, str)):
            self.initialize_from_string(pitch, A440=A440)
        elif(isinstance(pitch, int) or np.issubdtype(pitch, int) or np.issubdtype(pitch, np.uint32)):
            self.initialize_from_pitch_number(pitch, A440=A440)
        else:
            raise ValueError("argument should be int or string")

    def initialize_from_pitch_number(self, pitch_number, A440=440):
        if(pitch_number < 0 or pitch_number > 127):
            raise ValueError("pitch_number is out of range")
        self.pitch_number = pitch_number
        self.pitch_class = int(self.pitch_number % 12)
        self.octave = int(self.pitch_number / 12) - 1
        self.freq = A440 * np.power(2, ((self.pitch_number - 69) / 12.0))

    def initialize_from_string(self, string, A440=440):
        note, octave = split_note(string)
        note = parse_note(note)
        octave = int(octave)
        pitch_number = (octave + 1) * 12 + note
        self.initialize_from_pitch_number(pitch_number, A440=A440)

    @staticmethod
    def from_freq(freq, A440=440):
        pitch_number = 12.0 * np.log2(freq / A440) + 69
        return Pitch(np.round(pitch_number), A440=A440)

    @property
    def name(self):
        return PITCH_CLASS_NAMES[self.pitch_class] + str(self.octave)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if(isinstance(other, int) or isinstance(other, np.int64)):
            return self.pitch_number == other
        return self.pitch_number == other.pitch_number

    def __lt__(self, other):
        return self.pitch_number < other.pitch_number

    def __add__(self, other):
        if(isinstance(other, int)):
            return Pitch(self.pitch_number + other)
        raise ValueError("not an int")

    def __sub__(self, other):
        if(isinstance(other, int)):
            return Pitch(self.pitch_number - other)
        if hasattr(other, 'pitch_number'):
            return self.pitch_number - other.pitch_number
        raise ValueError("not an int or Pitch")


class Chord:
    def __init__(self, chord):
        if(is_binary_pitch_class_list(chord)):
            self.initialize_from_tones(chord)
        elif(is_pitch_class_list(chord)):
            tones = binary_pitch_class_list(chord)
            self.initialize_from_tones(tones)
        elif(isinstance(chord, str)):
            self.initialize_from_string(chord)
        else:
            raise ValueError("argument should be list or string")

    def initialize_from_tones(self, tones):
        if(len(tones) != 12):
            raise ValueError("chord_tones length != 12")
        if(isinstance(tones, list)):
            tones = np.array(tones, dtype=np.int)
        self.tones = tones
        self.pitch_classes = np.nonzero(self.tones)[0]  #indici dove non sono 0

    def initialize_from_root_plus_intervals(self, root, intervals):
        if(isinstance(intervals, list)):
            intervals = np.array(intervals)
        tones = [0 for i in range(12)]
        for i in intervals:
            tones[(root + i) % 12] = 1
        self.initialize_from_tones(tones)

    def initialize_from_root_plus_shorthand(self, root, shorthand):
        root = parse_note(root)
        if (shorthand not in CHORD_SHORTHANDS):
            raise ValueError('not recognized shorthand')
        intervals = CHORD_SHORTHANDS[shorthand]
        self.initialize_from_root_plus_intervals(root, intervals)

    def initialize_from_string(self, string):
        root, shorthand = split_note(string)
        self.initialize_from_root_plus_shorthand(root, shorthand)

    def contains(self, pitch):
        pn = 0
        if isinstance(pitch, int):
            pn = pitch
        elif hasattr(pitch, 'pitch_number'):
            pn = pitch.pitch_number
        return self.tones[pn % 12] == 1

    def __eq__(self, other):
        if(np.all(self.tones == other.tones)):
            return True
        return False

    def __str__(self):
        out = ""
        for p in self.pitch_classes:
            out += PITCH_CLASS_NAMES[p] + " "
        return out


class Scale:
    """
    Parameters
    ----------
    scale_tones : numpy array drype=np.int shape=12
        binary array of scale tones
        example (CMajor) numpy.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    """
    def __init__(self, scale):
        if is_binary_pitch_class_list(scale):
            self.initialize_from_tones(scale)
        elif is_pitch_class_list(scale):
            self.initialize_from_pitch_class_list(scale)
        elif isinstance(scale, str):
            self.initialize_from_string(scale)
        else:
            raise ValueError("argument should be list or string")

    def initialize_from_tones(self, tones):
        if(isinstance(tones, list)):
            tones = np.array(tones, dtype=np.int)
        self.tones = tones
        self.pitch_classes = np.nonzero(self.tones)[0]

    def initialize_from_pitch_class_list(self, pitch_class_list):
        bpcl = binary_pitch_class_list(pitch_class_list)
        self.initialize_from_tones(bpcl)

    def initialize_from_root_plus_intervals(self, root, intervals):
        if(isinstance(intervals, list)):
            intervals = np.array(intervals)
        tones = np.zeros(12)
        for i in intervals:
            tones[(root + i) % 12] = 1
        self.initialize_from_tones(tones)

    def initialize_from_root_plus_shorthand(self, root, shorthand):
        if isinstance(root, str):
            root = parse_note(root)
        if (shorthand not in SCALE_SHORTHANDS):
            raise ValueError('not recognized shorthand')
        intervals = SCALE_SHORTHANDS[shorthand]
        self.initialize_from_root_plus_intervals(root, intervals)

    def initialize_from_string(self, string):
        root, shorthand = split_note(string)
        self.initialize_from_root_plus_shorthand(root, shorthand)

    def __str__(self):
        out = ""
        for p in self.pitch_classes:
            out += PITCH_CLASS_NAMES[p] + " "
        return out


class Voicing:
    def __init__(self, pitches):
        if hasattr(pitches[0], 'pitch_number'):
            self.initialize_from_list_of_pitches(pitches)
        else:
            self.initialize_from_list(pitches)

    def initialize_from_list_of_pitches(self, pitches):
        self.nvoices = len(pitches)
        self.pitches = sorted(pitches)
        self.compute_intervals()

    def initialize_from_list(self, in_list):
        pitches = []
        for elem in in_list:
            pitches.append(Pitch(elem))
        self.initialize_from_list_of_pitches(pitches)

    def compute_intervals(self):
        nout = int((self.nvoices * (self.nvoices - 1)) / 2)
        self.intervals = np.zeros(nout, dtype=np.int)
        ii = 0
        for d in range(1, self.nvoices):
            for i in range(self.nvoices - d):
                self.intervals[ii] = self.pitches[i + d] - self.pitches[i]
                ii += 1

    def __sub__(self, other):
        if(other.nvoices != self.nvoices):
            raise ValueError("number of voices does not match")
        out = np.zeros(self.nvoices, dtype=np.int)
        for i in range(self.nvoices):
            out[i] = self.pitches[i] - other.pitches[i]
        return out

    def to_string(self, i_bold):
        out = ""
        for i, p in enumerate(self.pitches):
            if(i == i_bold):
                out += "<"
            out += str(p)
            if(i == i_bold):
                out += ">"
            out += " "
        return out

    def __str__(self):
        out = ""
        for p in self.pitches:
            out += str(p) + " "
        return out
