import numpy as np
import re

from utils.clean_algorithm import CHORDS_PR_DICT


def get_chord_symbolic_from_piano_roll(piano_roll_chord):
    for chord, pr_format in CHORDS_PR_DICT.items():
        if np.array_equal(piano_roll_chord, pr_format):
            return chord

    print("Chord doesn't exist")
    return None


def get_chord_sequence_symbolic_from_piano_roll(piano_roll_sequence):
    """
    Converts chords sequence from piano roll format to symbolic one
    """
    chords_sequence_symbolic = []
    for ch in piano_roll_sequence:
        ch_symbolic = get_chord_symbolic_from_piano_roll(ch)
        if ch_symbolic:
            chords_sequence_symbolic.append(ch_symbolic)
        else:
            print("Error: chord in sequence doesn't exist")
            return

    return chords_sequence_symbolic


def get_chords_sequence_symbolic_divided_by_name_type(symbolic_sequence):
    """
    Splits chords sequence from symbolic roll format in chords and types
    """
    regex_ch_type = 'min|maj|7|5'
    symbolic_sequence_name_type = []

    for ch in symbolic_sequence:
        ch_name = re.split(regex_ch_type, ch)[0]
        ch_type = re.split(ch_name, ch)[-1]
        symbolic_sequence_name_type.extend((ch_name, ch_type))

    return symbolic_sequence_name_type
