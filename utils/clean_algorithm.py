import numpy as np
import constant

NOTE_INDEX = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
              'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}

CHORD_TYPES = {'maj': [0, 4, 7],
               'min': [0, 3, 7],
               '5': [0, 7],
               '7': [0, 4, 7, 10]}

"""
Dictionary of chords:
    :key: chord, pitch + type
    :value: Piano Roll format
"""
CHORDS_PR_DICT = {}
for note_name in NOTE_INDEX:
    for ch_type in list(CHORD_TYPES.keys()):
        chord_name = note_name + ch_type
        chord_piano_roll = np.zeros(12)
        for i in CHORD_TYPES[ch_type]:
            index_root = (i + NOTE_INDEX[note_name]) % 12
            chord_piano_roll[index_root] = 1
        CHORDS_PR_DICT[chord_name] = chord_piano_roll


def cos_similarity(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)

    return cos


def get_most_similar_chord(decoded_chord):
    """
    :param decoded_chord: decoded chord in raw format
    :return: valid chord using cosine similarity
    """
    greatest_similarity = 0
    most_similar = None

    for chord in CHORDS_PR_DICT.keys():
        similarity = cos_similarity(decoded_chord, CHORDS_PR_DICT[chord])

        if similarity > greatest_similarity:
            greatest_similarity = similarity
            most_similar = chord

    # print(f'Most similar: {most_similar}. Similarity: {greatest_similarity}')

    return most_similar


def get_cleaned_sequence(raw_decoded_sequence):
    """
    :param raw_decoded_sequence: decoded chords sequence in raw format
    :return: cleaned chords sequence with valid chords
    """
    cleaned_sequence = []
    for chord in raw_decoded_sequence:
        cleaned_chord = get_most_similar_chord(chord)
        cleaned_chord_PR = CHORDS_PR_DICT[cleaned_chord]
        cleaned_sequence.append(cleaned_chord_PR)

    cleaned_sequence = np.reshape(cleaned_sequence, (constant.N_MEASURES, constant.N_PITCHES))

    return cleaned_sequence
