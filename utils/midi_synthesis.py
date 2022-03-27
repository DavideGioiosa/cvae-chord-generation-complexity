import itertools
import os
import sys

from utils.defs_music import *
from midiutil import MIDIFile
from utils import symbolic_format_converter


def get_chords_and_pc_bass(chords_sequence_piano_roll):
    """
    :param chords_sequence_piano_roll:
    :return: chords (chord name and type format) and root note of the chords
    """
    bass_notes = []
    chords_name_and_type = []
    ch_seq_symbolic = \
        symbolic_format_converter.get_chord_sequence_symbolic_from_piano_roll(chords_sequence_piano_roll)

    for ch_name_and_type in ch_seq_symbolic:
        chords_name_and_type.append(ch_name_and_type)

    ch_seq_symbolic_divided_by_name_type = \
        symbolic_format_converter.get_chords_sequence_symbolic_divided_by_name_type(ch_seq_symbolic)

    # root note of the chord
    for root in ch_seq_symbolic_divided_by_name_type[0::2]:
        bass_notes.append(PITCHES_DICT[root])
    return chords_name_and_type, bass_notes


def symbolic_2_midi(chords_sequence_piano_roll, midi_name):
    np.seterr(all='raise')
    pitch_range = [32, 100]
    chords, pc_bass = get_chords_and_pc_bass(chords_sequence_piano_roll)

    # i.e. Chord('Cmaj')
    chords_list = [Chord(chords[0]), Chord(chords[1]), Chord(chords[2]), Chord(chords[3]), Chord(chords[4])]

    voicings = voicings_from_chords_list(chords_list, pc_bass, pitch_range,
                                         n_voices=4, max_voicings_per_chord=200)

    events = []
    time = 0
    for i, v in enumerate(voicings):
        time = i * 4
        events.extend(arrgmt_sus1(v, time=time))
    events.append({'type': 'note', 'pitch': 0, 'time': time + 4, 'duration': 1, 'vel': 0})

    filename = midi_name + '.mid'

    create_midi(filename, 160, events)


# *****
# Chords Voicings from Di Giorgi et al. :
# A Data-Driven Model of Tonal Chord Sequence Complexity
# *****

def create_midi(filename, bpm, events, reverb=64, path='./Generated_outputs'):
    """
    events = [event1, ..., eventN]
    where each event is a dict
    {"type": 'note', "pitch": MIDI_pitch, "time": time_quarters,
    "duration": duration_quarters, "vel": velocity[0-127]}
    {"type": 'cc', "time":time_quarters, "cc":ccnum, "val":value[0, 127]}
    :param path: where the .mid files will be saved
    :param filename: name of the midi file
    :param bpm: Beats Per Minute
    :param events: MIDI events
    :param reverb:
    :return:
    """

    # Create the MIDIFile Object with 1 track
    MyMIDI = MIDIFile(1)

    # Add track name and tempo.
    MyMIDI.addTrackName(track=0, time=0, trackName="Piano")
    MyMIDI.addTempo(track=0, time=0, tempo=bpm)

    MyMIDI.addControllerEvent(track=0, channel=0, time=0, controller_number=91, parameter=reverb)

    for event in events:
        if event['type'] == 'note':
            MyMIDI.addNote(track=0, channel=0,
                           pitch=event['pitch'], time=event['time'],
                           duration=event['duration'], volume=event['vel'])
        """
        if event['type'] == 'cc':
            MyMIDI.addControllerEvent(track=0, channel=0,
                                      time=event['time'],
                                      eventType=event['cc'], parameter1=event['val'])
        """

    # path where the .mid files will be saved
    with open(os.path.join(path,
                           filename), "wb") as output_file:
        MyMIDI.writeFile(output_file)

    print("Midi Created")
    return


def pc_in_range(pc_list, pitch_range):
    out = []
    for p in range(pitch_range[0], pitch_range[1]):  # for each octave
        pc = int(p % 12)
        if pc in pc_list:
            out.append(p)
    return out


def voicings_in_range(chord, pc_bass, n_upper_voices, pitch_range, verbose=True):

    if verbose: print('generating voicings for ', str(chord), '...', end=""); sys.stdout.flush()
    out = []
    for p_bass in pc_in_range([pc_bass], pitch_range):
        p_list = pc_in_range(chord.pitch_classes, [p_bass + 1, pitch_range[1]])  # list of candidate pitches
        voicings_ = list(itertools.combinations(p_list, n_upper_voices))
        voicings_ = [(p_bass,) + v for v in voicings_]
        out.extend(voicings_)

    if verbose: print(len(out), 'voicings found')
    out = [Voicing(o) for o in out]
    return out


def prune_voicings(voicings, chord, ev, nkeep=100):
    evs = np.zeros(len(voicings))
    for i, v in enumerate(voicings):
        evs[i] = ev(v, chord)
    out = []
    idx = np.argsort(evs)[::-1]
    for i in idx[:nkeep]:
        out.append(voicings[i])
    return out


class VoicingEv:
    def __call__(self, voicing, chord):
        raise NotImplemented()


class VoicingTransEv:
    def __call__(self, voicing1, voicing2):
        raise NotImplemented()


class VoicingEvComposite(VoicingEv):
    def __init__(self, voicing_ev_list, weights):
        self.voicing_ev_list = voicing_ev_list
        self.weights = weights

    def __call__(self, voicing, chord):
        out = 0
        for w, v in zip(self.weights, self.voicing_ev_list):
            out += w * v(voicing, chord)
        return out


class VoicingTransEvComposite(VoicingTransEv):
    def __init__(self, voicing_trans_ev_list, weights):
        self.voicing_trans_ev_list = voicing_trans_ev_list
        self.weights = weights

    def __call__(self, voicing1, voicing2):
        out = 0
        for w, v in zip(self.weights, self.voicing_trans_ev_list):
            out += w * v(voicing1, voicing2)
        return out


class VoicingEvIntervals(VoicingEv):
    # intervals should decrease from lower to upper voices
    def __call__(self, voicing, chord):
        return voicing.intervals[0] - np.max(voicing.intervals[:voicing.nvoices - 1])


class VoicingEvChordInVoicing(VoicingEv):
    # how many chord tones are in the voicing
    def __call__(self, voicing, chord):
        out = 0
        pc = [p.pitch_class for p in voicing.pitches]
        for ct in chord.pitch_classes:
            if (ct in pc):
                out += 1
        return out


class VoicingTransEvDistance(VoicingTransEv):
    # sum of the distance between the relative voices
    def __call__(self, voicing1, voicing2):
        out = 0
        for v1, v2 in zip(voicing1.pitches, voicing2.pitches):
            out -= abs(v1 - v2)
        return out


class VoicingTransEvDistanceSoprano(VoicingTransEv):
    def __call__(self, voicing1, voicing2):
        return -abs(max(voicing1.pitches) - max(voicing2.pitches))


def log_dist(x, eps=1e-12):
    x -= np.min(x)
    if (np.sum(x) == 0):
        return np.log(np.ones(len(x)) * (1.0 / len(x)))
    x = x / np.sum(x) + eps
    return np.log(x)


def log_dist_mat(X, eps=1e-12):
    for i in range(X.shape[0]):
        X[i, :] -= np.min(X[i, :])
        X[i, :] = X[i, :] / np.sum(X[i, :]) + eps
        X[i, :] = np.log(X[i, :])
    return X


def viterbi(obs, trans):
    """
    obs and trans are 2 dictionaries
    obs[t] is a np array containing log distribution
    trans[t] is a np matrix containing log state transitions
    """
    assert (len(obs) == len(trans) + 1)
    T = len(obs)

    n_states = len(obs[0])
    splog = np.log(1 / n_states) * np.ones(n_states)

    T1 = {}
    T2 = {}

    # time 0 : initialization, first column of T1
    T1[0] = splog + obs[0]

    # time > 0
    for t in range(1, T):
        n_states = len(obs[t])
        T1[t] = np.zeros(n_states)
        T2[t] = np.zeros(n_states, dtype=np.int)
        for st in range(n_states):
            prob = T1[t - 1] + trans[t][:, st]
            maxi = np.argmax(prob)

            T2[t][st] = maxi
            T1[t][st] = prob[maxi] + obs[t][st]

    # best last state
    x = np.zeros(T, dtype=np.int)
    x[T - 1] = np.argmax(T1[T - 1])
    for t in range(T - 2, -1, -1):
        x[t] = T2[t + 1][x[t + 1]]
    return x


def voicings_from_chords_list(chords, pc_bass, pitch_range,
                              n_voices=3, max_voicings_per_chord=100, verbose=True):
    ev1 = VoicingEvIntervals()
    ev2 = VoicingEvChordInVoicing()
    ev_obs = VoicingEvComposite([ev1, ev2], [1, 2])

    ev1 = VoicingTransEvDistance()
    ev2 = VoicingTransEvDistanceSoprano()
    ev_trans = VoicingEvComposite([ev1, ev2], [1, 2])

    # num of chords in the sequence
    T = len(chords)

    voicings = []
    for c, b in zip(chords, pc_bass):
        v_ = voicings_in_range(c, b, n_voices, pitch_range, verbose=verbose)
        if len(v_) > max_voicings_per_chord:
            v_ = prune_voicings(v_, c, ev_obs, max_voicings_per_chord)
        voicings.append(v_)

    #for s in voicings:
    #    for v in s:
    #        print(v)

    if verbose: print('computing observations', end=""); sys.stdout.flush()
    obs = {}
    for i, c in enumerate(chords):
        if verbose: print(".", end=""); sys.stdout.flush()
        dist = []
        for v in voicings[i]:
            dist.append(ev_obs(v, c))
        obs[i] = log_dist(np.array(dist))
    if verbose: print(); sys.stdout.flush()

    if verbose: print('computing transitions', end=""); sys.stdout.flush()
    trans = {}
    for t in range(1, T):
        if verbose: print(".", end=""); sys.stdout.flush()
        trans[t] = np.zeros((len(voicings[t - 1]), len(voicings[t])))
        for i1, v1 in enumerate(voicings[t - 1]):
            for i2, v2 in enumerate(voicings[t]):
                trans[t][i1, i2] = ev_trans(v1, v2)
        trans[t] = log_dist_mat(trans[t])
    if (verbose): print(); sys.stdout.flush()

    x = viterbi(obs, trans)

    out = []
    for t in range(T):
        out.append(voicings[t][x[t]])

    return out


def random_vel(mean_vel=64, std_vel=8):
    vel = int(np.random.normal(mean_vel, std_vel))
    vel = min(max(vel, 0), 127)
    return vel


def arrgmt_sus1(voicing, mean_vel=64, std_vel=3, time=0):
    out = []
    for p in voicing.pitches:
        vel = random_vel(mean_vel, std_vel)
        out.append({'type': 'note', 'pitch': p.pitch_number, 'time': time, 'duration': 3.8, 'vel': vel})
    return out


def arrgmt_arp1(voicing, pitch_pattern, vel_pattern, std_vel=3, time=0, eps=.01):
    out = []

    n_voices = len(voicing.pitches)

    out.append({'type': 'cc', 'time': time + eps, 'cc': 64, 'val': 64})
    bass = min(voicing.pitches)
    vel = random_vel(max(vel_pattern), std_vel)
    out.append({'type': 'note', 'pitch': bass.pitch_number, 'time': time, 'duration': 4, 'vel': vel})

    pitches = []
    for p in voicing.pitches:
        if (p == bass): continue
        pitches.append(p)

    pitches = sorted(pitches)

    duration = 4 / len(pitch_pattern)
    for i, (p, v) in enumerate(zip(pitch_pattern, vel_pattern)):
        vel = random_vel(v, std_vel)
        out.append({'type': 'note', 'pitch': pitches[p].pitch_number, 'time': time + i * duration, 'duration': duration,
                    'vel': vel})

    out.append({'type': 'cc', 'time': time + 4 - eps, 'cc': 64, 'val': 0})

    return out

