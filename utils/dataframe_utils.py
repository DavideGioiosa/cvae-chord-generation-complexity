import numpy as np
from sklearn.model_selection import train_test_split
from utils import constants


def split_train_test_validation(dataframe, val_size=0.1, test_size=0.2):
    """
    Split Dataset in train, validation, test
    stratify parameter allows to have a balanced complexity bins dataset
    """
    # Train - Test Split
    train_data, test_data, train_ground, test_ground = train_test_split(dataframe,
                                                                        dataframe,
                                                                        test_size=test_size,
                                                                        random_state=13,
                                                                        stratify=dataframe['Bin'])
    # Train - Validation Split

    train_data, valid_data, train_ground, valid_ground = train_test_split(train_data,
                                                                          train_data,
                                                                          test_size=val_size,
                                                                          random_state=13,
                                                                          stratify=train_data['Bin'])
    return train_data, valid_data, test_data


def get_chord_sequences_from_csv(dataframe):
    """
    Transform each chords sequence from list of strings (each one
    representing a chord) to list of np.array
    """
    all_sequences = dataframe.loc[:, 'Chord_1':'Chord_5'].values.tolist()

    data_sequences = []
    for seq in all_sequences:
        # single sequence of chords
        supp_chords_sequence = []
        for chord in seq:
            # eliminate [, ], "
            supp = chord.replace(']', '').replace('[', '')
            supp = supp.replace('"', '').split(" ")
            supp = np.array(supp, dtype=np.float32)
            supp_chords_sequence.append(supp)

        data_sequences.append(supp_chords_sequence)

    data_sequences = np.asarray(data_sequences)
    return data_sequences


def decrease_num_of_bins(dataset_orig, new_n_of_bins):
    """
    Decrease number of harmonic complexity bins,
    can divide for 2,3,5,6,10,15
    :param dataset_orig:
    :param new_n_of_bins: reduced number of complexity classes
    :return: dataset updated
    """
    dataset = dataset_orig.copy()

    # i.e. n_of_bins = 5 => bin_size = 6
    # every 6 classes correspond to a value
    bin_size = constants.COMPLEXITY_BINS // new_n_of_bins

    for i in range(new_n_of_bins):
        for j in range(bin_size):
            dataset.loc[dataset.Bin == (j + i * bin_size), 'Bin'] = i

    return dataset


def get_chord_sequences_with_complexity_bin(dataset, dataset_sequences, harmonic_complexity_bin):
    """
    return the list of chord sequences in the dataset with the given harmonic complexity value
    """
    indices = dataset.loc[dataset.Bin == harmonic_complexity_bin].index

    chord_sequences = []
    for i in indices:
        chord_sequences.append(dataset_sequences[i])

    return chord_sequences
