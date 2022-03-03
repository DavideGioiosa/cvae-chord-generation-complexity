import numpy as np
from utils import constants
import random

from utils.clean_algorithm import get_cleaned_sequence


class LatentSpaceManagerRVAE:
    def __init__(self, df, df_sequences, encoder, is_complexity_axis_vertical):
        """
        Provides methods for handling coordinates in the latent space of the Regression VAE
        :param df:
        :param df_sequences: chords sequences
        :param encoder: regression vae encoder
        :param is_complexity_axis_vertical: boolean indicating if the complexity axis is Horizontal or Vertical
        """
        self.df = df
        self.df_sequences = df_sequences
        self.encoder = encoder
        self.is_complexity_axis_vertical = is_complexity_axis_vertical
        self.z_complexity_axis_coord_per_bin_list = self.__get_z_coord()

    def __get_complexity_axis(self):
        if self.is_complexity_axis_vertical:
            return 1
        return 0

    def __get_z_coord(self):
        """
        :return: list of coord indicating the position on the complexity-axis for each complexity bin
        """
        z_coord_per_complexity_bin = []

        # Get coordinates ranges for each of the complexity classes on the C-axis
        # i.e. HC = 3 is in range [2,5] of the C-axis
        for c in range(constant.COMPLEXITY_BINS):
            c_val = np.array(c)
            indexes = self.df.loc[self.df['Bin'] == c_val].index
            c_val = np.reshape(c_val, (1, 1))

            # mean btw encoded sequences to get a trustable value
            encoded_coord_list = []
            # mean on X values, you can also delete this row and do a mean over all
            if len(indexes) > 10:
                indexes = indexes[:10]

            complexity_axis = self.__get_complexity_axis()
            for i in indexes:
                selected_seq = self.df_sequences[i].reshape(1, 60)
                # predict z coord using the encoder
                encoded_coord = self.encoder.predict([selected_seq, c_val])[2]  # [2]: z

                encoded_coord_list.append(encoded_coord[0][complexity_axis])

            mean_z_complexity_coord = round(sum(encoded_coord_list) / len(encoded_coord_list), 2)
            z_coord_per_complexity_bin.append(mean_z_complexity_coord)

        return z_coord_per_complexity_bin

    def get_complexity_bin_range(self, c_bin):
        return self.z_complexity_axis_coord_per_bin_list[c_bin]

    def get_complexity_bin_reduced_from_coord(self, z_complexity):
        """
        :param z_complexity: c-axis coordinate
        :return: the reduced harmonic complexity bin corresponding to the coord on the c-axis
        """

        is_c_axis_positive = self.z_complexity_axis_coord_per_bin_list[-1] > 0

        if is_c_axis_positive:
            for i, coord in enumerate(self.z_complexity_axis_coord_per_bin_list):
                if z_complexity <= coord:
                    if i == 0:
                        print("Z-complexity axis value may be out of range and cause an invalid value")
                    else:
                        print("Harmonic complexity reduced bin is ", i // constant.COMPLEXITY_RANGE)
                    return
        else:
            for i, coord in enumerate(self.z_complexity_axis_coord_per_bin_list):
                if z_complexity >= coord:
                    if i == 0:
                        print("Z-complexity axis value may be out of range and cause an invalid value")
                    else:
                        print("Harmonic complexity reduced bin is ", i // constant.COMPLEXITY_RANGE)
                    return

        print("Z complexity-axis value may be too high and cause an invalid value")
        return

    def get_range_of_complexity_bin_reduced(self, c_bin):
        """
        :param c_bin: complexity value [0-4]
        :return: c-axis range of complexity bin reduced
        """
        if c_bin < 0 or c_bin > 4:
            return "Complexity Range is not correct, allowed values: [0, 4]"
        return self.z_complexity_axis_coord_per_bin_list[c_bin * constant.COMPLEXITY_RANGE:
                                                    (c_bin + 1) * constant.COMPLEXITY_RANGE]


class GeneratorRVAE:
    def __init__(self, latent_space_manager: LatentSpaceManagerRVAE, decoder):
        self.latent_space_manager = latent_space_manager
        self.decoder = decoder

    def decode(self, z):
        """
        :param z: latent space point
        :return: decoded and cleaned sequence from z
        """
        z = np.reshape(z, [-1, 2])  # decoder input format
        decoded = self.decoder.predict(z)
        decoded = np.reshape(decoded, (constant.N_MEASURES, constant.N_PITCHES))

        # Cleaning Algorithm using Cosine Distance
        decoded_z_clean = get_cleaned_sequence(decoded)

        return decoded_z_clean

    def generate_with_complexity_reduced(self, c_bin):
        """
        Generation of chords sequence with c_bin complexity
        :param c_bin: harmonic complexity bin [0-4]
        :return: chords sequence with c_bin complexity
        """

        if c_bin < 0 or c_bin > 4:
            return "Complexity Range is not correct, allowed values: [0, 4]"

        c_range = self.latent_space_manager.get_range_of_complexity_bin_reduced(c_bin)
        print("Range on c-axis for the complexity bin (reduced) ", c_bin, ":")
        print(c_range)

        # get a random value in the range of the complexity bin (reduced) and for the free axis
        random_z_complexity_ax = round(random.uniform(c_range[0], c_range[-1]), 4)
        random_free_ax = round(random.uniform(-2.5, 2.5), 4)

        print("Z: ", random_z_complexity_ax, " ", random_free_ax)
        if self.latent_space_manager.is_complexity_axis_vertical:
            return self.decode([random_free_ax, random_z_complexity_ax])
        else:
            return self.decode([random_z_complexity_ax, random_free_ax])

    def generate_with_complexity_reduced_and_z_free(self, c_bin, z_free):
        """
        Generation of a chords sequence with c_bin complexity and setting the coordinate on the free axis
        :param c_bin: harmonic complexity bin [0-4]
        :param z_free: free-axis coordinate
        :return: chords sequence with c_bin complexity
        """
        if c_bin < 0 or c_bin > 4:
            return "Complexity Range is not correct, allowed values: [0, 4]"

        if z_free < -4 or z_free > 4:
            print("Z free-axis value may be too high")

        c_range = self.latent_space_manager.get_range_of_complexity_bin_reduced(c_bin)
        print("Range on c-axis for the complexity bin (reduced) ", c_bin, ":")
        print(c_range)
        # get random value in the range
        random_z_complexity_ax = round(random.uniform(c_range[0], c_range[-1]), 4)

        if self.latent_space_manager.is_complexity_axis_vertical:
            print("Z: ", z_free, " ", random_z_complexity_ax)
            return self.decode([z_free, random_z_complexity_ax])
        else:
            print("Z: ", random_z_complexity_ax, " ", z_free)
            return self.decode([random_z_complexity_ax, z_free])

    def generate_with_z_complexity_and_z_free(self, z_complexity, z_free):
        """
        Generation of a chords sequence from a point in latent space
        :param z_complexity: c-axis coordinate
        :param z_free: free-axis coordinate
        :return: chords sequence
        """
        if z_free < -4 or z_free > 4:
            print("Z-free axis value may be out of range and cause an invalid value")

        # Print the bin of reduced harmonic complexity corresponding to the coordinate on the c-axis
        self.latent_space_manager.get_complexity_bin_reduced_from_coord(z_complexity)

        if self.latent_space_manager.is_complexity_axis_vertical:
            print("Z: ", z_free, " ", z_complexity)
            return self.decode([z_free, z_complexity])
        else:
            print("Z: ", z_complexity, " ", z_free)
            return self.decode([z_complexity, z_free])
