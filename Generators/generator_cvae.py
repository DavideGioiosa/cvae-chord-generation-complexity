import numpy as np
import random
from utils import constants

from utils.clean_algorithm import get_cleaned_sequence

# Complexity classes have been reduced from 30 to 5
# constants.REDUCED_COMPLEXITY_CLASSES


class GeneratorCVAE:
    def __init__(self, decoder, latent_dim=2):
        self.decoder = decoder
        self.latent_dim = latent_dim

    def get_latent_vector(self, complexity_bin_one_hot, z):
        """
        Creation of the latent vector with shape [1, 7], input of the decoder:
        first 2 values are the latent space coordinates,
        the other 5 are the complexity in one hot encoding format

        :param complexity_bin_one_hot: harmonic complexity in one hot encoding format
        :param z: latent coord
        :return: latent vector [1, 7] for the decoder
        """

        out = np.zeros((1, self.latent_dim + constants.REDUCED_COMPLEXITY_BINS))
        out[:, complexity_bin_one_hot + 2] = 1.

        for i in range(len(z)):
            out[:, i] = z[i]
        return out

    def __decode(self, z_vec):
        """
        :param z_vec: latent space point + complexity in one hot
        :return: decoded and 'cleaned' sequence obtained from z
        """
        decoded = self.decoder.predict(z_vec)
        decoded = np.reshape(decoded, (constants.N_MEASURES, constants.N_PITCHES))

        # Cleaning with Cosine Distance
        decoded_z_clean = get_cleaned_sequence(decoded)

        return decoded_z_clean

    def generate_with_complexity(self, c_bin):
        """
        Generate chords sequence with c_bin harmonic complexity
        :param c_bin: harmonic complexity bin [0-4]
        :return: chords sequence with the indicated harmonic complexity
        """
        if c_bin < 0 or c_bin > 4:
            return "Complexity Range is not correct, allowed values: [0, 4]"

        z_x = round(random.uniform(-2.5, 2.5), 4)
        z_y = round(random.uniform(-2.5, 2.5), 4)
        z_ = [z_x, z_y]

        print("\nZ_: ", z_)
        vec = self.get_latent_vector(c_bin, z_)  # concat with complexity

        generated_seq = self.__decode(vec)

        return generated_seq

    def generate_with_complexity_and_z_coord(self, c_bin, z_x, z_y):
        """
        Generate chords sequence with the indicated harmonic complexity and z coords
        :param c_bin: complexity value [0-4]
        :param z_x: latent space coord
        :param z_y: latent space coord
        :return:
        """
        if c_bin < 0 or c_bin > 4:
            return "Complexity Range is not correct, allowed values: [0, 4]"

        if z_x < -5 or z_x > 5 or z_y < -5 or z_y > 5:
            print("Z values may be out of range and cause an invalid value")

        z_ = [z_x, z_y]
        vec = self.get_latent_vector(c_bin, z_)  # concat with complexity

        generated_seq = self.__decode(vec)

        return generated_seq
