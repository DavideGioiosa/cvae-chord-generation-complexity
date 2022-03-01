"""
Conditional VAE: CVAE
"""

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.layers import concatenate as concat
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import constant

# Harmonic Complexity bins reduced from 30 to 5
# constant.REDUCED_COMPLEXITY_CLASSES


class ConditionalVAE:
    def __init__(self, latent_dim=2, intermediate_dim=512,
                 n_epoch=250, batch_size=32, bool_new_model=True):
        if bool_new_model:
            self.latent_dim = latent_dim
            self.intermediate_dim = intermediate_dim
            self.n_epoch = n_epoch
            self.batch_size = batch_size
            self.original_dim = constant.N_MEASURES * constant.N_PITCHES  # 5x12

            self.seq_input = Input(shape=(self.original_dim,), name="sequences_in")  # Flat chords sequence
            self.label_input = Input(shape=(constant.REDUCED_COMPLEXITY_BINS,), name="bin_in")  # One-hot encode

            self.encoder = self.__build_encoder()
            self.decoder = self.__build_decoder()
            self.conditional_vae = Model([self.seq_input, self.label_input], self.outputs)
            self.__set_loss()

        # else:
        # self.encoder = tf.keras.models.load_model("CVAE_final/cvae_encoder.h5")
        # self.decoder = tf.keras.models.load_model("CVAE_final/cvae_decoder.h5")
        # self.conditional_vae = tf.keras.models.load_model("CVAE_final/cvae_full.h5")

    def __sampling(self, args):
        z_mean, z_log_var = args
        eps = K.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.0)  # 0.1
        return z_mean + K.exp(0.5 * z_log_var) * eps

    def __build_encoder(self):
        # concatenation of chords sequence and complexity label
        inputs = concat([self.seq_input, self.label_input])

        encoder_h = Dense(self.intermediate_dim, activation='relu')(inputs)
        encoder_h = tf.keras.layers.Dropout(0.2)(encoder_h)
        encoder_h = Dense(self.intermediate_dim, activation='relu')(encoder_h)
        encoder_h = Dense(self.intermediate_dim // 4, activation='relu')(encoder_h)
        mu = Dense(self.latent_dim, activation='linear', name='z_mean')(encoder_h)
        l_sigma = Dense(self.latent_dim, activation='linear', name='z_log_var')(encoder_h)

        # Sampling
        z = Lambda(self.__sampling, output_shape=(self.latent_dim,), name='z')([mu, l_sigma])
        # CVAE: concatenate latent space coord z with complexity label
        self.zc = concat([z, self.label_input])

        self.__kl_loss(l_sigma, mu)

        encoder = Model([self.seq_input, self.label_input], [mu, l_sigma, z])

        return encoder

    def __build_decoder(self):
        decoder_hidden = Dense(self.intermediate_dim // 4, activation='relu')
        decoder_hidden2 = Dense(self.intermediate_dim, activation='relu')
        decoder_hidden3 = Dense(self.intermediate_dim, activation='relu')
        decoder_out = Dense(self.original_dim, activation='sigmoid')

        # input of the decoder is [latent space,complexity value]
        h_p = decoder_hidden(self.zc)
        h_p = decoder_hidden2(h_p)
        h_p = decoder_hidden3(h_p)
        self.outputs = decoder_out(h_p)

        # decoder
        d_in = Input(shape=(self.latent_dim + 5,))
        d_h = decoder_hidden(d_in)
        d_h = decoder_hidden2(d_h)
        d_h = decoder_hidden3(d_h)
        d_out = decoder_out(d_h)
        decoder = Model(d_in, d_out)

        return decoder

    def __kl_loss(self, l_sigma, mu):
        kl_loss = 1 + l_sigma - K.square(mu) - K.exp(l_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        self.kl_loss = kl_loss

    def __set_loss(self):
        reconstruction_loss = tf.keras.losses.binary_crossentropy(self.seq_input, self.outputs)
        reconstruction_loss *= self.original_dim

        cvae_loss = K.mean((reconstruction_loss + self.kl_loss))

        self.conditional_vae.add_loss(cvae_loss)
        self.conditional_vae.compile(optimizer='adam')

    def get_model_summary(self):
        self.conditional_vae.summary()

    def train(self, train_chords_sequences_flat, train_harmonic_cmplx_bins_one_hot,
              valid_chords_sequences_flat, valid_harmonic_cmplx_bins_one_hot):

        cvae_hist = self.conditional_vae.fit([train_chords_sequences_flat, train_harmonic_cmplx_bins_one_hot],
                                             shuffle=True,
                                             verbose=1, batch_size=32, epochs=self.n_epoch,
                                             validation_data=(
                                                 [valid_chords_sequences_flat, valid_harmonic_cmplx_bins_one_hot]),
                                             callbacks=[
                                                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
                                                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                                                      patience=8,
                                                                                      min_lr=0., verbose=1,
                                                                                      cooldown=5)])

    def plots(self, chord_sequences_flat, harmonic_complexity_bins_one_hot,
              harmonic_complexity_bins, data_subset_type):
        """
        Plot of the latent space
        :param chord_sequences_flat:
        :param harmonic_complexity_bins_one_hot: harmonic complexity in one hot
        :param harmonic_complexity_bins: harmonic complexity
        :param data_subset_type: String - Training or Test
        :return:
        """
        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.encoder.predict([chord_sequences_flat, harmonic_complexity_bins_one_hot])
        z_mean = np.asarray(z_mean)
        plt.figure(figsize=(7, 7))
        plt.title('Generators: ' + data_subset_type)
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=harmonic_complexity_bins)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()
