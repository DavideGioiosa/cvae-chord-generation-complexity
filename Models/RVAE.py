"""
Regression VAE
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import constant


class RegressionVAE:
    def __init__(self, latent_dim=2, intermediate_dim=512,
                 n_epochs=250, bool_new_model=True):
        if bool_new_model:
            self.latent_dim = latent_dim
            self.intermediate_dim = intermediate_dim
            self.n_epochs = n_epochs
            self.original_dim = constant.N_MEASURES * constant.N_PITCHES  # 5x12

            self.input_sequence = tf.keras.Input(shape=(self.original_dim,), name='input_sequence')  # flat input
            self.inputs_r = tf.keras.Input(shape=(1,), name='ground_truth')  # harmonic complexity val

            self.encoder = self.__build_encoder()
            self.decoder = self.__build_decoder()

            # instantiate VAE model
            self.decoder_outputs = self.decoder(self.encoder([self.input_sequence, self.inputs_r])[2])  # [2]: z
            self.regression_vae = tf.keras.Model([self.input_sequence, self.inputs_r], self.decoder_outputs)
            self.__set_loss()

        # Load trained model
        # else:
        #    self.encoder = keras.models.load_model(constant.RVAE_ENCODER_PATH)
        #    self.decoder = keras.models.load_model(constant.RVAE_DECODER_PATH)
        #    self.regression_vae = keras.models.load_model(constant.RVAE_FULL_PATH)

    def __sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], self.latent_dim),
                                                 mean=0., stddev=1.0)
        return z_mean + tf.keras.backend.exp(0.5 * z_log_sigma) * epsilon

    def __build_encoder(self):
        x = tf.keras.layers.Dense(self.intermediate_dim, activation="relu")(self.input_sequence)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(self.intermediate_dim / 2, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(self.intermediate_dim / 8, activation="relu")(x)

        # q(z|x)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name="z_log_var")(x)

        # posterior on Y; probabilistic Regressor
        r_mean = tf.keras.layers.Dense(1, name='r_mean')(x)
        r_log_var = tf.keras.layers.Dense(1, name='r_log_var')(x)

        # z = self.__sampling()([z_mean, z_log_var])
        z = tf.keras.layers.Lambda(self.__sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        r = tf.keras.layers.Lambda(self.__sampling, output_shape=(1,), name='r')([r_mean, r_log_var])

        # latent generator: Mean and st deviation for the predicted complexity value
        pz_mean = tf.keras.layers.Dense(self.latent_dim, name='pz_mean',
                                        kernel_constraint=tf.keras.constraints.unit_norm())(r)
        # pz_log_var = Dense(1, name='pz_log_var')(r)

        self.__kl_loss(z_mean, pz_mean, z_log_var)
        self.__label_loss(r_mean, r_log_var)

        encoder = tf.keras.Model([self.input_sequence, self.inputs_r],
                                 [z_mean, z_log_var, z, r_mean, r_log_var, r, pz_mean],
                                 name='encoder')
        encoder.summary()

        return encoder

    def __build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(self.intermediate_dim / 8, activation="relu")(latent_inputs)
        x = tf.keras.layers.Dense(self.intermediate_dim / 2, activation="relu")(x)
        x = tf.keras.layers.Dense(self.intermediate_dim, activation="relu")(x)
        decoder_outputs = tf.keras.layers.Dense(self.original_dim, activation="sigmoid")(x)

        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        return decoder

    def get_model_summary(self):
        self.regression_vae.summary()

    def __kl_loss(self, z_mean, pz_mean, z_log_var):
        kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean - pz_mean) - tf.keras.backend.exp(z_log_var)
        # kl_loss = 1 + z_log_var - pz_log_var - K.tf.divide(K.square(z_mean-pz_mean),K.exp(pz_log_var)) - K.tf.divide(K.exp(z_log_var),K.exp(pz_log_var))
        kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        self.kl_loss = kl_loss

    def __label_loss(self, r_mean, r_log_var):
        self.label_loss = tf.divide(0.5 * tf.keras.backend.square(r_mean - self.inputs_r),
                                    tf.keras.backend.exp(r_log_var)) + 0.5 * r_log_var

    def __set_loss(self):
        reconstruction_loss = tf.keras.losses.binary_crossentropy(self.input_sequence, self.decoder_outputs)
        reconstruction_loss *= self.original_dim

        # total loss
        vae_loss = tf.keras.backend.mean(reconstruction_loss + self.kl_loss + self.label_loss)
        self.regression_vae.add_loss(vae_loss)
        self.regression_vae.compile(optimizer='adam')

    def train(self, train_chords_sequences_flat, train_harmonic_cmplx_bins,
              valid_chords_sequences_flat, valid_harmonic_cmplx_bins):
        rvae_h = self.regression_vae.fit([train_chords_sequences_flat, np.array(train_harmonic_cmplx_bins)],
                                         epochs=self.n_epochs,
                                         batch_size=32,
                                         validation_data=[valid_chords_sequences_flat,
                                                          np.array(valid_harmonic_cmplx_bins)],
                                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30),
                                                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                                                         patience=10,
                                                                                         min_lr=0.0001,
                                                                                         verbose=1, cooldown=5)])

    def plots(self, chord_sequences_flat, harmonic_complexity_bins, data_subset_type):
        """
        display a 2D plot of the Harmonic Complexity classes in the latent space
        :param chord_sequences_flat:
        :param harmonic_complexity_bins:
        :param data_subset_type: String - Training or Test
        :return:
        """

        [z_mean, z_log_var, z, r_mean, r_log_var, r_vae, pz_mean] = \
            self.encoder.predict([chord_sequences_flat, harmonic_complexity_bins])

        plt.figure(figsize=(9, 9))
        plt.title('Generators: ' + data_subset_type + ', Labels: Bin')
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=harmonic_complexity_bins)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()
