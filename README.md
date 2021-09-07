<p align="center"><img width="200" src="https://camo.githubusercontent.com/7bc40fff3c166508ead40e4ec7354e4d616bdd1a7733941d2ae75a154668958d/68747470733a2f2f63646e2e737461746963616c6c792e696f2f696d672f706e67696d6167652e6e65742f77702d636f6e74656e742f75706c6f6164732f323031382f30362f706f6c697465636e69636f2d6d696c616e6f2d6c6f676f2d706e672d352e706e67"></p>

# Modeling Harmonic Complexity in Automatic Music Generation using Conditional Variational Autoencoders

Author: Davide Gioiosa

## Goal 

Is it possible to use complexity as a parameter to automatically generate music? This is the question that motivates our research. In the area of automatic music composition, several neural network models have been implemented to generate music of a certain musical genre, e.g. rock, pop, jazz, or to capture and imitate the style of a composer.
Recent studies in this area of research, focus on providing the ability not only to generate music, but also to be able to condition the creative process.

From previous researches we know that complexity is a parameter closely related to the amount of brain activity of the listener (the so-called "arousal potential"). It also affects a person's musical preferences. Given this close correlation with a listener's perceptions, we decide to explore the use of this parameter in music.

Complexity is present in each of the aspects in which the music can be divided, e.g. chords, rhythm, melody, etc. Among these we choose to focus on the harmony. In particular, in this work we explore **Harmonic Complexity** and its use as a parameter to condition the generation of chord sequences. 
For the automatic generation process we exploit two conditional neural network models both based on the Variational Autoencoder. We evaluated, through a perceptual test, the ability to generate chord sequences give a desired complexity values. 

## Dataset

The starting dataset used for this experimental thesis comes from this research: https://www.researchgate.net/publication/320029367_A_Data-Driven_Model_of_Tonal_Chord_Sequence_Complexity, containing 5-chord sequences associated with a complexity bin.

## Conditional Variational Autoencoder
The Conditional Variational Autoencoder (CVAE) is an extension of the VAE model and it's a type of Conditional Architectures, which are networks characterized by the addition of the conditioning feature as an additional input layer to the network model. This type of model provide the ability to have a control over the data during the generation process through the conditioning with the target feature.
<p align="center"><img width="500" src="https://github.com/DavideGioiosa/master-thesis-polimi/blob/main/Img/Conditional_VAE.png"></p>
We implemented two different Conditional Architectures with **Python** using **Tensorflow-Keras**.

### Model A
This first model of CVAE incorporates the conditioning information by concatenating the layer at the input of both the encoder.
<p align="center"><img width="500" src="https://github.com/DavideGioiosa/master-thesis-polimi/blob/main/Img/CVAE_1.png"></p>

### Model B
This second model of CVAE is composed by the combination of the standard VAE with a Regressor, which has as input the complexity value that explicitly conditions the latent representation z of the data X.
<p align="center"><img width="500" src="https://github.com/DavideGioiosa/master-thesis-polimi/blob/main/Img/CVAE_2.png"></p>

With this model we obtained a disentangled-dimension in the latent space that models the harmonic complexity feature. 
<p align="center"><img width="500" src="https://github.com/DavideGioiosa/master-thesis-polimi/blob/main/Img/CVAE_2_latent_space.png"></p>

### Generation of new chord progressions
<p align="center"><img width="200" src="https://github.com/DavideGioiosa/master-thesis-polimi/blob/main/Img/CVAEs_sampling.png"></p>

### Listening Test
A web-app has been designed using **Flask** and **AWS** to collect ratings on the generated sequences. In the fist part of the experiment, the participants are profied based on their music background using the self-report questionnaire of the _Goldsmiths Musical Sophistication Index (https://www.gold.ac.uk/music-mind-brain/gold-msi/)_. The second part is the perceptual test in which the participants were asked to express their level of agreement to the indicated complexity value provided for each chord progressions. The evaluation is expressed using the Likert scale scores from 0 to 4, where completely agree is the highest score and completely disagree the lowest one.



More detailed information about the study can be found in the thesis.
