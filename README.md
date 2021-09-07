<p align="center"><img width="200" src="https://camo.githubusercontent.com/7bc40fff3c166508ead40e4ec7354e4d616bdd1a7733941d2ae75a154668958d/68747470733a2f2f63646e2e737461746963616c6c792e696f2f696d672f706e67696d6167652e6e65742f77702d636f6e74656e742f75706c6f6164732f323031382f30362f706f6c697465636e69636f2d6d696c616e6f2d6c6f676f2d706e672d352e706e67"></p>

# Modeling Harmonic Complexity in Automatic Music Generation using Conditional Variational Autoencoders

07/04/2021

Davide Gioiosa

# Goal 

Is it possible to use complexity as a parameter to automatically generate music? This is the question that motivates our research. In the area of automatic music composition, several neural network models have been implemented to generate music of a certain musical genre, e.g. rock, pop, jazz, or to capture and imitate the style of a composer.
Recent studies in this area of research, focus on providing the ability not only to generate music, but also to be able to condition the creative process.

From previous researches we know that complexity is a parameter closely related to the amount of brain activity of the listener (the so-called "arousal potential"). It also affects a person's musical preferences. Given this close correlation with a listener's perceptions, we decide to explore the use of this parameter in music.

Complexity is present in each of the aspects in which the music can be divided, e.g. chords, rhythm, melody, etc. Among these we choose to focus on the harmony. In particular, in this work we explore harmonic complexity and its use as a parameter to condition the generation of chord sequences. 
For the automatic generation process we exploit two conditional neural network models both based on the Variational Autoencoder. We evaluated, through a perceptual test, the ability to generate chord sequences give a desired complexity values. 

# Conditional Variational Autoencoder
![Prova](master-thesis-polimi/Img/Conditional_VAE.png)

# Model 1

# Model 2
