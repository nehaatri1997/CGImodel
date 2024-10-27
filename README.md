# Pyhton-project-CGI model

This script defines a basic Generative Adversarial Network (GAN) to generate images similar to the MNIST dataset using TensorFlow and Keras. Here’s an overview of the workflow, along with the purpose of each major component:

# Data Preparation:
Loads the MNIST dataset and scales the images to a range of [-1, 1] (necessary for the tanh activation in the generator).
Adds a channel dimension to make the images compatible with the model.


# Building the Generator:
Takes a random noise vector (100-dimensional) as input and transforms it through dense layers with LeakyReLU activation and BatchNormalization.
Finally, reshapes the output to a 28x28 image format, matching the MNIST data structure.
The generator's output uses a tanh activation function to generate pixel values between [-1, 1].


# Building the Discriminator:
Receives a 28x28 image and flattens it into a dense layer structure, also using LeakyReLU.
Outputs a probability score indicating whether the image is real or generated (fake) using a sigmoid activation.


# Compiling and Training:
The discriminator is compiled with a binary cross-entropy loss, as it’s a binary classification model (real vs. fake).
In the GAN model, the discriminator is frozen (trainable=False) to focus on training the generator.
Training involves alternating between updating the discriminator and generator:
Discriminator: Trained on real images labeled as 1 and generated images labeled as 0.
Generator: Trained to generate realistic images to "fool" the discriminator, using labels set to 1 for all generated images.


# Training Loop:
For each epoch, random samples are used to train the discriminator and generator.
Every specified number of epochs, the discriminator’s accuracy is printed, and generated images are saved as visual checkpoints of progress.


# Save and Display Generated Images:
The save_generated_images function generates a new set of images from random noise and saves them for evaluation over time.


This code builds and trains a GAN model for generating MNIST-like images. You can adapt and expand this script to improve training stability and quality, such as by tweaking the generator and discriminator structures or adjusting hyperparameters like the learning rate and batch size.


# Overview
Briefly describe the project:

Objective: Train a Generative Adversarial Network (GAN) to generate images similar to MNIST digits.
Key components: Generator and Discriminator models trained using the MNIST dataset.


# Features
Customizable GAN model for image generation.
Real-time training output with saved generated samples every set number of epochs.
Visualizes discriminator accuracy and loss over training.


# Usage
Train the GAN: Run python gan_mnist.py (or your script's filename) to start training.
Customizing Parameters: Modify epochs, batch size, and save_interval within the train_gan() function.
