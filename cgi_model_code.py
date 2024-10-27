# -*- coding: utf-8 -*-
"""cgi model code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZEN-ssRsmSvca3MpyJNWBNNOJnmxY47V
"""

pip install tensorflow matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Load the MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize the images to the range [-1, 1]
train_images = train_images.astype('float32') / 255.0
train_images = (train_images - 0.5) * 2  # Scale to [-1, 1]
train_images = np.expand_dims(train_images, axis=-1)  # Add channel dimension

print(train_images.shape)

def build_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(28 * 28 * 1, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

generator = build_generator()

# prompt: print the generator

generator.summary()

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(28, 28, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1, activation='sigmoid'))  # Output a probability
    return model

discriminator = build_discriminator()

# prompt: show the discriminatyor

discriminator.summary()

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create the GAN model
discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)

# Compile the GAN
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Set discriminator.trainable = False after compiling the GAN model
discriminator.trainable = False

def train_gan(epochs=10000, batch_size=128, save_interval=1000):
    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, train_images.shape[0], batch_size)
        real_images = train_images[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_images = generator.predict(noise)

        # Labels for real and fake images
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        print("Real images shape:", real_images.shape)
        print("Real images dtype:", real_images.dtype)
        print("Real labels shape:", real_labels.shape)
        print("Real labels dtype:", real_labels.dtype)
        print("Discriminator input shape:", discriminator.input_shape)

        # Train on real images
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        # Train on fake images
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)

        # Calculate the average discriminator loss
        d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_labels = np.array([1] * batch_size)  # All labels are real for the generator

        g_loss = gan.train_on_batch(noise, valid_labels)

               # Print the progress
        if epoch % save_interval == 0:
            _, accuracy = discriminator.evaluate(real_images, real_labels)
            print(f"{epoch} [D loss: {d_loss:.4f}, acc.: {accuracy}] [G loss: {g_loss[0]:.4f}]")  # Access the first element for g_loss
            save_generated_images(epoch)

def save_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, 0], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan_generated_epoch_{epoch}.png")
    plt.close()

# Start training the GAN
train_gan(epochs=10000, batch_size=128, save_interval=1000)