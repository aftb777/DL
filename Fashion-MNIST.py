# Question:
# Build a Generative Adversarial Network (GAN) using MNIST.
# Plot generator and discriminator losses and display generated images
# during training.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Sequential

# 1. Load MNIST Dataset
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# 2. Normalize Images to [-1, 1]
X_train = X_train.astype("float32")
X_train = (X_train - 127.5) / 127.5
X_train = X_train.reshape(-1, 784)

# 3. Build Generator
generator = Sequential([
    layers.Input(shape=(100,)),
    layers.Dense(256),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dense(512),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dense(784, activation="tanh")
])

# 4. Build Discriminator
discriminator = Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dense(256),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dense(1, activation="sigmoid")
])

# 5. Compile Discriminator
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 6. Build Combined GAN
discriminator.trainable = False

gan = Sequential([generator, discriminator])

gan.compile(
    optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
    loss="binary_crossentropy"
)

# 7. Training Parameters
epochs = 10
batch_size = 128
half_batch = batch_size // 2

d_losses = []
g_losses = []

# 8. Train GAN
for epoch in range(epochs):

    # Train Discriminator on real and generated images
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_images = X_train[idx]

    noise = np.random.normal(0, 1, (half_batch, 100))
    fake_images = generator.predict(noise, verbose=0)

    real_labels = np.ones((half_batch, 1))
    fake_labels = np.zeros((half_batch, 1))

    discriminator.trainable = True

    d_loss_real = discriminator.train_on_batch(
        real_images, real_labels
    )

    d_loss_fake = discriminator.train_on_batch(
        fake_images, fake_labels
    )

    d_loss = (float(d_loss_real[0]) + float(d_loss_fake[0])) / 2

    # Train Generator through the combined GAN
    discriminator.trainable = False

    noise = np.random.normal(0, 1, (batch_size, 100))
    misleading_labels = np.ones((batch_size, 1))

    g_loss = gan.train_on_batch(
        noise, misleading_labels
    )

    d_losses.append(d_loss)
    g_losses.append(float(g_loss))

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"D Loss: {d_loss:.4f} | G Loss: {float(g_loss):.4f}"
    )

    # Display generated images
    noise = np.random.normal(0, 1, (10, 100))
    generated_images = generator.predict(noise, verbose=0)

    plt.figure(figsize=(10, 2))

    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(
            generated_images[i].reshape(28, 28),
            cmap="gray",
            vmin=-1,
            vmax=1
        )
        plt.axis("off")

    plt.suptitle(f"Generated Images - Epoch {epoch + 1}")
    plt.show()

# 9. Plot Generator and Discriminator Losses
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss")
plt.legend()
plt.show()