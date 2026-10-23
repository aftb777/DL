# Build an autoencoder that compresses and
# reconstructs digit images, show original vs reconstructed images and report
# reconstruction 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1. Load MNIST Dataset
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

# 2. Normalize and Flatten Images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# 3. Build Encoder
input_img = Input(shape=(784,))

encoded = Dense(128, activation="relu")(input_img)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)

# 4. Build Decoder
decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(decoded)
decoded = Dense(784, activation="sigmoid")(decoded)

# 5. Create Autoencoder Model
autoencoder = Model(input_img, decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()

# 6. Train Autoencoder
autoencoder.fit(
    X_train,
    X_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1
)

# 7. Reconstruct Images
reconstructed = autoencoder.predict(X_test[:10], verbose=0)

# 8. Calculate Reconstruction Error
mse = np.mean((X_test[:10] - reconstructed) ** 2)

print("Reconstruction MSE:", mse)

# 9. Display Original and Reconstructed Images
plt.figure(figsize=(12, 4))

for i in range(10):

    # Original Image
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

    # Reconstructed Image
    plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

plt.suptitle("Original Images (Top) vs Reconstructed Images (Bottom)")
plt.show()