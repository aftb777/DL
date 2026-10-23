# CNN-Based Image Classification: Build a CNN to classify images into
# 10 classes: preprocess data, set training parameters, add dropout, save and
# reload the model, and evaluate it.


import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)

# 1. Load CIFAR-10 Dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# 2. Normalize Images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# 3. Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu",
           padding="same", input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu", padding="same"),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

# 4. Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# 6. Evaluate Model
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

# 7. Save Model
model.save("cifar10_cnn.keras")
print("Model Saved Successfully!")

# 8. Reload Saved Model
loaded_model = load_model("cifar10_cnn.keras")

# 9. Evaluate Reloaded Model
loss, accuracy = loaded_model.evaluate(
    X_test, y_test, verbose=0
)

print("Reloaded Model Accuracy:", accuracy * 100, "%")

# 10. Plot Accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()