# Question:
# Build MLP models with different hidden layers, neuron counts,
# and activation functions (sigmoid, tanh, ReLU).
# Compare classification accuracy using the Fashion-MNIST dataset.

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# 1. Load Fashion-MNIST Dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# 2. Normalize Pixel Values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

results = {}

# 3. Compare Activation Functions and Neurons
for activation in ["sigmoid", "tanh", "relu"]:

    for neurons in [64, 128]:

        print("\nActivation:", activation, "| Neurons:", neurons)

        # 4. Build MLP Model
        model = Sequential([
            Flatten(input_shape=(28, 28)),

            Dense(neurons, activation=activation),

            Dense(neurons // 2, activation=activation),

            Dense(10, activation="softmax")
        ])

        # 5. Compile Model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # 6. Train Model
        history = model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=128,
            validation_split=0.1,
            verbose=0
        )

        # 7. Evaluate Model
        loss, accuracy = model.evaluate(
            X_test, y_test, verbose=0
        )

        name = activation + "_" + str(neurons)
        results[name] = accuracy

        print("Test Accuracy:", round(accuracy * 100, 2), "%")

# 8. Display Results
print("\nFinal Accuracy Comparison")

for name, accuracy in results.items():
    print(name, ":", round(accuracy * 100, 2), "%")

# 9. Plot Comparison
plt.figure(figsize=(10, 5))

plt.bar(
    results.keys(),
    [accuracy * 100 for accuracy in results.values()]
)

plt.xlabel("Activation and Neuron Configuration")
plt.ylabel("Test Accuracy (%)")
plt.title("MLP Classification Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()