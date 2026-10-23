# Question:
# Train a CNN using Xavier and He weight initialization.
# Compare SGD, Adam and RMSprop optimizers and a learning-rate schedule.
# Evaluate convergence and accuracy using CIFAR-10.

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.initializers import GlorotUniform, HeNormal

# 1. Load CIFAR-10 Dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# 2. Normalize Images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

results = {}

# 3. Compare Initializers and Optimizers
for init_name in ["Xavier", "He"]:
    for opt_name in ["SGD", "Adam", "RMSprop"]:

        print("\nTraining:", init_name, opt_name)

        # Select initializer
        if init_name == "Xavier":
            initializer = GlorotUniform()
        else:
            initializer = HeNormal()

        # Build CNN
        model = Sequential([
            Conv2D(
                32, (3, 3), activation="relu",
                kernel_initializer=initializer,
                input_shape=(32, 32, 3)
            ),
            MaxPooling2D((2, 2)),

            Conv2D(
                64, (3, 3), activation="relu",
                kernel_initializer=initializer
            ),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(
                64, activation="relu",
                kernel_initializer=initializer
            ),
            Dense(10, activation="softmax")
        ])

        # Select optimizer
        if opt_name == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        elif opt_name == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        else:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

        # Learning-rate schedule
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: 0.001 if epoch < 2 else 0.0003
        )

        # Compile Model
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Train Model
        history = model.fit(
            X_train, y_train,
            epochs=3,
            batch_size=128,
            validation_split=0.1,
            callbacks=[lr_schedule],
            verbose=0
        )

        # Evaluate Model
        loss, accuracy = model.evaluate(
            X_test, y_test, verbose=0
        )

        results[init_name + "-" + opt_name] = accuracy

        print("Test Accuracy:", round(accuracy * 100, 2), "%")

# 4. Display Comparison
print("\nFinal Accuracy Comparison")

for name, accuracy in results.items():
    print(name, ":", round(accuracy * 100, 2), "%")

# 5. Plot Results
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), [x * 100 for x in results.values()])
plt.xlabel("Initialization and Optimizer")
plt.ylabel("Test Accuracy (%)")
plt.title("CNN Optimization Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()