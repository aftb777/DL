# Question:
# Build a feedforward neural network to classify breast tumours
# as malignant or benign. Compare activation functions and optimizers
# using binary cross-entropy loss.
# Dataset: Breast Cancer Wisconsin.

import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Dataset
data = load_breast_cancer()

X = data.data
y = data.target

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Standardize Features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results = {}

# 4. Compare Activation Functions and Optimizers
for activation in ["relu", "tanh"]:
    for optimizer in ["adam", "sgd"]:

        print("\nActivation:", activation, "| Optimizer:", optimizer)

        # 5. Build DNN Model
        model = Sequential([
            Dense(32, activation=activation,
                  input_shape=(X_train.shape[1],)),
            Dense(16, activation=activation),
            Dense(1, activation="sigmoid")
        ])

        # 6. Compile Model
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # 7. Train Model
        history = model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )

        # 8. Evaluate Model
        loss, accuracy = model.evaluate(
            X_test, y_test, verbose=0
        )

        results[activation + "_" + optimizer] = accuracy

        print("Test Loss:", round(loss, 4))
        print("Test Accuracy:", round(accuracy * 100, 2), "%")

        # 9. Classification Report for Each Model
        predictions = (model.predict(X_test, verbose=0) >= 0.5).astype(int)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        print(classification_report(
            y_test,
            predictions,
            target_names=data.target_names,
            zero_division=0
        ))

# 10. Compare Results
print("\nFinal Accuracy Comparison")

for name, accuracy in results.items():
    print(name, ":", round(accuracy * 100, 2), "%")

# 11. Plot Results
plt.figure(figsize=(8, 5))

plt.bar(
    results.keys(),
    [acc * 100 for acc in results.values()]
)

plt.xlabel("Activation and Optimizer")
plt.ylabel("Test Accuracy (%)")
plt.title("DNN Binary Classification Comparison")
plt.show()