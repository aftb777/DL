# Regression Using Keras: Build a house price prediction model trained
# with gradient descent, and compare versions with L1 regularization, L2
# regularization and dropout to

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2

# 1. Load Dataset
data = fetch_california_housing()

X = data.data
y = data.target

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Normalize Features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create Model
def create_model(reg_type="none"):

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))

    if reg_type == "l1":
        model.layers[0].kernel_regularizer = l1(0.01)

    elif reg_type == "l2":
        model.layers[0].kernel_regularizer = l2(0.01)

    if reg_type == "dropout":
        model.add(Dropout(0.3))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer="sgd",
        loss="mse",
        metrics=["mae"]
    )

    return model

# 5. Train and Evaluate Different Models
results = {}

for reg in ["none", "l1", "l2", "dropout"]:

    print("\nTraining Model:", reg)

    model = create_model(reg)

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    results[reg] = mae

    print("Test MSE:", round(loss, 4))
    print("Test MAE:", round(mae, 4))

    # Plot training and validation loss
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Comparison: " + reg)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

# 6. Compare Models
print("\nFinal MAE Comparison:")

for name, mae in results.items():
    print(name, ":", round(mae, 4))