# Question:
# Build a neural network with a softmax output layer to classify
# wines into 3 classes using the Wine dataset.
# Report accuracy, precision, recall and confusion matrix.

import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Dataset
data = load_wine()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Number of Classes:", len(set(y)))

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

# 4. Build Neural Network
model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(3, activation="softmax")
])

# 5. Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# 7. Predict Classes
predictions = model.predict(X_test, verbose=0)
y_pred = predictions.argmax(axis=1)

# 8. Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=data.target_names,
    zero_division=0
))

# 10. Plot Confusion Matrix
plt.imshow(cm, cmap="Blues")
plt.title("Wine Classification Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(range(3), data.target_names, rotation=30)
plt.yticks(range(3), data.target_names)

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()

# 11. Plot Training Accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()