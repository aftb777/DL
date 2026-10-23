# Question:
# Build a Transformer-based text classification model using self-attention
# to classify AG News articles into four categories.
# Compare its accuracy and training time with an LSTM model.

import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Sequential

# 1. Load Dataset
# CSV must contain columns: text and label (labels 0, 1, 2, 3).
df = pd.read_csv("ag_news.csv")

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"].astype(int),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# 2. Text Preprocessing
max_tokens = 10000
sequence_length = 100

vectorizer = layers.TextVectorization(
    max_tokens=max_tokens,
    output_sequence_length=sequence_length
)

vectorizer.adapt(X_train)

vocab_size = len(vectorizer.get_vocabulary())

# 3. Build Transformer Block
def transformer_block(x):
    attention = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=32
    )(x, x)

    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    feed_forward = layers.Dense(64, activation="relu")(x)
    feed_forward = layers.Dense(64)(feed_forward)

    x = layers.Add()([x, feed_forward])
    x = layers.LayerNormalization()(x)

    return x

# 4. Create Transformer Model
def create_transformer():

    inputs = layers.Input(shape=(), dtype=tf.string)

    x = vectorizer(inputs)
    x = layers.Embedding(vocab_size, 64)(x)

    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(4, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# 5. Create LSTM Model
def create_lstm():

    model = Sequential([
        vectorizer,
        layers.Embedding(vocab_size, 64, mask_zero=True),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# 6. Train and Evaluate Both Models
results = {}

for name, model in [
    ("LSTM", create_lstm()),
    ("Transformer", create_transformer())
]:

    print("\nTraining", name, "Model")

    start = time.time()

    history = model.fit(
        X_train,
        y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    training_time = time.time() - start

    loss, accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    results[name] = {
        "accuracy": accuracy,
        "time": training_time,
        "history": history
    }

    print("Test Accuracy:", round(accuracy * 100, 2), "%")
    print("Training Time:", round(training_time, 2), "seconds")

# 7. Compare Results
print("\nFinal Comparison")

for name, result in results.items():
    print(
        name,
        "| Accuracy:", round(result["accuracy"] * 100, 2), "%",
        "| Time:", round(result["time"], 2), "seconds"
    )

# 8. Plot Test Accuracy
plt.bar(
    results.keys(),
    [r["accuracy"] * 100 for r in results.values()]
)

plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.title("Transformer vs LSTM")
plt.show()