# Sentiment Analysis Using RNN, LSTM and GRU: Classify movie
# reviews as positive or negative with RNN, LSTM and GRU models, and
# compare accuracy, training time and handling of long reviews.


import tensorflow as tf
import time
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense

# 1. Load IMDB Dataset
vocab_size = 10000
max_length = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 2. Pad Sequences
X_train = pad_sequences(
    X_train, maxlen=max_length, padding="post", truncating="post"
)

X_test = pad_sequences(
    X_test, maxlen=max_length, padding="post", truncating="post"
)

# 3. Create Model Function
def create_model(model_type):

    model = Sequential()
    model.add(Embedding(vocab_size, 64, mask_zero=True))

    if model_type == "RNN":
        model.add(SimpleRNN(64))

    elif model_type == "LSTM":
        model.add(LSTM(64))

    elif model_type == "GRU":
        model.add(GRU(64))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# 4. Train and Evaluate All Models
results = {}

for model_type in ["RNN", "LSTM", "GRU"]:

    print("\nTraining", model_type, "Model")

    model = create_model(model_type)

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
        epochs=2,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    training_time = time.time() - start_time

    loss, accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    results[model_type] = {
        "accuracy": accuracy,
        "time": training_time
    }

    print("Test Loss:", round(loss, 4))
    print("Test Accuracy:", round(accuracy * 100, 2), "%")
    print("Training Time:", round(training_time, 2), "seconds")

# 5. Compare Results
print("\nFinal Comparison")

for name, result in results.items():
    print(
        name,
        "| Accuracy:", round(result["accuracy"] * 100, 2), "%",
        "| Time:", round(result["time"], 2), "seconds"
    )

# 6. Plot Accuracy Comparison
names = list(results.keys())
accuracies = [
    results[name]["accuracy"] * 100 for name in names
]

plt.bar(names, accuracies)
plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.title("RNN vs LSTM vs GRU")
plt.show()