# Question:
# Classify social media post sentiment using an LSTM.
# Build a user interaction graph using @mentions and visualize
# sentiment across the network.
# Dataset: Sentiment140 / Social-Network Dataset.

import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load Sentiment Dataset
df = pd.read_csv("sentiment.csv")

# Required columns: text, label, user
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)

# Convert sentiment labels to 0 and 1 if necessary.
# Sentiment140: 0 = negative, 4 = positive.
df["label"] = df["label"].replace({4: 1})

# 2. Preprocess Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"].astype(int),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].astype(int)
)

# 4. Convert Text into Sequences
vectorizer = TextVectorization(
    max_tokens=10000,
    output_sequence_length=100
)

vectorizer.adapt(X_train)

# 5. Build LSTM Model
model = Sequential([
    vectorizer,
    Embedding(10000, 64, mask_zero=True),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 6. Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1
)

# 7. Evaluate Sentiment Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Accuracy:", round(accuracy * 100, 2), "%")

predictions = model.predict(X_test, verbose=0).flatten()
predicted_labels = (predictions >= 0.5).astype(int)

print(classification_report(
    y_test,
    predicted_labels,
    zero_division=0
))

# 8. Predict Sentiment of a New Post
sample_post = ["I really enjoyed this movie"]

score = model.predict(
    np.array(sample_post), verbose=0
)[0][0]

if score >= 0.5:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")

# 9. Load User Interaction Edges
edges = pd.read_csv("edges.csv")

# Required columns: source, target
G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target",
    create_using=nx.DiGraph()
)

# 10. Calculate Average Sentiment per User
df["predicted_sentiment"] = (
    model.predict(df["text"], batch_size=256, verbose=0).flatten()
)

user_sentiment = df.groupby("user")[
    "predicted_sentiment"
].mean().to_dict()

# 11. Visualize Network
node_colors = [
    user_sentiment.get(str(node), 0.5)
    for node in G.nodes()
]

plt.figure(figsize=(10, 8))

pos = nx.spring_layout(G, seed=42)

nx.draw_networkx(
    G,
    pos,
    node_color=node_colors,
    cmap=plt.cm.coolwarm,
    node_size=300,
    font_size=7,
    with_labels=True,
    arrows=False
)

plt.title("User Interaction Network: Sentiment Visualization")
plt.colorbar(
    plt.cm.ScalarMappable(
        cmap=plt.cm.coolwarm,
        norm=plt.Normalize(0, 1)
    ),
    label="Predicted Sentiment (0 = Negative, 1 = Positive)"
)

plt.axis("off")
plt.show()