# Question:
# Build an embedding-based recommendation model using user-product
# interactions from the Retailrocket E-commerce Dataset.
# Evaluate the model using Precision@K and Recall@K.

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from sklearn.model_selection import train_test_split

# 1. Load Dataset
df = pd.read_csv("events.csv")

# 2. Select User-Product Interactions
df = df[df["event"].isin(["view", "addtocart", "transaction"])]
df = df.dropna(subset=["visitorid", "itemid"])

# Remove duplicate interactions
df = df.drop_duplicates(subset=["visitorid", "itemid"])

# 3. Convert User and Item IDs into Integer IDs
df["user_id"] = df["visitorid"].astype("category").cat.codes
df["item_id"] = df["itemid"].astype("category").cat.codes

num_users = df["user_id"].nunique()
num_items = df["item_id"].nunique()

print("Total Users:", num_users)
print("Total Products:", num_items)

# 4. Split Data
train_df, test_df = train_test_split(
    df[["user_id", "item_id"]],
    test_size=0.2,
    random_state=42
)

# 5. Create Negative Samples
# Observed interactions = positive examples
rng = np.random.default_rng(42)

train_users = train_df["user_id"].to_numpy()
train_items = train_df["item_id"].to_numpy()

negative_users = rng.integers(0, num_users, len(train_df))
negative_items = rng.integers(0, num_items, len(train_df))

X_users = np.concatenate([train_users, negative_users])
X_items = np.concatenate([train_items, negative_items])

y = np.concatenate([
    np.ones(len(train_df)),
    np.zeros(len(train_df))
])

# 6. Build Embedding Model
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

user_embedding = Embedding(num_users, 32)(user_input)
item_embedding = Embedding(num_items, 32)(item_input)

user_vector = Flatten()(user_embedding)
item_vector = Flatten()(item_embedding)

combined = Concatenate()([user_vector, item_vector])

hidden = Dense(64, activation="relu")(combined)
hidden = Dense(32, activation="relu")(hidden)

output = Dense(1, activation="sigmoid")(hidden)

model = Model(
    inputs=[user_input, item_input],
    outputs=output
)

# 7. Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 8. Train Model
model.fit(
    [X_users, X_items],
    y,
    epochs=5,
    batch_size=1024,
    validation_split=0.1
)

# 9. Recommend Top-K Products for a User
def recommend_products(user_id, k=5):

    # Products already seen by this user in training data
    seen_items = set(
        train_df.loc[
            train_df["user_id"] == user_id, "item_id"
        ]
    )

    candidates = [
        item for item in range(num_items)
        if item not in seen_items
    ]

    if not candidates:
        return []

    users = np.full(len(candidates), user_id)

    scores = model.predict(
        [users, np.array(candidates)],
        batch_size=4096,
        verbose=0
    ).flatten()

    top_indices = np.argsort(scores)[::-1][:k]

    return [
        (candidates[i], round(float(scores[i]), 4))
        for i in top_indices
    ]

# 10. Evaluate Precision@K and Recall@K
def evaluate_recommendations(k=5, max_users=100):

    precisions = []
    recalls = []

    # Test interactions are used as held-out relevant items.
    test_users = test_df["user_id"].unique()[:max_users]

    for user in test_users:

        actual_items = set(
            test_df.loc[
                test_df["user_id"] == user, "item_id"
            ]
        )

        recommendations = recommend_products(user, k)
        recommended_items = {
            item for item, score in recommendations
        }

        hits = len(recommended_items & actual_items)

        precisions.append(hits / k)
        recalls.append(
            hits / len(actual_items) if actual_items else 0
        )

    print("Precision@K:", np.mean(precisions))
    print("Recall@K:", np.mean(recalls))

evaluate_recommendations(k=5)

# 11. Display Example Recommendations
example_user = int(test_df["user_id"].iloc[0])

print("\nExample User:", example_user)
print("Recommended Products:", recommend_products(example_user, k=5))