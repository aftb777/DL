# Question:
# Extract image features using pretrained ResNet50 and build a
# CNN + RNN (LSTM) model to generate image captions.
# Dataset: Flickr8k.

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Add
from tensorflow.keras.models import Model

# 1. Load Captions
df = pd.read_csv("flickr8k/captions.txt")

df["caption"] = df["caption"].astype(str).str.lower()
df["caption"] = df["caption"].apply(
    lambda x: "startseq " + x + " endseq"
)

# 2. Tokenize Captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["caption"])

vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(df["caption"])
max_length = max(len(seq) for seq in sequences)

print("Vocabulary Size:", vocab_size)
print("Maximum Caption Length:", max_length)

# 3. Load Pretrained ResNet50
cnn = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# 4. Extract Image Features
features = {}

for image_name in df["image"].unique():
    image_path = os.path.join("flickr8k/Images", image_name)

    if not os.path.exists(image_path):
        continue

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = cnn.predict(image, verbose=0)
    features[image_name] = feature[0]

print("Extracted Features:", len(features))

# 5. Prepare Training Sequences
X_image = []
X_text = []
y_word = []

for _, row in df.iterrows():
    image_name = row["image"]

    if image_name not in features:
        continue

    seq = tokenizer.texts_to_sequences([row["caption"]])[0]

    for i in range(1, len(seq)):
        in_seq = seq[:i]
        out_word = seq[i]

        in_seq = pad_sequences(
            [in_seq], maxlen=max_length - 1, padding="pre"
        )[0]

        X_image.append(features[image_name])
        X_text.append(in_seq)
        y_word.append(out_word)

X_image = np.array(X_image, dtype="float32")
X_text = np.array(X_text, dtype="int32")
y_word = np.array(y_word, dtype="int32")

print("Training Samples:", len(y_word))

# 6. Build Image Feature Branch
image_input = Input(shape=(2048,))
image_features = Dense(256, activation="relu")(image_input)

# 7. Build Caption Sequence Branch
text_input = Input(shape=(max_length - 1,))
text_features = Embedding(
    vocab_size, 256, mask_zero=True
)(text_input)

text_features = LSTM(256)(text_features)

# 8. Combine Image and Text Features
combined = Add()([image_features, text_features])
output = Dense(vocab_size, activation="softmax")(combined)

model = Model(
    inputs=[image_input, text_input],
    outputs=output
)

# 9. Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 10. Train Model
model.fit(
    [X_image, X_text],
    y_word,
    epochs=5,
    batch_size=256,
    validation_split=0.1
)

# 11. Generate Caption for an Image
def generate_caption(image_name):
    if image_name not in features:
        print("Image feature not found!")
        return

    caption = "startseq"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]

        seq = pad_sequences(
            [seq], maxlen=max_length - 1, padding="pre"
        )

        prediction = model.predict(
            [features[image_name][None, :], seq],
            verbose=0
        )

        predicted_id = int(np.argmax(prediction[0]))
        predicted_word = tokenizer.index_word.get(predicted_id, "")

        if not predicted_word or predicted_word == "endseq":
            break

        caption += " " + predicted_word

    return caption.replace("startseq", "").strip()

# 12. Display Generated Caption
sample_image = next(iter(features))
caption = generate_caption(sample_image)

image = load_img(
    os.path.join("flickr8k/Images", sample_image),
    target_size=(224, 224)
)

plt.imshow(image)
plt.title(caption)
plt.axis("off")
plt.show()

print("Generated Caption:", caption)