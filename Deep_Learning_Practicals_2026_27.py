"""
DEEP LEARNING PRACTICAL EXAM — PRACTICE PROBLEM LIST (2026–27)
Exam revision templates. Each section starts with the practical question.
Install common packages:
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
Some practicals require extra datasets/packages; see comments in each section.
"""

# ============================================================
# PRACTICAL 1
# Question: Perform basic tensor operations in TensorFlow, then build a
# single-layer Perceptron and an MLP for handwritten digit recognition
# and compare their accuracy. Dataset: MNIST
# ============================================================
def practical_01():
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tensorflow.keras import layers, models

    # Basic tensor operations
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    print("Addition:\n", tf.add(a, b))
    print("Matrix multiplication:\n", tf.matmul(a, b))
    print("Reshape:\n", tf.reshape(a, (4, 1)))

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Single-layer perceptron (one Dense output layer)
    perceptron = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(10, activation="softmax")
    ])
    perceptron.compile(optimizer="adam",
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    perceptron.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=1)
    p_loss, p_acc = perceptron.evaluate(x_test, y_test, verbose=0)

    # Multilayer perceptron
    mlp = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    mlp.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    mlp.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=1)
    m_loss, m_acc = mlp.evaluate(x_test, y_test, verbose=0)
    print(f"Perceptron accuracy: {p_acc:.4f}; MLP accuracy: {m_acc:.4f}")


# ============================================================
# PRACTICAL 2
# Question: Build a house-price prediction model trained with gradient
# descent, and compare L1, L2 regularization and dropout to study
# overfitting. Dataset: California Housing
# ============================================================
def practical_02():
    import tensorflow as tf
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras import layers, regularizers, models

    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    def build_model(kind="none"):
        reg = regularizers.l1(0.001) if kind == "l1" else (
            regularizers.l2(0.001) if kind == "l2" else None)
        model = models.Sequential([layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(64, activation="relu", kernel_regularizer=reg)])
        if kind == "dropout":
            model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation="relu"))
        if kind == "dropout":
            model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1))
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                      loss="mse", metrics=["mae"])
        return model

    for kind in ["none", "l1", "l2", "dropout"]:
        model = build_model(kind)
        model.fit(X_train, y_train, epochs=20, batch_size=32,
                  validation_split=0.2, verbose=0)
        mse, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"{kind}: Test MSE={mse:.4f}, MAE={mae:.4f}")


# ============================================================
# PRACTICAL 3
# Question: Build a CNN to classify CIFAR-10 images into 10 classes;
# preprocess, set training parameters, add dropout, save/reload model,
# and evaluate it. Dataset: CIFAR-10
# ============================================================
def practical_03():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64,
              validation_split=0.1)
    print("Test:", model.evaluate(x_test, y_test, verbose=0))
    model.save("cifar10_cnn.keras")
    loaded = tf.keras.models.load_model("cifar10_cnn.keras")
    print("Reloaded model test:", loaded.evaluate(x_test, y_test, verbose=0))


# ============================================================
# PRACTICAL 4
# Question: Use a pretrained CNN (VGG16 or MobileNetV2) to classify
# cats and dogs; compare feature extraction with fine-tuning.
# Dataset: Cats vs. Dogs (folder structure: data/train/cats, data/train/dogs,
# data/validation/cats, data/validation/dogs)
# ============================================================
def practical_04():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    train_dir, val_dir = "data/train", "data/validation"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(160, 160), batch_size=32)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(160, 160), batch_size=32)
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    base = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3), include_top=False, weights="imagenet")
    base.trainable = False  # Feature extraction
    inputs = layers.Input((160, 160, 3))
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=3)
    print("Feature extraction:", model.evaluate(val_ds, verbose=0))

    # Fine-tuning: unfreeze last layers and train with a low learning rate.
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=2)
    print("Fine-tuned:", model.evaluate(val_ds, verbose=0))


# ============================================================
# PRACTICAL 5
# Question: Classify IMDB movie reviews using RNN, LSTM and GRU;
# compare accuracy, training time and handling of long reviews.
# Dataset: IMDB Movie Reviews
# ============================================================
def practical_05():
    import time
    import tensorflow as tf
    from tensorflow.keras import layers, models

    vocab, maxlen = 10000, 200
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    for kind in ["SimpleRNN", "LSTM", "GRU"]:
        recurrent = {"SimpleRNN": layers.SimpleRNN,
                     "LSTM": layers.LSTM, "GRU": layers.GRU}[kind]
        model = models.Sequential([
            layers.Embedding(vocab, 64, input_length=maxlen),
            recurrent(64),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        start = time.time()
        model.fit(x_train, y_train, epochs=2, batch_size=128,
                  validation_split=0.1, verbose=1)
        elapsed = time.time() - start
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"{kind}: accuracy={acc:.4f}, training_seconds={elapsed:.1f}")


# ============================================================
# PRACTICAL 6
# Question: Build an autoencoder that compresses and reconstructs digit
# images, show original vs reconstructed images and report reconstruction
# error. Dataset: MNIST
# ============================================================
def practical_06():
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras import layers, models

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
    x_test = x_test.astype("float32").reshape(-1, 784) / 255.0
    inp = layers.Input(shape=(784,))
    encoded = layers.Dense(64, activation="relu")(inp)
    encoded = layers.Dense(32, activation="relu")(encoded)
    decoded = layers.Dense(64, activation="relu")(encoded)
    out = layers.Dense(784, activation="sigmoid")(decoded)
    autoencoder = models.Model(inp, out)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(x_train, x_train, epochs=5, batch_size=256,
                    validation_split=0.1)
    reconstructed = autoencoder.predict(x_test[:10], verbose=0)
    print("Reconstruction MSE:", np.mean((x_test[:10] - reconstructed) ** 2))
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for i in range(10):
        axes[0, i].imshow(x_test[i].reshape(28, 28), cmap="gray")
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original"); axes[1, 0].set_ylabel("Rebuilt")
    plt.tight_layout(); plt.show()


# ============================================================
# PRACTICAL 7
# Question: Train CNNs with Xavier vs He initialization, different
# learning-rate schedules and SGD, Adam, RMSprop; compare convergence
# and accuracy. Dataset: CIFAR-10
# ============================================================
def practical_07():
    import tensorflow as tf
    from tensorflow.keras import layers, models, initializers

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    for init_name, init in [("Xavier", initializers.GlorotUniform()),
                            ("He", initializers.HeNormal())]:
        for opt_name in ["SGD", "Adam", "RMSprop"]:
            model = models.Sequential([
                layers.Input((32, 32, 3)),
                layers.Conv2D(32, 3, activation="relu", kernel_initializer=init),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(64, activation="relu", kernel_initializer=init),
                layers.Dense(10, activation="softmax")
            ])
            optimizer = (tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
                         if opt_name == "SGD" else
                         tf.keras.optimizers.Adam(learning_rate=0.001)
                         if opt_name == "Adam" else
                         tf.keras.optimizers.RMSprop(learning_rate=0.001))
            model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            # Example learning-rate schedule
            schedule = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: 0.001 if epoch < 2 else 0.0003)
            model.fit(x_train, y_train, epochs=3, batch_size=128,
                      validation_split=0.1, callbacks=[schedule], verbose=0)
            _, acc = model.evaluate(x_test, y_test, verbose=0)
            print(init_name, opt_name, "accuracy:", round(acc, 4))


# ============================================================
# PRACTICAL 8
# Question: Extract features with pretrained ResNet50 and build a
# CNN + RNN (LSTM) model to generate image captions.
# Dataset: Flickr8k (requires image files and captions text file).
# ============================================================
def practical_08():
    # This is a compact starter template. Arrange Flickr8k images in
    # flickr8k/Images/ and captions in flickr8k/captions.txt with columns image,caption.
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras import layers, models

    captions_df = pd.read_csv("flickr8k/captions.txt")
    captions_df.columns = [c.strip().lower() for c in captions_df.columns]
    image_col = "image" if "image" in captions_df.columns else captions_df.columns[0]
    caption_col = "caption" if "caption" in captions_df.columns else captions_df.columns[1]
    captions = ["startseq " + str(c).lower() + " endseq" for c in captions_df[caption_col]]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max(len(c.split()) for c in captions)
    sequences = tokenizer.texts_to_sequences(captions)
    sequences = pad_sequences(sequences, maxlen=max_len, padding="post")
    # ResNet50 feature extraction shown for one image; repeat/cache for dataset.
    encoder = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    image_name = captions_df[image_col].iloc[0]
    img = load_img("flickr8k/Images/" + image_name, target_size=(224, 224))
    arr = preprocess_input(np.expand_dims(img_to_array(img), axis=0))
    feature = encoder.predict(arr, verbose=0)
    print("Image feature shape:", feature.shape, "Vocabulary:", vocab_size)
    # Caption decoder (requires matching cached image features and padded sequences).
    image_input = layers.Input(shape=(2048,))
    image_dense = layers.Dense(256, activation="relu")(image_input)
    text_input = layers.Input(shape=(max_len - 1,))
    emb = layers.Embedding(vocab_size, 256, mask_zero=True)(text_input)
    decoder = layers.LSTM(256)(emb, initial_state=None)
    merged = layers.Add()([image_dense, decoder])
    output = layers.Dense(vocab_size, activation="softmax")(merged)
    model = models.Model([image_input, text_input], output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    print("Decoder built. For training, prepare image-feature / partial-caption / next-word triples.")


# ============================================================
# PRACTICAL 9
# Question: Build an embedding-based product recommendation model from
# user–product interactions and evaluate Precision@K or Recall@K.
# Dataset: Retailrocket E-commerce Dataset (events.csv).
# ============================================================
def practical_09():
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("events.csv")  # Retailrocket events.csv
    df = df[df["event"].isin(["view", "addtocart", "transaction"])]
    df = df.dropna(subset=["visitorid", "itemid"])
    df["user"] = df["visitorid"].astype("category").cat.codes
    df["item"] = df["itemid"].astype("category").cat.codes
    n_users, n_items = df.user.nunique(), df.item.nunique()
    interactions = df[["user", "item"]].drop_duplicates()
    train, test = train_test_split(interactions, test_size=0.2, random_state=42)
    user_in = layers.Input(shape=(1,))
    item_in = layers.Input(shape=(1,))
    u = layers.Flatten()(layers.Embedding(n_users, 32)(user_in))
    i = layers.Flatten()(layers.Embedding(n_items, 32)(item_in))
    x = layers.Concatenate()([u, i])
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model([user_in, item_in], out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Positive-only interactions need sampled negative examples for meaningful training.
    rng = np.random.default_rng(42)
    neg_users = rng.integers(0, n_users, len(train))
    neg_items = rng.integers(0, n_items, len(train))
    users = np.concatenate([train.user.values, neg_users])
    items = np.concatenate([train.item.values, neg_items])
    labels = np.concatenate([np.ones(len(train)), np.zeros(len(train))])
    model.fit([users, items], labels, epochs=3, batch_size=1024, validation_split=0.1)
    print("Model trained. For Precision@K/Recall@K, rank candidate items for each held-out user.")
    print("Test interactions:", len(test))


# ============================================================
# PRACTICAL 10
# Question: Classify post sentiment with RNN/LSTM, build a user interaction
# graph (e.g. @mentions), and visualize sentiment across the network.
# Dataset: Sentiment140 / social-network data. CSV should have text and label;
# mention edges should be available in a separate edges.csv with source,target.
# ============================================================
def practical_10():
    import re
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("sentiment.csv")  # columns: text,label (0=negative, 1=positive)
    df["text"] = df["text"].astype(str).str.lower().str.replace(r"http\\S+|@\\w+", " ", regex=True)
    x_train, x_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=0.2, random_state=42, stratify=df.label)
    vectorizer = layers.TextVectorization(max_tokens=10000, output_sequence_length=100)
    vectorizer.adapt(x_train)
    model = models.Sequential([layers.Input(shape=(), dtype=tf.string), vectorizer,
        layers.Embedding(10000, 64, mask_zero=True), layers.LSTM(64),
        layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3, validation_split=0.1)
    print("Test evaluation:", model.evaluate(x_test, y_test, verbose=0))

    edges = pd.read_csv("edges.csv")  # columns: source,target
    graph = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph())
    sentiment_by_user = df.groupby("user")["label"].mean().to_dict() if "user" in df else {}
    node_colors = [sentiment_by_user.get(n, 0.5) for n in graph.nodes()]
    nx.draw(graph, node_color=node_colors, cmap=plt.cm.coolwarm, node_size=40, arrows=False)
    plt.title("User interaction graph (colour = average sentiment where available)")
    plt.show()


# ============================================================
# PRACTICAL 11
# Question: Build a GAN that generates synthetic images, plot generator
# and discriminator losses, and show generated samples during training.
# Dataset: MNIST / Fashion-MNIST
# ============================================================
def practical_11():
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tensorflow.keras import layers, models

    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") - 127.5) / 127.5
    x_train = x_train.reshape(-1, 784)
    generator = models.Sequential([layers.Input((100,)), layers.Dense(256),
        layers.LeakyReLU(negative_slope=0.2), layers.Dense(512),
        layers.LeakyReLU(negative_slope=0.2), layers.Dense(784, activation="tanh")])
    discriminator = models.Sequential([layers.Input((784,)), layers.Dense(256),
        layers.LeakyReLU(negative_slope=0.2), layers.Dense(1, activation="sigmoid")])
    discriminator.compile(optimizer="adam", loss="binary_crossentropy")
    discriminator.trainable = False
    gan = models.Sequential([generator, discriminator])
    gan.compile(optimizer="adam", loss="binary_crossentropy")
    d_losses, g_losses = [], []
    batch_size = 128
    for epoch in range(5):
        idx = tf.random.uniform((batch_size,), maxval=len(x_train), dtype=tf.int32)
        real = tf.gather(x_train, idx)
        noise = tf.random.normal((batch_size, 100))
        fake = generator.predict(noise, verbose=0)
        discriminator.trainable = True
        d1 = discriminator.train_on_batch(real, tf.ones((batch_size, 1)))
        d2 = discriminator.train_on_batch(fake, tf.zeros((batch_size, 1)))
        discriminator.trainable = False
        g = gan.train_on_batch(tf.random.normal((batch_size, 100)), tf.ones((batch_size, 1)))
        d_losses.append((float(d1) + float(d2)) / 2); g_losses.append(float(g))
        print("Epoch", epoch + 1, "D loss", d_losses[-1], "G loss", g_losses[-1])
    plt.plot(d_losses, label="Discriminator"); plt.plot(g_losses, label="Generator")
    plt.legend(); plt.title("GAN losses"); plt.show()
    samples = generator.predict(tf.random.normal((10, 100)), verbose=0)
    fig, axes = plt.subplots(1, 10, figsize=(12, 2))
    for ax, sample in zip(axes, samples):
        ax.imshow(sample.reshape(28, 28), cmap="gray"); ax.axis("off")
    plt.show()


# ============================================================
# PRACTICAL 12
# Question: Build a Transformer self-attention model to classify AG News
# into four categories and compare accuracy/training time with an LSTM.
# Dataset: AG News CSV files with columns label,title,description.
# ============================================================
def practical_12():
    import time
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("ag_news.csv")  # columns: text,label; labels must be 0..3
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"].astype(str), df["label"].astype(int), test_size=0.2,
        random_state=42, stratify=df["label"])
    max_tokens, seq_len = 20000, 100
    vectorizer = layers.TextVectorization(max_tokens=max_tokens,
                                           output_sequence_length=seq_len)
    vectorizer.adapt(x_train)
    vocab_size = len(vectorizer.get_vocabulary())

    def transformer_block(x):
        attn = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
        x = layers.LayerNormalization()(x + attn)
        ff = layers.Dense(64, activation="relu")(x)
        ff = layers.Dense(64)(ff)
        return layers.LayerNormalization()(x + ff)

    def build(kind):
        inp = layers.Input(shape=(), dtype=tf.string)
        x = vectorizer(inp)
        x = layers.Embedding(vocab_size, 64)(x)
        if kind == "Transformer":
            x = transformer_block(x)
            x = layers.GlobalAveragePooling1D()(x)
        else:
            x = layers.LSTM(64)(x)
        out = layers.Dense(4, activation="softmax")(x)
        m = models.Model(inp, out)
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        return m

    for kind in ["LSTM", "Transformer"]:
        model = build(kind)
        start = time.time()
        model.fit(x_train, y_train, epochs=3, batch_size=64,
                  validation_split=0.1, verbose=1)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(kind, "accuracy:", round(acc, 4), "seconds:", round(time.time()-start, 1))


# ============================================================
# PRACTICAL 13
# Question: Build MLPs with different hidden layers, neurons and sigmoid,
# tanh and ReLU activations; compare classification accuracy.
# Dataset: Fashion-MNIST
# ============================================================
def practical_13():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    for activation in ["sigmoid", "tanh", "relu"]:
        for units in [64, 128]:
            model = models.Sequential([layers.Input((28, 28)), layers.Flatten(),
                layers.Dense(units, activation=activation),
                layers.Dense(units // 2, activation=activation),
                layers.Dense(10, activation="softmax")])
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
            _, acc = model.evaluate(x_test, y_test, verbose=0)
            print(activation, units, "accuracy:", round(acc, 4))


# ============================================================
# PRACTICAL 14
# Question: Build a feedforward network to classify breast tumours as
# malignant or benign; compare activations and optimizers using
# binary cross-entropy. Dataset: Breast Cancer Wisconsin
# ============================================================
def practical_14():
    import tensorflow as tf
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras import layers, models

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    for activation in ["relu", "tanh"]:
        for opt in ["adam", "sgd"]:
            model = models.Sequential([layers.Input((X_train.shape[1],)),
                layers.Dense(32, activation=activation), layers.Dense(16, activation=activation),
                layers.Dense(1, activation="sigmoid")])
            model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            _, acc = model.evaluate(X_test, y_test, verbose=0)
            print(activation, opt, "accuracy:", round(acc, 4))


# ============================================================
# PRACTICAL 15
# Question: Use a softmax output and categorical cross-entropy to classify
# wines into three classes; report accuracy, precision, recall and confusion matrix.
# Dataset: Wine Dataset
# ============================================================
def practical_15():
    import tensorflow as tf
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow.keras import layers, models

    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    model = models.Sequential([layers.Input((X_train.shape[1],)),
        layers.Dense(32, activation="relu"), layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    probs = model.predict(X_test, verbose=0)
    pred = probs.argmax(axis=1)
    print("Accuracy:", (pred == y_test).mean())
    print(classification_report(y_test, pred, target_names=data.target_names))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))


# ============================================================
# PRACTICAL 16
# Question: Build an LSTM to forecast future temperature from past
# readings and compare predicted vs actual values with MAE and RMSE.
# Dataset: Jena Climate CSV; expected column 'T (degC)'.
# ============================================================
def practical_16():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from tensorflow.keras import layers, models

    df = pd.read_csv("jena_climate_2009_2016.csv")
    values = df["T (degC)"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    values = scaler.fit_transform(values)
    lookback = 24
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i])
        y.append(values[i])
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    model = models.Sequential([layers.Input((lookback, 1)),
        layers.LSTM(32), layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    pred = model.predict(X_test, verbose=0)
    actual = scaler.inverse_transform(y_test)
    pred = scaler.inverse_transform(pred)
    print("MAE:", mean_absolute_error(actual, pred))
    print("RMSE:", np.sqrt(mean_squared_error(actual, pred)))
    plt.plot(actual[:200], label="Actual"); plt.plot(pred[:200], label="Predicted")
    plt.legend(); plt.title("Temperature forecast"); plt.show()


# ============================================================
# PRACTICAL 17
# Question: Add noise to images and train a denoising autoencoder;
# measure restoration quality using MSE or PSNR.
# Dataset: MNIST / Fashion-MNIST
# ============================================================
def practical_17():
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras import layers, models

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., None] / 255.0
    x_test = x_test.astype("float32")[..., None] / 255.0
    rng = np.random.default_rng(42)
    noisy_train = np.clip(x_train + rng.normal(0, 0.3, x_train.shape), 0, 1)
    noisy_test = np.clip(x_test + rng.normal(0, 0.3, x_test.shape), 0, 1)
    model = models.Sequential([layers.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2, padding="same"),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.UpSampling2D(2),
        layers.Conv2D(1, 3, activation="sigmoid", padding="same")])
    model.compile(optimizer="adam", loss="mse")
    model.fit(noisy_train, x_train, epochs=5, batch_size=128, validation_split=0.1)
    denoised = model.predict(noisy_test[:10], verbose=0)
    mse = np.mean((x_test[:10] - denoised) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    print("MSE:", mse, "PSNR:", psnr, "dB")
    fig, axes = plt.subplots(3, 10, figsize=(12, 4))
    for i in range(10):
        for row, arr in enumerate([x_test, noisy_test, denoised]):
            axes[row, i].imshow(arr[i].squeeze(), cmap="gray"); axes[row, i].axis("off")
    axes[0, 0].set_ylabel("Original"); axes[1, 0].set_ylabel("Noisy"); axes[2, 0].set_ylabel("Denoised")
    plt.show()


# ============================================================
# PRACTICAL 18
# Question: Train a CNN with rotation, flip, zoom and shift augmentation;
# compare accuracy and overfitting with and without augmentation.
# Dataset: Cats vs. Dogs folders: data/train and data/validation
# ============================================================
def practical_18():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    train_ds = tf.keras.utils.image_dataset_from_directory(
        "data/train", image_size=(160, 160), batch_size=32)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/validation", image_size=(160, 160), batch_size=32)
    augment = models.Sequential([layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1), layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1)])
    def build(use_aug):
        inp = layers.Input((160, 160, 3))
        x = augment(inp) if use_aug else inp
        x = layers.Rescaling(1./255)(x)
        x = layers.Conv2D(32, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        m = models.Model(inp, out)
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return m
    for flag in [False, True]:
        model = build(flag)
        hist = model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)
        print("Augmentation", flag, "final val accuracy:", hist.history["val_accuracy"][-1])


# ============================================================
# PRACTICAL 19
# Question: Use a pretrained object detector to locate and label objects,
# draw bounding boxes and evaluate with IoU or mAP. Dataset: Pascal VOC 2007.
# Requires a pretrained TensorFlow Hub detector and compatible image data.
# ============================================================
def practical_19():
    # Simple inference template using TensorFlow Hub SSD MobileNet.
    # Install: pip install tensorflow-hub pillow
    # Download an image to test and set IMAGE_PATH.
    import tensorflow as tf
    import tensorflow_hub as hub
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    IMAGE_PATH = "test_image.jpg"
    detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    image = Image.open(IMAGE_PATH).convert("RGB").resize((640, 640))
    arr = np.array(image)
    input_tensor = tf.convert_to_tensor(arr, dtype=tf.uint8)[tf.newaxis, ...]
    result = detector(input_tensor)
    boxes = result["detection_boxes"][0].numpy()
    scores = result["detection_scores"][0].numpy()
    classes = result["detection_classes"][0].numpy().astype(int)
    plt.imshow(arr)
    h, w, _ = arr.shape
    for box, score, cls in zip(boxes, scores, classes):
        if score < 0.5:
            continue
        y1, x1, y2, x2 = box
        plt.gca().add_patch(plt.Rectangle((x1*w, y1*h), (x2-x1)*w, (y2-y1)*h,
                                          fill=False, linewidth=2))
        plt.text(x1*w, y1*h, f"class {cls}: {score:.2f}")
    plt.axis("off"); plt.show()
    # IoU = intersection area / union area; compare predicted boxes to ground-truth boxes.
    # mAP requires ground-truth annotations and evaluation over all classes/images.
    print("Detections above threshold:", int((scores >= 0.5).sum()))


# ============================================================
# PRACTICAL 20
# Question: Train a character-level LSTM to generate text; experiment
# with sequence length, embedding size and sampling temperature.
# Dataset: Shakespeare text file named shakespeare.txt
# ============================================================
def practical_20():
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models

    text = open("shakespeare.txt", encoding="utf-8").read()
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = np.array(chars)
    encoded = np.array([char_to_id[c] for c in text], dtype=np.int32)
    seq_len = 100
    ds = tf.data.Dataset.from_tensor_slices(encoded)
    ds = ds.window(seq_len + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(seq_len + 1))
    ds = ds.map(lambda s: (s[:-1], s[1:])).batch(64).prefetch(tf.data.AUTOTUNE)
    model = models.Sequential([layers.Embedding(len(chars), 64),
        layers.LSTM(128, return_sequences=True), layers.Dense(len(chars))])
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(ds, epochs=5)
    def generate(seed="To be", length=300, temperature=0.8):
        result = seed
        for _ in range(length):
            seq = [char_to_id.get(c, 0) for c in result[-seq_len:]]
            seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=seq_len)
            logits = model(seq, training=False)[0, -1] / temperature
            next_id = int(tf.random.categorical(logits[None, :], 1)[0, 0])
            result += id_to_char[next_id]
        return result
    print(generate())


# ============================================================
# PRACTICAL 21
# Question: Build a sequence-labeling model (e.g. BiLSTM) for NER that
# tags persons, organizations and locations; report precision, recall and F1.
# Dataset: CoNLL-2003. Prepare token-level sentences and BIO labels first.
# ============================================================
def practical_21():
    # Input files should be preprocessed into sentence-level token and tag lists.
    # Example format: tokens_list = [["John", "works", "at", "Google"]]
    # tags_list = [["B-PER", "O", "O", "B-ORG"]]
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.metrics import classification_report

    tokens_list = [["John", "works", "at", "Google"], ["Mary", "lives", "in", "Paris"]]
    tags_list = [["B-PER", "O", "O", "B-ORG"], ["B-PER", "O", "O", "B-LOC"]]
    words = sorted({w for s in tokens_list for w in s})
    tags = sorted({t for s in tags_list for t in s})
    word2id = {w: i + 1 for i, w in enumerate(words)}
    tag2id = {t: i for i, t in enumerate(tags)}
    maxlen = max(map(len, tokens_list))
    X = tf.keras.preprocessing.sequence.pad_sequences(
        [[word2id[w] for w in s] for s in tokens_list], maxlen=maxlen, padding="post")
    y = tf.keras.preprocessing.sequence.pad_sequences(
        [[tag2id[t] for t in s] for s in tags_list], maxlen=maxlen, padding="post")
    model = models.Sequential([layers.Embedding(len(word2id)+1, 32, mask_zero=True),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.TimeDistributed(layers.Dense(len(tag2id), activation="softmax"))])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y[..., None], epochs=30, verbose=0)
    pred = model.predict(X, verbose=0).argmax(axis=-1)
    true_flat, pred_flat = [], []
    for i, seq in enumerate(tokens_list):
        for j in range(len(seq)):
            true_flat.append(tags[y[i, j]])
            pred_flat.append(tags[pred[i, j]])
    print(classification_report(true_flat, pred_flat, zero_division=0))


# ============================================================
# PRACTICAL 22
# Question: Build an attention-based encoder-decoder model to translate
# English sentences into French and evaluate with BLEU.
# Dataset: English–French (ManyThings). File format: English TAB French.
# ============================================================
def practical_22():
    # Compact teacher-forcing encoder-decoder example; use a larger dataset for quality.
    # Download fra-eng.zip from ManyThings and set DATA_PATH to fra.txt.
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split

    DATA_PATH = "fra.txt"
    pairs = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\\t")
            if len(parts) >= 2:
                pairs.append((parts[0].lower(), "start " + parts[1].lower() + " end"))
            if len(pairs) >= 10000:
                break
    eng, fra = zip(*pairs)
    eng_vec = layers.TextVectorization(max_tokens=10000, output_sequence_length=20)
    fra_vec = layers.TextVectorization(max_tokens=10000, output_sequence_length=20)
    eng_vec.adapt(eng); fra_vec.adapt(fra)
    X = eng_vec(np.array(eng))
    Y = fra_vec(np.array(fra)).numpy()
    dec_in, dec_out = Y[:, :-1], Y[:, 1:]
    enc_in = layers.Input((20,), dtype="int64")
    enc_emb = layers.Embedding(10000, 128, mask_zero=True)(enc_in)
    enc_out, state_h, state_c = layers.LSTM(128, return_state=True)(enc_emb)
    dec_input = layers.Input((19,), dtype="int64")
    dec_emb = layers.Embedding(10000, 128, mask_zero=True)(dec_input)
    dec_seq = layers.LSTM(128, return_sequences=True)(dec_emb, initial_state=[state_h, state_c])
    # Attention over a one-vector encoder context is a minimal illustrative example.
    context = layers.RepeatVector(19)(enc_out)
    attn = layers.Attention()([dec_seq, context])
    merged = layers.Concatenate()([dec_seq, attn])
    out = layers.TimeDistributed(layers.Dense(10000, activation="softmax"))(merged)
    model = models.Model([enc_in, dec_input], out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit([X, dec_in], dec_out[..., None], epochs=3, batch_size=64, validation_split=0.1)
    print("Translation model trained. Evaluate generated translations with BLEU using nltk.translate.bleu_score.")
    print("Note: for accurate BLEU, split into train/test and decode token IDs back to words.")


# ============================================================
# PRACTICAL 23
# Question: Build a VAE that learns a latent space of digit images,
# visualize the latent space and generate digits by sampling.
# Dataset: MNIST
# ============================================================
def practical_23():
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras import layers, models

    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
    latent_dim = 2
    inp = layers.Input((784,))
    h = layers.Dense(256, activation="relu")(inp)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    def sample(args):
        mean, log_var = args
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps
    z = layers.Lambda(sample)([z_mean, z_log_var])
    encoder = models.Model(inp, [z_mean, z_log_var, z])
    latent_in = layers.Input((latent_dim,))
    x = layers.Dense(256, activation="relu")(latent_in)
    out = layers.Dense(784, activation="sigmoid")(x)
    decoder = models.Model(latent_in, out)

    class VAE(tf.keras.Model):
        def train_step(self, data):
            if isinstance(data, tuple): data = data[0]
            with tf.GradientTape() as tape:
                mean, log_var, z_sample = encoder(data)
                reconstruction = decoder(z_sample)
                rec = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=-1))
                kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1))
                loss = rec + kl
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {"loss": loss, "reconstruction_loss": rec, "kl_loss": kl}
    vae = VAE()
    vae.compile(optimizer="adam")
    vae.fit(x_train, epochs=5, batch_size=128)
    z_points = encoder.predict(x_train[:2000], verbose=0)[0]
    plt.scatter(z_points[:, 0], z_points[:, 1], s=3)
    plt.title("VAE latent space"); plt.show()
    generated = decoder.predict(np.random.normal(size=(10, latent_dim)), verbose=0)
    fig, axes = plt.subplots(1, 10, figsize=(12, 2))
    for ax, img in zip(axes, generated):
        ax.imshow(img.reshape(28, 28), cmap="gray"); ax.axis("off")
    plt.show()


# ============================================================
# PRACTICAL 24
# Question: Train an autoencoder on normal transactions only, flag fraud
# using high reconstruction error, select a threshold, and report precision,
# recall and F1. Dataset: Credit Card Fraud Detection (creditcard.csv).
# ============================================================
def practical_24():
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow.keras import layers, models

    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    normal_train = X_train[y_train == 0]
    inp = layers.Input((X.shape[1],))
    x = layers.Dense(32, activation="relu")(inp)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(X.shape[1])(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(normal_train, normal_train, epochs=10, batch_size=256,
              validation_split=0.1, verbose=1)
    val_errors = np.mean((normal_train - model.predict(normal_train, verbose=0))**2, axis=1)
    threshold = np.percentile(val_errors, 95)
    test_errors = np.mean((X_test - model.predict(X_test, verbose=0))**2, axis=1)
    pred = (test_errors > threshold).astype(int)
    print("Threshold:", threshold)
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["Normal", "Fraud"], zero_division=0))


# ============================================================
# PRACTICAL 25
# Question: Build a CNN segmentation model (e.g. U-Net) that labels each
# image pixel, display predicted masks and evaluate IoU.
# Dataset: Oxford-IIIT Pet. Requires image and segmentation-mask files.
# ============================================================
def practical_25():
    # Put images in data/images/ and corresponding masks in data/masks/
    # Use matching filenames. Masks should be integer class IDs, e.g. 0=background,
    # 1=pet, 2=border. Resize images and masks to 128x128.
    import os
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from tensorflow.keras import layers, models

    IMG_DIR, MASK_DIR = "data/images", "data/masks"
    SIZE = 128
    image_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png"))])
    images, masks = [], []
    for name in image_files:
        mask_name = os.path.splitext(name)[0] + ".png"
        if not os.path.exists(os.path.join(MASK_DIR, mask_name)):
            continue
        img = tf.keras.utils.load_img(os.path.join(IMG_DIR, name), target_size=(SIZE, SIZE))
        mask = tf.keras.utils.load_img(os.path.join(MASK_DIR, mask_name),
                                       target_size=(SIZE, SIZE), color_mode="grayscale",
                                       interpolation="nearest")
        images.append(tf.keras.utils.img_to_array(img) / 255.0)
        masks.append(tf.keras.utils.img_to_array(mask).squeeze().astype("int32"))
    X, Y = np.array(images), np.array(masks)
    # If masks are encoded as 0/128/255, map to 0/1/2 class IDs.
    if Y.max() > 2:
        Y = np.where(Y < 64, 0, np.where(Y < 192, 1, 2))
    Y = np.clip(Y, 0, 2)
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        return x
    inp = layers.Input((SIZE, SIZE, 3))
    c1 = conv_block(inp, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
    b = conv_block(p3, 256)
    u3 = layers.UpSampling2D()(b); u3 = layers.Concatenate()([u3, c3]); c4 = conv_block(u3, 128)
    u2 = layers.UpSampling2D()(c4); u2 = layers.Concatenate()([u2, c2]); c5 = conv_block(u2, 64)
    u1 = layers.UpSampling2D()(c5); u1 = layers.Concatenate()([u1, c1]); c6 = conv_block(u1, 32)
    out = layers.Conv2D(3, 1, activation="softmax")(c6)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, Y, epochs=5, batch_size=8, validation_split=0.2)
    pred = model.predict(X[:3], verbose=0).argmax(axis=-1)
    for i in range(min(3, len(X))):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].imshow(X[i]); ax[0].set_title("Image")
        ax[1].imshow(Y[i]); ax[1].set_title("Ground truth")
        ax[2].imshow(pred[i]); ax[2].set_title("Predicted mask")
        for a in ax: a.axis("off")
        plt.show()
    # Mean IoU for a quick pixel-level evaluation
    if len(X):
        print("Mean IoU:", tf.keras.metrics.MeanIoU(num_classes=3)(
            Y.flatten(), pred.flatten()).numpy())


# Run one selected practical by uncommenting its function call.
# Example: practical_01()
