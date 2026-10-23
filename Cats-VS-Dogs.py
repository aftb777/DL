# Transfer Learning for Image Classification: Use a pretrained CNN (e.g. VGG16, MobileNetV2) to classify cat and dog images, and compare
# feature extraction with fine-tuning.


import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Load Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(160, 160),
    batch_size=32,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(160, 160),
    batch_size=32,
    label_mode="binary"
)

print("Classes:", train_ds.class_names)

# 2. Load Pretrained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)

# 3. Feature Extraction
# Freeze pretrained layers
base_model.trainable = False

inputs = layers.Input(shape=(160, 160, 3))

x = preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

# 4. Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 5. Train Feature Extraction Model
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

loss1, acc1 = model.evaluate(val_ds, verbose=0)

print("Feature Extraction Accuracy:", acc1 * 100, "%")

# 6. Fine-Tuning
# Unfreeze some pretrained layers
base_model.trainable = True

# Freeze all except last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 7. Train Fine-Tuned Model
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2
)

loss2, acc2 = model.evaluate(val_ds, verbose=0)

print("Fine-Tuning Accuracy:", acc2 * 100, "%")

# 8. Compare Results
print("\nAccuracy Comparison")
print("Feature Extraction:", round(acc1 * 100, 2), "%")
print("Fine-Tuning:", round(acc2 * 100, 2), "%")

# 9. Plot Accuracy
plt.plot(history1.history["val_accuracy"],
         label="Feature Extraction")

plt.plot(history2.history["val_accuracy"],
         label="Fine-Tuning")

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Transfer Learning Comparison")
plt.legend()
plt.show()