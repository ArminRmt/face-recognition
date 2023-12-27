import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Function to load images from the specified subset
def load_images(subset):
    data = []
    labels = []
    base_folder = "./YaleB"

    # Get the list of subfolders in YaleB
    subfolders = [
        folder
        for folder in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, folder))
    ]

    # Randomly select one subfolder
    selected_subfolder = random.choice(subfolders)

    # Load images from the selected subfolder
    person_folder = os.path.join(base_folder, str(selected_subfolder))
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(160, 160))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        data.append(img_array)
        labels.append(selected_subfolder)

    return np.array(data), np.array(labels)


# Load index.txt
subset_II = [2, 5, 10, 11, 12, 13, 15, 39, 40, 41, 42, 44]
subset_train = [
    1,
    7,
    8,
    9,
    36,
    37,
    3,
    6,
    14,
    16,
    17,
    19,
    20,
    43,
    45,
    46,
    48,
    49,
    18,
    21,
    22,
    23,
    24,
    25,
    26,
    47,
    50,
    51,
    52,
    53,
    54,
    55,
    4,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
]

# Load images
train_data, train_labels = load_images(subset_train)
test_data, test_labels = load_images(subset_II)


# Normalize pixel values to be between 0 and 1
train_data, test_data = train_data / 255.0, test_data / 255.0

# Define the model
model = models.Sequential(
    [
        layers.Conv2D(16, (3, 3), activation="relu", input_shape=(160, 160, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((4, 4)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.AveragePooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(148),
    ]
)

# Compile the model
model.compile(
    # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.001),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    train_data,
    train_labels,
    epochs=100,
    batch_size=8,
    validation_data=(test_data, test_labels),
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Predict on the test set
predictions = np.argmax(model.predict(test_data), axis=1)

# Calculate F1 score
f1 = f1_score(test_labels, predictions, average="weighted")
print(f"Weighted F1 score: {f1}")

# Plot the training loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
