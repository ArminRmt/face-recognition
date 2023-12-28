import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
BASE_FOLDER = "./YaleB"
TARGET_SIZE = (160, 160)
SUBSET_II = [2, 5, 10, 11, 12, 13, 15, 39, 40, 41, 42, 44]
SUBSET_TRAIN = [i for i in range(1, 66) if i not in SUBSET_II]


def load_images(subset):
    train_data, train_labels, test_data, test_labels = [], [], [], []

    subfolders = [
        folder
        for folder in os.listdir(BASE_FOLDER)
        if os.path.isdir(os.path.join(BASE_FOLDER, folder))
    ]
    selected_subfolders = random.sample(subfolders, 2)

    for selected_subfolder in selected_subfolders:
        person_folder = os.path.join(BASE_FOLDER, str(selected_subfolder))
        for img_name in os.listdir(person_folder):
            img_number = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(person_folder, img_name)
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=TARGET_SIZE
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)

            if img_number in subset:
                train_data.append(img_array)
                train_labels.append(selected_subfolder)
            else:
                test_data.append(img_array)
                test_labels.append(selected_subfolder)

    return (
        np.array(train_data),
        np.array(train_labels),
        np.array(test_data),
        np.array(test_labels),
    )


def preprocess_labels(train_labels, test_labels):
    label_mapping = {label: i for i, label in enumerate(np.unique(train_labels))}
    train_labels = np.array([label_mapping[label] for label in train_labels])
    test_labels = np.array([label_mapping[label] for label in test_labels])
    return train_labels, test_labels


def normalize_data(train_data, test_data):
    return train_data / 66, test_data / 2


def create_model(input_shape):
    model = models.Sequential(
        [
            layers.Conv2D(16, (3, 3), activation="relu", input_shape=input_shape),
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
    return model


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def train_model(model, train_data, train_labels, test_data, test_labels):
    history = model.fit(
        train_data,
        train_labels,
        epochs=100,
        batch_size=8,
        validation_data=(test_data, test_labels),
    )
    return history


def evaluate_model(model, test_data, test_labels):
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print(f"\nTest accuracy: {test_acc}")


def calculate_weighted_f1(true_labels, predicted_labels):
    weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted")
    print(f"Weighted F1 score: {weighted_f1}")


def main():
    train_data, train_labels, test_data, test_labels = load_images(SUBSET_TRAIN)
    train_labels, test_labels = preprocess_labels(train_labels, test_labels)

    # Normalize pixel values
    train_data, test_data = normalize_data(train_data, test_data)

    # Define the model
    model = create_model(input_shape=(160, 160, 3))

    # Compile the model
    compile_model(model)

    # Train the model
    history = train_model(model, train_data, train_labels, test_data, test_labels)

    # Evaluate the model
    evaluate_model(model, test_data, test_labels)

    predictions = np.argmax(model.predict(test_data), axis=1)
    # Calculate and print weighted F1 score
    calculate_weighted_f1(test_labels, predictions)

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
