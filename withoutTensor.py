import os
import random
import numpy as np
from PIL import Image


# Define activation functions
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Define convolution operation
def conv2d(x, kernel, stride, padding):
    _, _, h, w = x.shape
    _, _, kh, kw = kernel.shape
    h_out = (h - kh + 2 * padding) // stride + 1
    w_out = (w - kw + 2 * padding) // stride + 1

    x_padded = np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )

    output = np.zeros((x.shape[0], kernel.shape[0], h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            x_slice = x_padded[
                :, :, i * stride : i * stride + kh, j * stride : j * stride + kw
            ]
            output[:, :, i, j] = np.sum(x_slice * kernel, axis=(2, 3))

    return output


# Define max pooling operation
def max_pooling(x, pool_size, stride):
    _, _, h, w = x.shape
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1

    output = np.zeros((x.shape[0], x.shape[1], h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            x_slice = x[
                :,
                :,
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size,
            ]
            output[:, :, i, j] = np.max(x_slice, axis=(2, 3))

    return output


def batch_norm(x, epsilon=1e-5):
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2, 3), keepdims=True)
    x_normalized = (x - mean) / np.sqrt(var + epsilon)
    return x_normalized


def avg_pooling(x, pool_size, stride):
    _, _, h, w = x.shape
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1

    output = np.zeros((x.shape[0], x.shape[1], h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            x_slice = x[
                :,
                :,
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size,
            ]
            output[:, :, i, j] = np.mean(x_slice, axis=(2, 3))

    return output


# Define fully connected layer
def fully_connected(x, weights, biases):
    return np.dot(x, weights) + biases


# Function to build the CNN model
def build_cnn_model(input_shape=(3, 160, 160), num_classes=148):
    model = {
        "conv1": {
            "weights": np.random.randn(16, 3, 3, 3) * 0.01,
            "bias": np.zeros((16, 1)),
        },
        "batch_norm1": {},
        "relu1": {},
        "maxpool1": {"pool_size": 4, "stride": 4},
        "conv2": {
            "weights": np.random.randn(32, 16, 3, 3) * 0.01,
            "bias": np.zeros((32, 1)),
        },
        "batch_norm2": {},
        "relu2": {},
        "maxpool2": {"pool_size": 2, "stride": 2},
        "conv3": {
            "weights": np.random.randn(64, 32, 3, 3) * 0.01,
            "bias": np.zeros((64, 1)),
        },
        "batch_norm3": {},
        "relu3": {},
        "avgpool": {"pool_size": 2, "stride": 2},
        "fc1": {
            "weights": np.random.randn(256, 64 * 2 * 2) * 0.01,
            "bias": np.zeros((256, 1)),
        },
        "relu4": {},
        "fc2": {
            "weights": np.random.randn(num_classes, 256) * 0.01,
            "bias": np.zeros((num_classes, 1)),
        },
    }

    return model


# Function to forward pass through the CNN model
def forward_pass(x, model):
    # Layer 1: Convolution, Batch Normalization, ReLU, Max Pooling
    x = conv2d(x, model["conv1"]["weights"], stride=1, padding=0)

    x = batch_norm(x)
    x = relu(x)
    x = max_pooling(x, model["maxpool1"]["pool_size"], model["maxpool1"]["stride"])

    # Layer 2: Convolution, Batch Normalization, ReLU, Max Pooling
    x = conv2d(x, model["conv2"]["weights"], stride=1, padding=0)
    x = batch_norm(x)
    x = relu(x)
    x = max_pooling(x, model["maxpool2"]["pool_size"], model["maxpool2"]["stride"])

    # Layer 3: Convolution, Batch Normalization, ReLU, Average Pooling
    x = conv2d(x, model["conv3"]["weights"], stride=1, padding=0)
    x = batch_norm(x)
    x = relu(x)
    x = avg_pooling(x, model["avgpool"]["pool_size"], model["avgpool"]["stride"])

    # Flatten
    x = x.reshape(x.shape[0], -1)

    # Fully Connected Layer 1
    x = fully_connected(x, model["fc1"]["weights"], model["fc1"]["bias"])
    x = relu(x)

    # Fully Connected Layer 2 (Output layer)
    x = fully_connected(x, model["fc2"]["weights"], model["fc2"]["bias"])
    output = softmax(x)

    return output


# Your load_data method
def load_data(data_directory, subset_indices):
    chosen_persons = random.sample(os.listdir(data_directory), 2)
    train_data, test_data, train_labels, test_labels = [], [], [], []

    for person in chosen_persons:
        person_path = os.path.join(data_directory, person)

        # Collect images for training set
        for filename in os.listdir(person_path):
            split_filename = filename.split("P")
            if len(split_filename) == 2:
                img_index = int(split_filename[1].split(".")[0])
                if img_index not in subset_indices["Subset II"]:
                    img_path = os.path.join(person_path, filename)
                    img = Image.open(img_path).convert("RGB")
                    #  NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, and LANCZOS
                    img = img.resize(image_size, Image.BICUBIC)
                    img_array = np.array(img)
                    train_data.append(img_array)
                    train_labels.append(person)

        # Collect images for test set (Subset II)
        for img_index in subset_indices["Subset II"]:
            img_path = os.path.join(person_path, f"{img_index}.tif")
            img = Image.open(img_path).convert("RGB")
            img = img.resize(image_size, Image.BICUBIC)
            img_array = np.array(img)
            test_data.append(img_array)
            test_labels.append(person)

    return (
        np.array(train_data),
        np.array(test_data),
        np.array(train_labels),
        np.array(test_labels),
    )


if __name__ == "__main__":
    # Dummy values for image_size (replace with actual values)
    image_size = (160, 160)

    # Load data using your method
    data_directory = "./YaleB"  # Replace with the actual path to your YaleB folder
    subset_indices = {
        "Subset I": [1, 7, 8, 9, 36, 37],
        "Subset II": [2, 5, 10, 11, 12, 13, 15, 39, 40, 41, 42, 44],
        "Subset III": [3, 6, 14, 16, 17, 19, 20, 43, 45, 46, 48, 49],
        "Subset IV": [18, 21, 22, 23, 24, 25, 26, 47, 50, 51, 52, 53, 54, 55],
        "Subset V": [
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
        ],
    }

    X_train, X_test, y_train, y_test = load_data(data_directory, subset_indices)

    # Hyperparameters
    learning_rate = 0.001
    l2_reg = 0.001
    batch_size = 8
    num_epochs = 100

    # Build model
    model = build_cnn_model()

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0  # Initialize the loss for the epoch

        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]

            # Forward pass
            predictions = forward_pass(batch_x, model)

            # Compute loss (cross-entropy with L2 regularization)
            loss = -np.sum(np.log(predictions) * batch_y) / batch_size
            loss += (
                0.5
                * l2_reg
                * (
                    np.sum(model["conv1"]["weights"] ** 2)
                    + np.sum(model["conv2"]["weights"] ** 2)
                    + np.sum(model["conv3"]["weights"] ** 2)
                    # + np.sum(model["conv4"]["weights"] ** 2)
                    # + np.sum(model["conv5"]["weights"] ** 2)
                    + np.sum(model["fc1"]["weights"] ** 2)
                    + np.sum(model["fc2"]["weights"] ** 2)
                )
            )

            # Backward pass (gradient descent)
            # Note: This is a simple implementation and does not include a complete backpropagation algorithm.
            # In practice, you would typically use a deep learning framework like TensorFlow or PyTorch for this.

            # Update weights using gradient descent
            for layer in model.keys():
                if "weights" in model[layer]:
                    model[layer]["weights"] -= learning_rate * np.gradient(
                        loss, model[layer]["weights"]
                    )
                if "bias" in model[layer]:
                    model[layer]["bias"] -= learning_rate * np.sum(
                        np.gradient(loss, model[layer]["bias"])
                    )

            # Accumulate the loss for the epoch
            epoch_loss += loss

        # Compute the average loss over all batches in the epoch
        if len(X_train) // batch_size == 0:
            avg_epoch_loss = epoch_loss
        else:
            avg_epoch_loss = epoch_loss / (len(X_train) // batch_size)

        # Print training loss for each epoch
        # print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss}")

    # Testing (similar to training loop, but using test data)
    # Replace this part with actual testing logic based on your dataset
    for i in range(0, len(X_test), batch_size):
        batch_x_test = X_test[i : i + batch_size]
        batch_y_test = y_test[i : i + batch_size]
        predictions_test = forward_pass(batch_x_test, model)

    # Evaluate the model on the test set
    # Implement your evaluation logic based on your problem (e.g., accuracy, precision, recall)
    # Replace the following with actual evaluation metrics based on your needs
    accuracy = np.mean(np.argmax(predictions_test, axis=1) == batch_y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
