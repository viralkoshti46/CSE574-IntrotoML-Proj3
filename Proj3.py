__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

import struct
import numpy as np
import matplotlib.pyplot as plt


def read_labels(file):
    """Reads an MNIST label file and returns the labels."""
    with open(file, "rb") as f:
        magic_number, number_of_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # MNIST label files should have magic number 2049
    if magic_number != 2049:
        raise ValueError("This isn't a valid label file.")

    if len(labels) != number_of_labels:
        print("Number of labels does not match the file header.")

    return labels


def read_images(file):
    """Reads an MNIST image file and returns the images."""
    with open(file, "rb") as f:
        magic_number, number_of_images, rows, columns = struct.unpack(
            ">IIII", f.read(16)
        )
        pixels = np.frombuffer(f.read(), dtype=np.uint8)

    # MNIST image files should have magic number 2051
    if magic_number != 2051:
        raise ValueError("This isn't a valid image file.")

    # Every image is 28 x 28 = 784 pixels
    expected_pixels = number_of_images * rows * columns

    if len(pixels) != expected_pixels:
        raise ValueError("Number of pixels does not match the file header.")

    images = pixels.reshape(number_of_images, rows * columns)

    # use float values because the weights are floating point values
    images = images.astype(float)

    return images


# Helper functions for  perceptron
def step_function(val):
    """Returns 1 when the value is >= 0, otherwise returns 0."""
    return (val >= 0).astype(float)


def desired_output(label):
    """Creates the 10 element desired output vector for one digit."""
    out = np.zeros(10)
    out[int(label)] = 1
    return out


def count_errors(weight_matrix, X_data, y_data):
    """
    Counts how many samples are classified incorrectly.
    The predicted class is the index with the largest value in W*x.
    """

    scores = X_data @ weight_matrix.T
    preds = np.argmax(scores, axis=1)

    # compare predictions with actual labels
    wrong = np.sum(preds != y_data)
    return int(wrong)


# Training loop...
def train_perceptron(
    X_train, y_train, n_samples, eta, epsilon, seed_val=0, max_epochs=None
):
    """
    Trains the multicategory perceptron using the first n training samples.
    Returns the final weights and the errors for every epoch.
    """

    X = X_train[:n_samples]
    y = y_train[:n_samples]

    # fixed seed makes the results the same every time the program is run
    np.random.seed(seed_val)

    # random 10 x 784 weight matrix
    W = np.random.uniform(-0.5, 0.5, (10, 784))

    err_list = []
    epoch_count = 0

    while True:
        # Count the errors before updating the weights
        errs = count_errors(W, X, y)
        err_list.append(errs)

        err_ratio = errs / n_samples
        print(
            "Epoch "
            + str(epoch_count)
            + ": "
            + str(errs)
            + " errors out of "
            + str(n_samples)
            + " (ratio: "
            + str(round(err_ratio, 5))
            + ")"
        )

        if err_ratio <= epsilon:
            print("Reached epsilon threshold! Breaking out...")
            break

        if max_epochs is not None and epoch_count >= max_epochs:
            print("Hit max epochs limit, stopping.")
            break

        # Update the weights using each training sample
        for i in range(n_samples):
            x_i = X[i]

            # v = W * x
            v = W @ x_i

            # Apply the step function to every output neuron
            actual = step_function(v)

            # Get the desired one-hot output using the sample label
            target = desired_output(y[i])

            # d(x_i) - u(W*x_i)
            diff = target - actual

            # W = W + eta(d - u) * x^T
            # Only rows with a nonzero difference need to be changed.
            update_indices = np.where(diff != 0)[0]
            if len(update_indices) > 0:
                W[update_indices] += eta * diff[update_indices, None] * x_i[None, :]

        epoch_count += 1

    return W, err_list


def test_perceptron(W, X_test, y_test):
    """Tests the trained perceptron on all 10,000 test images."""

    errs = count_errors(W, X_test, y_test)
    err_pct = (errs / len(y_test)) * 100
    acc = 100.0 - err_pct
    return errs, err_pct, acc


def plot_errors(errors, title_text, filename):
    """Plots epoch number against the number of misclassification errors."""

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(len(errors)), errors, marker="o", color="purple"
    )  # changed color cus why not
    plt.xlabel("Epoch")
    plt.ylabel("Errors")
    plt.title(title_text)
    plt.grid(True)
    plt.savefig(filename, dpi=150)
    plt.close()


def run_experiment(
    name, X_train, y_train, X_test, y_test, n, eta, epsilon, max_epochs=None
):
    """Runs training, testing, prints the results, and creates the plot."""

    print("\n------------------------------------------------------------")
    print("Running experiment: " + name)
    print(f"Params -> n: {n}, eta: {eta}, epsilon: {epsilon}")
    print("------------------------------------------------------------")

    W, errors = train_perceptron(
        X_train, y_train, n, eta, epsilon, seed_val=0, max_epochs=max_epochs
    )

    print("\nTraining done. Final errors:", errors[-1])

    if X_test is not None and y_test is not None:
        test_errs, test_pct, accuracy = test_perceptron(W, X_test, y_test)
        print(
            f"Test results -> Errors: {test_errs}, Pct: {test_pct:.2f}%, Accuracy: {accuracy:.2f}%"
        )

    plot_file = name + "_plot.png"
    plot_errors(errors, name + " Errors vs Epoch", plot_file)
    print(f"Saved plot to {plot_file}\n")

    return W, errors


if __name__ == "__main__":
    # File paths
    TRAIN_IMAGES_FILE = "train-images.idx3-ubyte"
    TRAIN_LABELS_FILE = "train-labels.idx1-ubyte"
    TEST_IMAGES_FILE = "t10k-images.idx3-ubyte"
    TEST_LABELS_FILE = "t10k-labels.idx1-ubyte"

    # Load everything up
    print("Loading mnist data files...")
    X_train = read_images(TRAIN_IMAGES_FILE)
    y_train = read_labels(TRAIN_LABELS_FILE)
    X_test = read_images(TEST_IMAGES_FILE)
    y_test = read_labels(TEST_LABELS_FILE)
    print("Data loaded successfully!!")

    # Part F
    run_experiment(
        "Part_F", X_train, y_train, X_test, y_test, n=50, eta=1, epsilon=0.001
    )

    # Part G
    run_experiment(
        "Part_G", X_train, y_train, X_test, y_test, n=1000, eta=1, epsilon=0.001
    )

    # Part H
    # Since epsilon is 0, the program would only stop if there are exactly
    # 0 errors. The full MNIST data may not converge, so 30 epochs are used
    # here to observe how the number of errors changes.
    run_experiment(
        "Part_H", X_train, y_train, None, None, n=60000, eta=1, epsilon=0, max_epochs=30
    )
