import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
from NN import DigitClassifier


# =========================
# EMNIST loading utilities
# =========================

def load_idx_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_idx_labels(path):
    with gzip.open(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

def fix_emnist(img):
    return np.fliplr(np.rot90(img, k=1))


# =========================
# Main test logic
# =========================

def main():
    # Load trained model
    net = DigitClassifier()
    with open("emnist_model.pkl", "rb") as f:
        net.w1, net.b1, net.w2, net.b2, net.w3, net.b3 = pickle.load(f)

    # Load EMNIST test set
    test_images = load_idx_images(
        "emnist-digits-test-images-idx3-ubyte.gz"
    )
    test_labels = load_idx_labels(
        "emnist-digits-test-labels-idx1-ubyte.gz"
    )

    # Fix orientation and normalize
    test_images = np.array([fix_emnist(img) for img in test_images])
    test_images = test_images.astype(np.float32) / 255.0

    # Random test sample
    idx = np.random.randint(len(test_images))
    x = test_images[idx].reshape(784, 1)
    true_label = test_labels[idx]

    # Predict
    pred = net.classify(x)

    # Display
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.title(f"True: {true_label} | Predicted: {pred}")
    plt.axis("off")
    plt.show()

    print("True label :", true_label)
    print("Predicted  :", pred)


if __name__ == "__main__":
    main()
