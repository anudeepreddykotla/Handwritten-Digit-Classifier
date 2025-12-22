import numpy as np
import gzip

class DigitClassifier:
    def __init__(self):
        self.layer1 = np.zeros((784, 1))
        self.layer2 = np.zeros((128, 1))
        self.layer3 = np.zeros((64, 1))
        self.layer4 = np.zeros((10, 1))

        self.w1 = np.random.uniform(-np.sqrt(1/784), np.sqrt(1/784), (128, 784))
        self.w2 = np.random.uniform(-np.sqrt(1/128), np.sqrt(1/128), (64, 128))
        self.w3 = np.random.uniform(-np.sqrt(1/64),  np.sqrt(1/64),  (10, 64))

        self.b1 = np.zeros((128, 1))
        self.b2 = np.zeros((64, 1))
        self.b3 = np.zeros((10, 1))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def classify(self, x):
        self.layer1 = x

        self.z2 = self.w1 @ self.layer1 + self.b1
        self.layer2 = self.sigmoid(self.z2)

        self.z3 = self.w2 @ self.layer2 + self.b2
        self.layer3 = self.sigmoid(self.z3)

        self.z4 = self.w3 @ self.layer3 + self.b3
        self.layer4 = self.softmax(self.z4)

        return int(np.argmax(self.layer4))

    def loss(self, label):
        self.desired = np.zeros((10, 1))
        self.desired[label] = 1.0
        return -np.sum(self.desired * np.log(self.layer4 + 1e-9))

    def deltas(self):
        delta4 = self.layer4 - self.desired
        delta3 = (self.w3.T @ delta4) * self.sigmoid_prime(self.z3)
        delta2 = (self.w2.T @ delta3) * self.sigmoid_prime(self.z2)
        return delta2, delta3, delta4

    def derivatives(self):
        delta2, delta3, delta4 = self.deltas()

        dw3 = delta4 @ self.layer3.T
        db3 = delta4

        dw2 = delta3 @ self.layer2.T
        db2 = delta3

        dw1 = delta2 @ self.layer1.T
        db1 = delta2

        return dw1, db1, dw2, db2, dw3, db3

    def SGD(self, mini_batch, lr):
        sum_dw1 = np.zeros_like(self.w1)
        sum_db1 = np.zeros_like(self.b1)
        sum_dw2 = np.zeros_like(self.w2)
        sum_db2 = np.zeros_like(self.b2)
        sum_dw3 = np.zeros_like(self.w3)
        sum_db3 = np.zeros_like(self.b3)

        for x, y in mini_batch:
            self.classify(x)
            self.loss(y)

            dw1, db1, dw2, db2, dw3, db3 = self.derivatives()

            sum_dw1 += dw1
            sum_db1 += db1
            sum_dw2 += dw2
            sum_db2 += db2
            sum_dw3 += dw3
            sum_db3 += db3

        m = len(mini_batch)
        self.w1 -= lr * (sum_dw1 / m)
        self.b1 -= lr * (sum_db1 / m)
        self.w2 -= lr * (sum_dw2 / m)
        self.b2 -= lr * (sum_db2 / m)
        self.w3 -= lr * (sum_dw3 / m)
        self.b3 -= lr * (sum_db3 / m)

def load_idx_images(path):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_idx_labels(path):
    with gzip.open(path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return labels

def fix_emnist(img):
    img = np.rot90(img, k=1)
    img = np.fliplr(img)
    return img


def create_mini_batches(images, labels, batch_size):
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    for start in range(0, len(images), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield [
            (images[i].reshape(784, 1), labels[i])
            for i in batch_idx
        ]

def evaluate_accuracy(network, images, labels):
    correct = 0
    for i in range(len(images)):
        x = images[i].reshape(784, 1)
        if network.classify(x) == labels[i]:
            correct += 1
    return correct / len(images)


def train_emnist(network, epochs, batch_size, lr):
    train_images = load_idx_images("emnist-digits-train-images-idx3-ubyte.gz")
    train_labels = load_idx_labels("emnist-digits-train-labels-idx1-ubyte.gz")

    test_images = load_idx_images("emnist-digits-test-images-idx3-ubyte.gz")
    test_labels = load_idx_labels("emnist-digits-test-labels-idx1-ubyte.gz")

    train_images = np.array([fix_emnist(img) for img in train_images])
    test_images  = np.array([fix_emnist(img) for img in test_images])

    train_images = train_images.astype(np.float32) / 255.0
    test_images  = test_images.astype(np.float32) / 255.0

    train_images = train_images.reshape(-1, 784)
    test_images  = test_images.reshape(-1, 784)

    for epoch in range(epochs):
        mini_batches = create_mini_batches(
            train_images, train_labels, batch_size
        )

        for mini_batch in mini_batches:
            network.SGD(mini_batch, lr)

        acc = evaluate_accuracy(network, test_images, test_labels)
        print(f"Epoch {epoch+1}: EMNIST Test Accuracy = {acc*100:.2f}%")

if __name__ == "__main__":
    net = DigitClassifier()

    train_emnist(
        network=net,
        epochs=10,
        batch_size=64,
        lr=0.005
    )

    import pickle
    with open("emnist_model.pkl", "wb") as f:
        pickle.dump(
            (net.w1, net.b1, net.w2, net.b2, net.w3, net.b3), f
        )