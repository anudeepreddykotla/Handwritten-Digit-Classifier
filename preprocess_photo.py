import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from NN import DigitClassifier
import pickle

path = "7.jpeg"

def preprocess_photo(path):
    # 1. Load grayscale
    img = Image.open(path).convert("L")
    img = np.array(img, dtype=np.float32)

    # 2. Invert (MNIST style)
    img = 255 - img
    img /= 255.0

    # 3. Remove background noise
    img[img < 0.4] = 0.0

    # 4. Crop bounding box
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        raise ValueError("No digit found")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    img = img[y0:y1+1, x0:x1+1]

    # 5. Resize largest side to 20
    h, w = img.shape
    scale = 20.0 / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img = np.array(img, dtype=np.float32) / 255.0

    # 6. Pad to 28×28
    canvas = np.zeros((28, 28))
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

    # 7. Center by center-of-mass
    total = np.sum(canvas)
    if total > 0:
        ys, xs = np.indices(canvas.shape)
        cy = np.sum(ys * canvas) / total
        cx = np.sum(xs * canvas) / total
        canvas = np.roll(canvas, int(14 - cy), axis=0)
        canvas = np.roll(canvas, int(14 - cx), axis=1)

    return canvas.reshape(784, 1), canvas


if __name__ == "__main__":
    for i in range(0,10):
        path = f"{i}.jpeg"
        x, vis = preprocess_photo(path)
        x = x.reshape(28, 28)
        x = np.rot90(x, 2)
        x = x.reshape(784, 1)
        plt.imshow(vis, cmap="gray")
        plt.title("Network input")
        plt.axis("off")
        plt.show()

        net = DigitClassifier()
        with open("emnist_model.pkl", "rb") as f:
            net.w1, net.b1, net.w2, net.b2, net.w3, net.b3 = pickle.load(f)
        pred = net.classify(x)
        print("Predicted: ", pred)
        print("Actual: ", i, "\n")