import os
import urllib.request
import gzip
import numpy as np


def MNIST(path="./data/mnist_data"):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    if not os.path.exists(path):
        os.makedirs(path)

    for file in files:
        if not os.path.exists(os.path.join(path, file)):
            urllib.request.urlretrieve(base_url + file, os.path.join(path, file))

    def load_image(filename, offset):
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=offset)
        return data

    X = (
        load_image(f"{path}/train-images-idx3-ubyte.gz", 16)
        .reshape(-1, 28 * 28)
        .astype(np.float32)
    )
    y = load_image(f"{path}/train-labels-idx1-ubyte.gz", 8).astype(np.int32)
    X_test = (
        load_image(f"{path}/t10k-images-idx3-ubyte.gz", 16)
        .reshape(-1, 28 * 28)
        .astype(np.float32)
    )
    y_test = load_image(f"{path}/t10k-labels-idx1-ubyte.gz", 8).astype(np.int32)

    # 50,000 train, 10,000 dev and 10,000 test set
    split_idx = 50000
    np.random.seed(42)
    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)
    X, y = X[idxs], y[idxs]
    X_train, y_train, X_val, y_val = (
        X[:split_idx],
        y[:split_idx],
        X[split_idx:],
        y[split_idx:],
    )
    X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0  # normalize
    return X_train, y_train, X_val, y_val, X_test, y_test
