import timeit
import numpy as np
from gradipy.tensor import Tensor
import gradipy.nn as nn
from gradipy import datasets
from gradipy.nn import optim


# hyperparameters
input_dim = 28 * 28
output_dim = 10
hidden_dim = 128
lr = 0.001
epochs = 1000
batch_size = 128

# load the dataset and split into train, validation and test set
X_train, y_train, X_val, y_val, X_test, y_test = datasets.MNIST()


def get_batch():
    ix = np.random.randint(X_train.shape[0], size=batch_size)
    x = Tensor(X_train[ix])
    y = y_train[ix]
    return x, y


# estimate the loss and the accuracy on the train and validation set
def evaluate(split):
    X, y = {"train": (X_train, y_train), "val": (X_val, y_val)}[split]
    logits = model.forward(Tensor(X))
    logprobs = logits.softmax().log()
    loss = -logprobs.data[range(X.shape[0]), y].mean()
    accuracy = (np.argmax(logprobs.data, axis=1) == y).astype(float).mean()
    return split, loss, accuracy


# define the model
class DenseNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = nn.init_kaiming_normal(input_dim, hidden_dim, nonlinearity="relu")
        self.W2 = nn.init_kaiming_normal(hidden_dim, output_dim, nonlinearity="relu")

    def forward(self, X):
        logits = X.matmul(self.W1).relu().matmul(self.W2)
        return logits


# initialize the model, loss function and optimizer
model = DenseNeuralNetwork(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([model.W1, model.W2], lr=lr)


st = timeit.default_timer()
# training loop
for epoch in range(epochs + 1):
    optimizer.zero_grad()
    xb, yb = get_batch()
    logits = model.forward(xb)
    loss = criterion(logits, yb)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        _, _, train_accuracy = evaluate("train")
        _, _, val_accuracy = evaluate("val")
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

et = timeit.default_timer()
print(f"Trained in {et-st:.3f} seconds")
print(evaluate("train"))
print(evaluate("val"))
