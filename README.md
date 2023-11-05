
# gradipy: A Lightweight Neural Network Library
![Tests](https://github.com/eljanmahammadli/gradipy/actions/workflows/ci.yml/badge.svg)

gradipy is an evolving project, and it will potentially provide PyTorch-like API for building and training neural networks, there are some features that are actively developed on and plan to support in future releases.

## Desired features to add:
- PyTorch like API for most important blocks for training NNs 
- Convolutional layers for image processing
- Recurrent layers for sequence data
- Potentially GPU acceleration

Please note that the library is currently in its early stages, and these features are expected in future updates.



## Sample Usage

Here's a basic example of using gradipy to create and train a simple neural network for MNIST (please refer to the [example usage](https://github.com/eljanmahammadli/gradipy/blob/main/examples/mnist.py)): 

```python
import gradipy.nn as nn
from gradipy import datasets
from gradipy.nn import optim

X_train, y_train, X_val, y_val, X_test, y_test = datasets.MNIST()

# define some utility function here...

class DenseNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = nn.init_kaiming_normal(input_dim, hidden_dim, nonlinearity="relu")
        self.W2 = nn.init_kaiming_normal(hidden_dim, output_dim, nonlinearity="relu")

    def forward(self, X):
        logits = X.matmul(self.W1).relu().matmul(self.W2)
        return logits

model = DenseNeuralNetwork(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([model.W1, model.W2], lr=lr)

for epoch in range(epochs + 1):
    optimizer.zero_grad()
    xb, yb = get_batch()
    logits = model.forward(xb)
    loss = criterion(logits, yb)
    loss.backward()
    optimizer.step()

    # log the results on each epoch...
```

In this example, we define a simple feedforward neural network, compile it, and train it on random data. gradipy will provide building blocks like `Linear` layers, `activation` functions, `loss` functions, and `optimizers` for creating and training neural networks.

## Installation
You can install gradipy using `pip`. It is recommended to create a virtual environment before installing the library.
```bash
# Create and activate a virtual environment (optional but recommended)
# Replace 'myenv' with your desired environment name
python -m venv myenv
source myenv/bin/activate  # On Windows, use 'myenv\Scripts\activate'

# Install gradipy using pip
pip install gradipy
```

## Testing
To run the tests for gradipy, ensure you have pytest and PyTorch installed. If you haven't installed them yet, you can do so using pip:
```bash
pip install -r requirements.txt
```
After installing the necessary dependencies, you can run the tests using pytest. During testing, we compare the output results and gradients of various operations with their counterparts in PyTorch to ensure the correctness and compatibility of gradipy.
```bash
python -m pytest
```

## Feature Roadmap

Here's a list of features we plan to implement in gradipy, along with their current status:

### To-Do

- [ ] Backward passes for: `mul` (problem with broadcasting), `tanh`
- [ ] Add more operations and their gradients
- [ ] Batchnorm
- [ ] Convolutional layers for image processing
- [ ] PyTorch's `nn.Module`
- [ ] More Loss functions (`nn.MSELoss` and `nn.NLLLoss`)
- [ ] Recurrent layers for sequence data
- [ ] GPU acceleration (no idea how to do that)

### Done

- [x] Basic Tensor wrapper around NumPy `ndarray`
- [x] Forward and backward passes implemented for: `add`, `matmul`, `softmax`, `relu`, `sub`, `log`, `exp`, `log softmax`, `cross entropy` 
- [x] Autograd just like PyTorch's `backward` method using topological sort
- [x] nn.CrossEntropyLoss function 
- [x] Train MNIST with `gradipy`
- [x] Kaiming, Xavier init (normal + uniform)
- [x] Implemented Adam and added momentum to SGD
