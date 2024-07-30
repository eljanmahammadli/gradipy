<div align="center">
  <img src="https://github.com/eljanmahammadli/gradipy/blob/main/logo.png" alt="Logo">
</div>


# gradipy: A Lightweight Neural Network Library
![Tests](https://github.com/eljanmahammadli/gradipy/actions/workflows/ci.yml/badge.svg)

gradipy is an evolving project, and it provides PyTorch-like API for building and training neural networks, there are some features that are actively developed on and plan to support in future releases. My inspiration for this project stems from [micrograd](https://github.com/karpathy/micrograd/tree/master) and [tinygrad](https://github.com/tinygrad/tinygrad). While the former focuses solely on scalar values, which makes it quite simple, the latter is a more advanced implementation. The library I am building, tentatively named gradipy, falls somewhere in between these two.

While gradipy may not be your top-tier production tool (we'll leave that to the heavyweights), it's more like a trusty sidekick on your learning expedition. Because, let's be honest, understanding the nitty-gritty of neural networks by rolling up your sleeves and implementing from scratch? That's the real superhero origin story. Serious learning, a bit of coding flair, and a sidekick named gradipy—what more could you want?

Additionally, any contributions to gradipy are welcomed. Given the evolving nature of the project, there might be seminar bugs or opportunities to refine design choices. Your input is appreciated in making gradipy even better.

## Sample Usage

In this example, we define a simple feedforward neural network to train classifier on the MNIST data. (please refer to the [example usage](https://github.com/eljanmahammadli/gradipy/blob/main/examples/mnist.py)): 

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
In order to run example architectures, refer to the `models/` and run the following command:
```bash
python -m examples.resnet "link_to_the_image"
```

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
- Backward pass for Conv2d and BatchNorm2d
- Backward passes for: `mul` (problem with broadcasting), `tanh`
- More Loss functions (`nn.MSELoss` and `nn.NLLLoss`)
- Recurrent layers for sequence data
- GPU acceleration (no idea how to do that, openCL?)
- Implement more model architectures such as Transformers
    - This requires more components and layers (new activations, layers such as Embedding and so on)
