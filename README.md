
# gradipy: A Lightweight Neural Network Library

gradipy is an evolving project, and it will potentially provide PyTorch-like API for building and training neural networks, there are some features that are actively developed on and plan to support in future releases.

## Desired features to add:
- PyTorch like API for most important blocks for training NNs 
- Convolutional layers for image processing
- Recurrent layers for sequence data
- Potentially GPU acceleration

Please note that the library is currently in its early stages, and these features are expected in future updates.



## Desired Sample Usage

Here's a basic desired example of using gradipy to create and train a simple neural network for image classification:

```python
import gradipy as gp

class SimpleNeuralNet:
  def __init__(self):
    self.W1 = gp.random.uniform(256, 64)
    self.W2 = gp.random.uniform(64, 10)

  def forward(self, x):
    z = x @ self.W1
    a = z.relu()
    logits = a @ self.W2
    probs = logits.softmax()
    return probs

model = SimpleNeuralNet()
criterion = gp.nn.CrossEntropyLoss()
optimizer = gp.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
  output = model.forward(x)
  loss = criterion(output, y)
  optimizer.step.zero_grad()
  loss.backward()
  optimizer.step()
```

In this example, we define a simple feedforward neural network, compile it, and train it on random data. gradipy will provide building blocks like `Linear` layers, `activation` functions, `loss` functions, and `optimizers` for creating and training neural networks.

## Feature Roadmap

Here's a list of features we plan to implement in gradipy, along with their current status:

### To-Do

- [ ] Linear layers
- [ ] Activation functions
- [ ] Loss functions
- [ ] Basic optimizer support
- [ ] Convolutional layers for image processing
- [ ] Recurrent layers for sequence data
- [ ] GPU acceleration

### Done

- [x] Basic Tensor definition like wrapper around NumPy `ndarray`
- [x] Softmax function and its derivative
