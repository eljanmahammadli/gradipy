import gradipy.nn as nn
from gradipy.tensor import Tensor
from extra.helpers import fetch


class AlexNet(nn.Module):
    """pure gradipy implementation of AlexNet."""

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(6)
        self.classifier = nn.Sequential(
            # nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten()
        x = self.classifier(x)
        return x

    def from_pretrained(self) -> None:
        index, weights = 0, []
        url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
        for key, value in fetch(url).items():
            # print(f"Key: {key}, Tensor Shape: {value.shape}")
            if "bias" in key and "features" in key:
                weights.append(Tensor(value.detach().numpy().reshape(-1, 1)))
            elif "weight" in key and "classifier" in key:
                weights.append(Tensor(value.detach().numpy().transpose(1, 0)))
            else:
                weights.append(Tensor(value.detach().numpy()))

        trainable_layers = [
            l for l in self.features.layers + self.classifier.layers if l.name in ["Conv2d", "Linear"]
        ]
        for layer in trainable_layers:
            layer.weight, layer.bias = weights[index], weights[index + 1]
            index += 2  # weight + bias
