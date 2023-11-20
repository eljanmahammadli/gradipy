import os, requests, argparse
from typing import Sequence
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torchvision import models, transforms
from gradipy.tensor import Tensor
import gradipy.nn as nn
from .imagenet import IMAGENET_CATEGORIES


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
            # nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten()
        x = self.classifier(x)
        return x

    def load_weights(self, weights: Sequence[Tensor]) -> None:
        index = 0
        trainable_layers = [
            l for l in self.features.layers + self.classifier.layers if l.name in ["Conv2d", "Linear"]
        ]
        for layer in trainable_layers:
            layer.weight, layer.bias = weights[index], weights[index + 1]
            index += 2  # weight + bias


def load_weights_from_pytorch(save_directory="./weights"):
    # download the file if it doesn't exist
    url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
    os.makedirs(save_directory, exist_ok=True)
    filename = url.split("/")[-1]
    save_path = os.path.join(save_directory, filename)
    if not os.path.exists(save_path):
        response = requests.get(url)
        with open(save_path, "wb") as file:
            file.write(response.content)
    # parse it into a list of Tensors using torch
    weights = []
    state_dict = torch.load(save_path)
    for key, value in state_dict.items():
        # print(f"Key: {key}, Tensor Shape: {value.shape}")
        if "bias" in key and "features" in key:
            weights.append(Tensor(value.detach().numpy().reshape(-1, 1)))
        elif "weight" in key and "classifier" in key:
            weights.append(Tensor(value.detach().numpy().transpose(1, 0)))
        else:
            weights.append(Tensor(value.detach().numpy()))
    return weights


def load_and_preprocess_image(url):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def pytorch_alexnet(img):
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        logits = alexnet(img)
    idx = torch.argmax(logits).item()
    value = torch.max(logits).item()
    cls = IMAGENET_CATEGORIES[idx]
    return logits, idx, value, cls


def gradipy_alexnet(img):
    alexnet = AlexNet()
    weights = load_weights_from_pytorch()
    alexnet.load_weights(weights)
    logits = alexnet(Tensor(img.numpy()))
    idx = np.argmax(logits.data, axis=1)[0]
    value = np.max(logits.data, axis=1)[0]
    cls = IMAGENET_CATEGORIES[idx]
    return logits, idx, value, cls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter the URL of an image to classify.")
    parser.add_argument("url", help="URL of image to classify.")
    args = parser.parse_args()
    img = load_and_preprocess_image(args.url)
    logits, idx, value, cls = pytorch_alexnet(img)
    logits_, idx_, value_, cls_ = gradipy_alexnet(img)
    print(f"PyTorch: {idx=}, {value=}, {cls=}")
    print(f"GradiPy: {idx_=}, {value_=}, {cls_=}")
    np.testing.assert_allclose(logits.data, logits_.data, atol=1e-4)
    assert idx == idx_ and int(value) == int(value_) and cls == cls_
    load_weights_from_pytorch()
