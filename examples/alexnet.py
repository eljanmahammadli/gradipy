import argparse
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from gradipy.tensor import Tensor
from .imagenet import IMAGENET_CATEGORIES

alexnet_weights_path = "./notebooks/alexnet-owt-7be5be79.pth"
state_dict = torch.load(alexnet_weights_path)
weights = []
for key, value in state_dict.items():
    # print(f"Key: {key}, Tensor Shape: {value.shape}")
    weights.append(Tensor(value.detach().numpy()))


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
    # plt.imshow(img)
    # plt.show()
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


def gradipyt_alexnet(img):
    inp = Tensor(img.numpy())
    x = inp.conv2d(weights[0], stride=4, padding=2)
    x = x + weights[1].reshape(1, -1, 1, 1)
    x = x.relu()
    x = x.max_pool2d(kernel_size=3, stride=2)
    x = x.conv2d(weights[2], padding=2)
    x = x + weights[3].reshape(1, -1, 1, 1)
    x = x.relu()
    x = x.max_pool2d(kernel_size=3, stride=2)
    x = x.conv2d(weights[4], padding=1)
    x = x + weights[5].reshape(1, -1, 1, 1)
    x = x.relu()
    x = x.conv2d(weights[6], padding=1)
    x = x + weights[7].reshape(1, -1, 1, 1)
    x = x.relu()
    x = x.conv2d(weights[8], padding=1)
    x = x + weights[9].reshape(1, -1, 1, 1)
    x = x.relu()
    x = x.max_pool2d(kernel_size=3, stride=2)
    x = x.adaptive_avg_pool2d(6)

    x = x.reshape(1, -1)  # flatten
    # x = x.dropout(0.5)
    x = x.matmul(Tensor(weights[10].data.T))
    x = (x + weights[11]).relu()
    # x = x.dropout(0.5)
    x = x.matmul(Tensor(weights[12].data.T))
    x = (x + weights[13]).relu()
    x = x.matmul(Tensor(weights[14].data.T))
    logits = x + weights[15]
    idx = np.argmax(x.data, axis=1)[0]
    value = np.max(x.data, axis=1)[0]
    cls = IMAGENET_CATEGORIES[idx]
    return logits, idx, value, cls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter the URL of an image to classify.")
    parser.add_argument("url", help="URL of image to classify.")
    args = parser.parse_args()
    img = load_and_preprocess_image(args.url)
    logits, idx, value, cls = pytorch_alexnet(img)
    logits_, idx_, value_, cls_ = gradipyt_alexnet(img)
    print(f"PyTorch: {idx=}, {value=}, {cls=}")
    print(f"GradiPy: {idx_=}, {value_=}, {cls_=}")

    np.testing.assert_allclose(logits.data, logits_.data, atol=1e-4)
