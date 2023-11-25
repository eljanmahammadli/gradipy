import os, requests
from gradipy.tensor import Tensor


def fetch(url: str, save_directory="./weights"):
    """downloads pytorch triained weights"""
    from torch import load

    os.makedirs(save_directory, exist_ok=True)
    filename = url.split("/")[-1]
    save_path = os.path.join(save_directory, filename)
    if not os.path.exists(save_path):
        response = requests.get(url)
        with open(save_path, "wb") as file:
            file.write(response.content)
    return load(save_path)


def load_and_preprocess_image(url) -> Tensor:
    """preprocess imaganet example"""
    from torchvision import transforms
    from PIL import Image
    from io import BytesIO

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
