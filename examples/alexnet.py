import argparse, timeit
import numpy as np
from gradipy.tensor import Tensor
from models.alexnet import AlexNet
from extra.imagenet import IMAGENET_CATEGORIES
from extra.helpers import load_and_preprocess_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter the URL of an image to classify.")
    parser.add_argument("url", nargs="?", help="URL of image to classify.")
    parser.add_argument("--test", action="store_true", help="Use a predefined URL for testing.")
    args = parser.parse_args()
    if args.test:
        test_url = "https://images.theconversation.com/files/86272/original/image-20150624-31498-1med6rz.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=926&fit=clip"
        args.url = test_url
    elif not args.url:
        parser.error("Please provide the URL of an image to classify.")

    img = Tensor(load_and_preprocess_image(args.url))
    st = timeit.default_timer()
    resnet50 = AlexNet()
    resnet50.from_pretrained()
    logits = resnet50(img)
    idx = np.argmax(logits.data, axis=1)[0]
    value = np.max(logits.data, axis=1)[0]
    cls = IMAGENET_CATEGORIES[idx]
    et = timeit.default_timer()
    print(f"Predicted in {et-st:.3f} seconds. Idx: {idx}, Logit: {value:.3f}, Category: {cls}")

    import matplotlib.pyplot as plt

    plt.plot(logits.transpose().data)
    plt.show()

    if args.test:
        expected_idx = 36
        expected_value = 24.387
        expected_cls = "terrapin"

        assert idx == expected_idx, f"Expected index: {expected_idx}, Actual index: {idx}"
        assert np.isclose(
            value, expected_value, atol=1e-3
        ), f"Expected value: {expected_value}, Actual value: {value}"
        assert cls == expected_cls, f"Expected category: {expected_cls}, Actual category: {cls}"
        print("Test passed.")
