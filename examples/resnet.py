import argparse, timeit
import numpy as np
from gradipy.tensor import Tensor
from models.resnet import ResNet, Bottleneck
from extra.helpers import load_and_preprocess_image
from extra.imagenet import IMAGENET_CATEGORIES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter the URL of an image to classify.")
    parser.add_argument("url", nargs="?", help="URL of image to classify.")
    parser.add_argument("--test", action="store_true", help="Use a predefined URL for testing.")
    args = parser.parse_args()
    if args.test:
        test_url = "https://us.feliway.com/cdn/shop/articles/7_Reasons_Why_Humans_Cats_Are_A_Match_Made_In_Heaven-9.webp?v=1667409797"
        args.url = test_url
    elif not args.url:
        parser.error("Please provide the URL of an image to classify.")

    img = Tensor(load_and_preprocess_image(args.url))
    st = timeit.default_timer()
    resnet50 = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])
    resnet50.from_pretrained()
    logits = resnet50(img)
    idx = np.argmax(logits.data, axis=1)[0]
    value = np.max(logits.data, axis=1)[0]
    cls = IMAGENET_CATEGORIES[idx]
    et = timeit.default_timer()
    print(f"Predicted in {et-st:.3f} seconds. Idx: {idx}, Logit: {value:.3f}, Category: {cls}")

    if args.test:
        expected_idx = 281
        expected_value = 10.254
        expected_cls = "tabby"

        assert idx == expected_idx, f"Expected index: {expected_idx}, Actual index: {idx}"
        assert np.isclose(
            value, expected_value, atol=1e-3
        ), f"Expected value: {expected_value}, Actual value: {value}"
        assert cls == expected_cls, f"Expected category: {expected_cls}, Actual category: {cls}"
        print("Test passed.")
