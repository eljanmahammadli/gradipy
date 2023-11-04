class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.data = None

    def __call__(self, logits, target):
        self.probs = logits.softmax(target)
        logprobs = self.probs.log()
        n = logprobs.shape[0]
        self.data = -logprobs.data[range(n), target.data].mean()
        return self

    def __repr__(self):
        return f"CrossEntropyLoss({self.data})"

    def backward(self):
        return self.probs.backward()
