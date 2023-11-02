class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.item = None

    def __call__(self, logits, target):
        self.probs = logits.softmax(target.data)
        logprobs = self.probs.log()
        n = logprobs.shape[0]
        self.item = -logprobs.data[range(n), target.data].mean()
        return self

    def __repr__(self):
        return f"CrossEntropyitem({self.item})"

    def backward(self):
        return self.probs.backward()
