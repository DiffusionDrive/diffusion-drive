import torch.nn as nn


class StateNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def normalize(self, x):
        return x / 20.0

    def inverse(self, x):
        return x * 20.0


class ObservationNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def normalize(self, x):
        return x / 20.0

    def inverse(self, x):
        return x * 20.0
