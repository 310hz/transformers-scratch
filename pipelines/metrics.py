from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class BaseMetric(ABC):
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, predicts, targets):
        pass

    @abstractmethod
    def compute(self, reset=True):
        pass


class NamedMetric:
    def __init__(self, metrics: dict[str, BaseMetric]):
        self.metrics = metrics

    def update(self, predicts, targets):
        for metric in self.metrics.values():
            metric.update(predicts, targets)

    def compute(self, reset=True):
        results = {
            name: metric.compute(reset)
            for name, metric in self.metrics.items()
        }
        return results


class BaseMeanMetric(BaseMetric):
    def reset(self):
        self.sum = 0
        self.n_total = 0

    def compute(self, reset=True):
        result = self.sum / self.n_total
        if reset:
            self.reset()
        return result


class Accuracy(BaseMeanMetric):
    def update(self, predicts, targets):
        self.sum += (predicts == targets).to(torch.float32).sum()
        self.n_total += predicts.numel()


class MeanSquareError(BaseMeanMetric):
    def update(self, predicts, targets):
        self.sum += F.mse_loss(predicts, targets, reduction="sum")
        self.n_total += predicts.numel()


class CrossEntropy(BaseMeanMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def update(self, predicts, targets):
        self.sum += F.cross_entropy(
            predicts,
            targets,
            reduction="sum",
            **self.kwargs
        )
        ignore_index = self.kwargs.get("ignore_index", -100)
        self.n_total += (targets != ignore_index).sum().item()


class Perplexity(CrossEntropy):
    def compute(self, reset=True):
        return torch.exp(super().compute(reset))
