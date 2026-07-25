from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torch.distributed as dist


def reduce(*tensors: torch.Tensor) -> None:
    if (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() >= 2
    ):
        for tensor in tensors:
            dist.all_reduce(tensor)


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
        s = self.sum.detach().clone()
        n_total = torch.tensor(
            self.n_total,
            device=s.device,
            dtype=torch.int64,
        )
        reduce(s, n_total)
        result = s / n_total
        if reset:
            self.reset()
        return result


class Accuracy(BaseMeanMetric):
    @torch.no_grad()
    def update(self, predicts, targets):
        self.sum += (predicts == targets).to(torch.float32).sum()
        self.n_total += predicts.numel()


class MeanSquareError(BaseMeanMetric):
    @torch.no_grad()
    def update(self, predicts, targets):
        self.sum += F.mse_loss(predicts, targets, reduction="sum")
        self.n_total += predicts.numel()


class CrossEntropy(BaseMeanMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    @torch.no_grad()
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
