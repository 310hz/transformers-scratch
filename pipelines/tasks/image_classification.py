import torch.nn.functional as F
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
from datasets import load_dataset

from ..base import Pipeline
from ..metrics import NamedMetric, CrossEntropy, Accuracy


class ImageDataset(IterableDataset):
    def __init__(self, ds):
        self.ds = ds
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
        ])

    def __iter__(self):
        for sample in self.ds:
            img = sample["image"]
            label = sample["label"]
            img = self.transform(img)
            yield img, label


class ImageClassificationPipeline(Pipeline):
    def get_dataset(self):
        ds_train = load_dataset(
            "ILSVRC/imagenet-1k",
            split="train",
            streaming=True,
        )
        ds_test = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation[:10000]",
        )
        return ds_train, ds_test, ImageDataset

    def get_metrics(self):
        return NamedMetric({
            "cross_entropy": CrossEntropy(),
            "accuracy": Accuracy(),
        })

    def forward(self, batch):
        imgs, targets = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        predicts = self.model(imgs)
        predicts = predicts.view(-1, predicts.size(-1))
        targets = targets.view(-1)
        return predicts, targets

    def loss_fn(self, predicts, targets):
        loss = F.cross_entropy(predicts, targets)
        return loss


class ImageClassificationNanoPipeline(ImageClassificationPipeline):
    def get_dataset(self):
        ds_train = load_dataset(
            "ylecun/mnist",
            split="train",
            streaming=True,
        )
        ds_test = load_dataset(
            "ylecun/mnist",
            split="test[:10]",
        )
        return ds_train, ds_test, ImageDataset
