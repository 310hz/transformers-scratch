import torch.nn.functional as F
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
from datasets import load_dataset

from ..base import Pipeline
from ..metrics import NamedMetric, CrossEntropy, Accuracy


class ImageDataset(IterableDataset):
    def __init__(self, ds, colname_image="image", colname_target="label"):
        self.ds = ds
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
        ])
        self.colname_image = colname_image
        self.colname_target = colname_target

    def __iter__(self):
        for sample in self.ds:
            imgs = sample[self.colname_image]
            targets = sample[self.colname_target]
            imgs = self.transform(imgs)
            yield imgs, targets


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
            "accuracy@1": Accuracy(k=1),
            "accuracy@5": Accuracy(k=5),
        })

    def forward(self, batch):
        imgs, targets = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        predicts = self.model(imgs)
        return predicts, targets

    def loss_fn(self, predicts, targets):
        loss = F.cross_entropy(predicts, targets)
        return loss


class ImageClassificationNanoPipeline(ImageClassificationPipeline):
    def get_dataset(self):
        ds_train = load_dataset(
            "uoft-cs/cifar10",
            split="train",
            streaming=True,
        )
        ds_test = load_dataset(
            "uoft-cs/cifar10",
            split="test[:1000]",
        )
        get_ds_func = lambda ds: ImageDataset(
            ds, colname_image="img", colname_target="label"
        )
        return ds_train, ds_test, get_ds_func
