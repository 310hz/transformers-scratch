import torch.nn as nn


def conv1x1(c_in, c_out, stride):
    return nn.Conv2d(
        c_in,
        c_out,
        kernel_size=1,
        padding=0,
        stride=stride,
        bias=False
    )

def conv3x3(c_in, c_out, stride):
    return nn.Conv2d(
        c_in,
        c_out,
        kernel_size=3,
        padding=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            conv3x3(c_in, c_out, stride),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            conv3x3(c_out, c_out, stride=1),
            nn.BatchNorm2d(c_out),
        )
        if c_in != c_out or downsample:
            self.skip = nn.Sequential(
                conv1x1(c_in, c_out, stride),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.skip = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class BottleneckBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            conv1x1(c_in, c_mid, stride),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
            conv3x3(c_mid, c_mid, stride=1),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(),
            conv1x1(c_mid, c_out, stride=1),
            nn.BatchNorm2d(c_out),
        )
        if c_in != c_out or downsample:
            self.skip = nn.Sequential(
                conv1x1(c_in, c_out, stride),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.skip = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class BasicBlocks(nn.Module):
    def __init__(self, n_layers, c_in, c_out, downsample=True):
        super().__init__()
        blocks = []
        blocks.append(BasicBlock(c_in, c_out, downsample=downsample))
        for _ in range(1, n_layers):
            blocks.append(BasicBlock(c_out, c_out, downsample=False))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class BottleneckBlocks(nn.Module):
    def __init__(self, n_layers, c_in, c_mid, c_out, downsample=True):
        super().__init__()
        blocks = []
        blocks.append(BottleneckBlock(c_in, c_mid, c_out, downsample=downsample))
        for _ in range(1, n_layers):
            blocks.append(BottleneckBlock(c_out, c_mid, c_out, downsample=False))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class ResNet34(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlocks(3, c_in=64, c_out=64, downsample=False),
            BasicBlocks(4, c_in=64, c_out=128),
            BasicBlocks(6, c_in=128, c_out=256),
            BasicBlocks(3, c_in=256, c_out=512),
            nn.AvgPool2d(kernel_size=7),
            nn.Flatten(),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ResNet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BottleneckBlocks(3, c_in=64, c_mid=64, c_out=256, downsample=False),
            BottleneckBlocks(4, c_in=256, c_mid=128, c_out=512),
            BottleneckBlocks(6, c_in=512, c_mid=256, c_out=1024),
            BottleneckBlocks(3, c_in=1024, c_mid=512, c_out=2048),
            nn.AvgPool2d(kernel_size=7),
            nn.Flatten(),
            nn.Linear(2048, n_classes),
        )

    def forward(self, x):
        return self.net(x)
