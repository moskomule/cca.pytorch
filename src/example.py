from pathlib import Path

import torch
from homura.utils import Trainer, callbacks
from homura.vision.models.cifar.resnet import resnet20
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cca import CCAHook


def get_loader(batch_size, root="~/.torch/data/cifar10"):
    root = Path(root).expanduser()
    if not root.exists():
        root.mkdir(parents=True)
    root = str(root)

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    data_augmentation = [transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip()]

    train_loader = DataLoader(
        datasets.CIFAR10(root, train=True, download=True,
                         transform=transforms.Compose(data_augmentation + to_normalized_tensor)),
        batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(
        datasets.CIFAR10(root, train=False, transform=transforms.Compose(to_normalized_tensor)),
        batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader


def main(batch_size):
    train_loader, test_loader = get_loader(128)
    model1 = resnet20(num_classes=10)
    model2 = resnet20(num_classes=10)
    optimizer1 = torch.optim.SGD(params=model1.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 50)
    trainer1 = Trainer(model1, optimizer1, F.cross_entropy, scheduler=scheduler1, callbacks=callbacks.Callback(),
                       verb=False)
    optimizer2 = torch.optim.SGD(params=model2.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 50)
    trainer2 = Trainer(model2, optimizer2, F.cross_entropy, scheduler=scheduler2, callbacks=callbacks.Callback(),
                       verb=False)
    hook = CCAHook([model1, model2], [["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "avgpool"],
                                      ["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "avgpool"]],
                   train_loader.dataset, batch_size=batch_size)
    for ep in range(200):
        print(f"{ep:>4}---")
        print(hook.distance())
        trainer1.train(train_loader)
        trainer2.train(train_loader)


if __name__ == '__main__':
    main(2048)
