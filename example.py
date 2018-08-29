from copy import deepcopy
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
    layers = ["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "fc"]
    train_loader, test_loader = get_loader(128)
    weight_save = callbacks.WeightSave("checkpoints")
    model = resnet20(num_classes=10)
    model2 = deepcopy(model)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)
    trainer = Trainer(model, optimizer, F.cross_entropy, scheduler=scheduler,
                      callbacks=weight_save, verb=False)
    for ep in range(100):
        trainer.train(train_loader)
    hooks1 = [CCAHook(model, name) for name in layers]
    hooks2 = [CCAHook(model2, name) for name in layers]
    input = hooks1[0].data(train_loader.dataset, batch_size=batch_size)
    history = []

    def distance():
        model.eval()
        model2.eval()
        model(input)
        model2(input)
        return [h1.distance(h2) for h1, h2 in zip(hooks1, hooks2)]

    history.append(distance())
    for ep in (29, 49, 99):
        saved = torch.load(weight_save.save_path / f"{ep}.pkl")
        model2.load_state_dict(saved["model"])
        distance()

    import matplotlib as mpl

    mpl.use('Agg')
    import matplotlib.pyplot as plt

    for k, v in zip(layers, torch.Tensor(history).t().tolist()):
        plt.plot(v, label=k)
    plt.xticks(["0", "30", "50", "100"])
    plt.legend()
    plt.savefig("save.png")


if __name__ == '__main__':
    main(6400)
