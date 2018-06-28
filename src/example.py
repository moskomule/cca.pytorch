from copy import deepcopy
from pathlib import Path

import torch
from homura.utils import Trainer, callbacks
from homura.vision.models.cifar.resnet import ResNet as OriginalResNet, BasicBlock
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cca import pwcca_distance, svcca_distance


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


class ResNet(OriginalResNet):
    def block_output(self, input, block_id):
        assert block_id in (1, 2, 3)
        self.eval()
        with torch.no_grad():
            x = self.conv1(input)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            if block_id == 1:
                return x
            x = self.layer2(x)
            if block_id == 2:
                return x
            x = self.layer3(x)
            if block_id == 3:
                return x
            raise RuntimeError("No output!")


def main(batch_size):
    train_loader, test_loader = get_loader(128)
    fixed_input, _ = next(iter(torch.utils.data.DataLoader(
        datasets.CIFAR10(Path("~/.torch/data/cifar10").expanduser(), train=True,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.2023, 0.1994, 0.2010))])),
        batch_size=batch_size, shuffle=False, num_workers=4)))
    model = ResNet(BasicBlock, 3, num_classes=10)
    model2 = deepcopy(model)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)
    trainer = Trainer(model, optimizer, F.cross_entropy, scheduler=scheduler, callbacks=callbacks.Callback(),
                      verb=False)
    fixed_input = fixed_input.to(trainer._device)
    model2.to(trainer._device)
    for ep in range(200):
        print(f"{ep:>4}---")
        output1 = model.block_output(fixed_input, 1)
        output2 = model2.block_output(fixed_input, 1)
        sv = svcca_distance(output1.view(batch_size, -1),
                            output2.view(batch_size, -1))
        print(f">>SVCCA: {sv.item():.4f}")
        pw = pwcca_distance(output1.view(batch_size, -1),
                            output2.view(batch_size, -1))
        print(f">>PWCCA: {pw.item():.4f}")
        trainer.train(train_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    main(1024)
