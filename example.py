from copy import deepcopy

import torch
from homura import trainers, callbacks
from homura.vision.data.loaders import cifar10_loaders
from homura.vision.models.classification import resnet20
from torch.nn import functional as F
from tqdm import trange

from cca import CCAHook


def plot(history, layers):
    import matplotlib.pyplot as plt

    x = [0, 30, 50, 100]
    for k, v in zip(layers, torch.Tensor(history).t().tolist()):
        plt.plot(x, v, label=k)
    plt.legend()
    plt.savefig("save.png")


def main(batch_size):
    layers = ["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "fc"]
    train_loader, test_loader = cifar10_loaders(128)
    weight_save = callbacks.WeightSave("checkpoints")
    model = resnet20(num_classes=10)
    model2 = deepcopy(model)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)
    trainer = trainers.SupervisedTrainer(model, optimizer, F.cross_entropy, scheduler=scheduler,
                                         callbacks=weight_save, verb=False)
    for ep in trange(100, ncols=80):
        trainer.train(train_loader)

    hooks1 = [CCAHook(model, name, svd_cpu=args.gpuonly) for name in layers]
    hooks2 = [CCAHook(model2, name, svd_cpu=args.gpuonly) for name in layers]
    device = next(model.parameters()).device
    model2.to(device)
    input = hooks1[0].data(train_loader.dataset, batch_size=batch_size).to(device)
    history = []

    def distance():
        model.eval()
        model2.eval()
        with torch.no_grad():
            model(input)
            model2(input)
        return [h1.distance(h2) for h1, h2 in zip(hooks1, hooks2)]

    # 0 and 99
    history.append(distance())

    # 29 and 99, ...
    for ep in (29, 49, 99):
        saved = torch.load(weight_save.save_path / f"{ep}.pkl")
        model2.load_state_dict(saved["model"])
        history.append(distance())
    plot(history, layers)


if __name__ == '__main__':
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("--gpuonly", action="store_true")
    args = p.parse_args()
    main(6400)
