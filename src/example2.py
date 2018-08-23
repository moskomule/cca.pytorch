from copy import deepcopy

import torch
from homura.utils import Trainer, callbacks
from homura.vision.models.cifar.resnet import resnet20
from torch.nn import functional as F

from cca import CCAHook
from example import get_loader


def main(batch_size):
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
    hook = CCAHook([model, model2], [["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "fc"],
                                     ["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "fc"]],
                   train_loader.dataset, batch_size=batch_size)
    hook.distance() # before training
    for ep in (29, 49, 99):
        saved = torch.load(weight_save.save_path / f"{ep}.pkl")
        model2.load_state_dict(saved["model"])
        hook.distance()

    import matplotlib as mpl

    mpl.use('Agg')
    import matplotlib.pyplot as plt

    for k, v in hook.history.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.savefig("save.png")


if __name__ == '__main__':
    main(6400)
