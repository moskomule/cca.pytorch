from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


def svd_reduction(tensor: torch.Tensor, accept_rate=0.99):
    left, diag, right = torch.svd(tensor)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(ratio < accept_rate,
                      torch.ones(1).to(ratio.device),
                      torch.zeros(1).to(ratio.device)
                      ).sum()
    return tensor @ right[:, :int(num)]


def zero_mean(tensor: torch.Tensor, dim):
    return tensor - tensor.mean(dim=dim, keepdim=True)


def _svd_cca(x, y):
    u_1, s_1, v_1 = x.svd()
    u_2, s_2, v_2 = y.svd()
    try:
        uu = u_1.t() @ u_2
        u, diag, v = (uu).svd()
    except RuntimeError as e:
        print(u_1.shape)
        print(u_2.shape)
        raise e
    a = v_1 @ s_1.reciprocal().diag() @ u
    b = v_2 @ s_2.reciprocal().diag() @ v
    return a, b, diag


def cca(x, y, method):
    """
    Canonical Correlation Analysis,
    cf. Press 2011 "Cannonical Correlation Clarified by Singular Value Decomposition"
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd"  or "qr"
    :return: cca vectors for input x, cca vectors for input y, canonical correlations
    """
    assert x.size(0) == y.size(0), f"Number of data needs to be same but {x.size(0)} and {y.size(0)}"
    assert x.size(0) >= x.size(1) and y.size(0) >= y.size(1), f"data[0] should be larger than data[1]"
    assert method in ("svd", "qr"), "Unknown method"

    x = zero_mean(x, dim=0)
    y = zero_mean(y, dim=0)
    return _svd_cca(x, y)


def svcca_distance(x, y, method="svd"):
    """
    SVCCA distance proposed in Raghu et al. 2017
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd" (default) or "qr"
    """
    x = svd_reduction(x)
    y = svd_reduction(y)
    div = min(x.size(1), y.size(1))
    a, b, diag = cca(x, y, method=method)
    return 1 - diag.sum() / div


def pwcca_distance(x, y, method="svd"):
    """
    Project Weighting CCA proposed in Marcos et al. 2018
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd" (default) or "qr"
    """
    a, b, diag = cca(x, y, method=method)
    alpha = (x @ a).abs().sum(dim=0)
    alpha = alpha / alpha.sum()
    return 1 - alpha @ diag


class CCAHook(object):
    cca = {"pwcca": pwcca_distance,
           "svcca": svcca_distance}

    def __init__(self, models: Iterable[nn.Module], names: Iterable[Iterable[str] or str],
                 dataset: Dataset, batch_size=1_024):
        """
        CCA distance between give two models
        >>>hook = CCAHook([model1, model2], ["layer3.0.conv1", "layer1.0.conv2"], train_loader.dataset)
        >>>hook.distance("pwcca")
        # [('layer3.0.conv1', 'layer3.0.conv2', 0.5082974433898926)]
        :param models: a pair of models
        :param names: names of layers to be compared
        :param dataset: dataset
        :param batch_size: batch size to be used to calculate CCA. Need to be large enough.
        """

        self.models = models
        self.batch_size = batch_size
        data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        self.fixed_input, _ = next(iter(data_loader))
        self.names = []
        for nms in names:
            if isinstance(nms, str):
                self.names.append([nms])
            else:
                self.names.append(list(nms))
        self.register_hooks()

    def register_hooks(self):
        for model, nms in zip(self.models, self.names):
            for n, m in model.named_modules():
                if n in nms:
                    m.register_forward_hook(self._hook)

    def collect(self):
        outputs = []
        for model, nms in zip(self.models, self.names):
            output = {}
            input = self.fixed_input.to(self._device(model))
            model(input)
            for n, m in model.named_modules():
                if n in nms:
                    _output = getattr(m, "_cca_hook", None)
                    output[n] = _output
            outputs.append(output)
        return outputs

    def resize(self, tensor):
        # Conv2d
        if tensor.ndimension() == 4:
            b, c, h, w = tensor.shape
            if b <= c * h * w:
                for i in range(h, 1, -2):
                    if b > c * i * i:
                        break
                if b <= c * i * i:
                    raise RuntimeError("Batch size is too small")
                tensor = F.adaptive_avg_pool2d(tensor, i)
        return tensor.view(self.batch_size, -1)

    def distance(self, method="pwcca"):
        assert method in ("pwcca", "svcca")
        assert len(self.models) == 2 and len(self.names) == 2
        _model1, _model2 = self.collect()
        outputs = []
        for _name1, _name2 in zip(*self.names):
            _param1 = self.resize(_model1[_name1])
            _param2 = self.resize(_model2[_name2])
            outputs.append((_name1, _name2, self.cca[method](_param1, _param2).item()))
        return outputs

    @staticmethod
    def _hook(module, input, output):
        if not hasattr(module, "_cca_hook"):
            setattr(module, "_cca_hook", output)

    @staticmethod
    def _device(model):
        return next(model.parameters()).device


if __name__ == '__main__':
    from homura.vision.models.cifar import resnet20
    from example import get_loader

    train_loader, test_loader = get_loader(64, )
    model1 = resnet20(num_classes=10)
    model2 = resnet20(num_classes=10)
    hook = CCAHook([model1, model2], [["layer3.0.conv1", "layer1.0.conv2"], ["layer3.0.conv2", "layer1.0.conv1"]],
                   train_loader.dataset, batch_size=1024)
    print(hook.distance())
