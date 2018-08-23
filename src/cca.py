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
    uu = u_1.t() @ u_2
    try:
        u, diag, v = (uu).svd()
    except RuntimeError as e:
        y = uu.abs()
        print(f"u_1^Tu_2: min/mean/max {y.min().item(), y.mean().item(), y.max().item()}")
        raise e
    a = v_1 @ s_1.reciprocal().diag() @ u
    b = v_2 @ s_2.reciprocal().diag() @ v
    return a, b, diag


def _cca(x, y, method):
    """
    Canonical Correlation Analysis,
    cf. Press 2011 "Cannonical Correlation Clarified by Singular Value Decomposition"
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd"  or "qr"
    :return: _cca vectors for input x, _cca vectors for input y, canonical correlations
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
    a, b, diag = _cca(x, y, method=method)
    return 1 - diag.sum() / div


def pwcca_distance(x, y, method="svd"):
    """
    Project Weighting CCA proposed in Marcos et al. 2018
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd" (default) or "qr"
    """
    a, b, diag = _cca(x, y, method=method)
    alpha = (x @ a).abs().sum(dim=0)
    alpha = alpha / alpha.sum()
    return 1 - alpha @ diag


class CCAHook(object):
    _available_modules = {nn.Conv2d, nn.Linear}

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

        assert len(models) == 2 and len(names) == 2

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
        self._history = []

    def register_hooks(self):
        for model, nms in zip(self.models, self.names):
            for n, m in model.named_modules():
                if n in nms:
                    if type(m) not in self._available_modules:
                        raise RuntimeError(f"cannot resgister a hook for a module {type(m)}")
                    m.register_forward_hook(self._hook)

    def _cca(self, x, y, method):
        f = svcca_distance if method == "svcca" else pwcca_distance
        try:
            distance = f(x, y)
        except Exception as e:
            print(e)
            distance = None
        return distance

    def collect(self):
        outputs = []
        for model, nms in zip(self.models, self.names):
            output = {}
            input = self.fixed_input.to(self._device(model))
            with torch.no_grad():
                model.eval()
                model(input)
            for n, m in model.named_modules():
                if n in nms:
                    _output = getattr(m, "_cca_hook", None)
                    output[n] = _output
            outputs.append(output)
        return outputs

    @staticmethod
    def _conv2d_reshape(tensor, factor):
        b, c, h, w = tensor.shape
        tensor = F.adaptive_avg_pool2d(tensor, (h // factor, w // factor))
        tensor = tensor.reshape(b, c, -1).transpose(0, -1).transpose(-1, -2)
        return tensor

    def _conv2d(self, tensor1, tensor2, method, factor=2):
        assert tensor1.shape == tensor2.shape
        tensor1 = self._conv2d_reshape(tensor1, factor)
        tensor2 = self._conv2d_reshape(tensor2, factor)
        d = [None] * tensor1.size(0)
        for i, (t1, t2) in enumerate(zip(tensor1, tensor2)):
            d[i] = self._cca(t1, t2, method).item()

        return torch.Tensor([i for i in d if i is not None]).mean()

    def distance(self, method="pwcca"):
        assert method in ("pwcca", "svcca")
        model1, model2 = self.collect()
        outputs = []
        for name1, name2 in zip(*self.names):
            param1 = model1[name1]
            param2 = model2[name2]
            param1_dim = param1.ndimension()
            param2_dim = param2.ndimension()
            if param1_dim == 2 and param2_dim == 2:
                distance = self._cca(param1, param2, method).item()
            elif param1_dim == 4 and param2_dim == 4:
                distance = self._conv2d(param1, param2, method).item()
            else:
                raise RuntimeError("Tensor shape mismatch!")

            outputs.append((name1, name2, distance))
        self._history.append(outputs)
        return outputs

    @property
    def history(self):
        assert len(self._history) != 0
        values = [[v if v is not None else 0 for n_1, n_2, v in val]
                  for val in self._history]
        values = torch.Tensor(values).t()
        return {f"{n_1}-{n_2}": v.tolist()
                for (n_1, n_2), v in zip(zip(*self.names), values)}

    @staticmethod
    def _hook(module, input, output):
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
