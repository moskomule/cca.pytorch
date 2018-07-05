from typing import Iterable

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from cca import pwcca_distance, svcca_distance


class CCAHook(object):
    cca = {"pwcca": pwcca_distance,
           "svcca": svcca_distance}

    def __init__(self, models: Iterable[nn.Module], names: Iterable[Iterable[str] or str],
                 dataset: Dataset, batch_size=1_024):
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
