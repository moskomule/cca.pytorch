# CCA.pytorch

PyTorch implementation of 
* **﻿SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability** 
* **﻿Insights on representational similarity in neural networks with canonical correlation**

# Requirements

* Python>=3.6
* PyTorch>=0.4.1
* torchvision>=0.2.1
* homura (to run `example.py`)

# Usage

```python
from cca import CCAHook
hook1 = CCAHook(model, "layer3.0.conv1")
hook2 = CCAHook(model, "layer3.0.conv2")
model.eval()
model(torch.randn(1200, 3, 224, 224))
hook1.distance(hook2, size=8) # resize to 8x8
```

# Note

While the original SVCCA uses DFT for resizing, we use global average pooling for simplicity.