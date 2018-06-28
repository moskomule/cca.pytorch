import torch


def svd_reduction(tensor: torch.Tensor, accept_rate=0.99):
    left, diag, right = torch.svd(tensor)
    max_size = len(diag)
    full = diag.abs().sum()
    for i in range(max_size - 1, 0, -1):
        if diag[:i].abs().sum() < accept_rate * full:
            break
    rank = i + 1
    return tensor @ right[:, :rank]


def zero_mean(tensor: torch.Tensor, dim):
    return tensor - tensor.mean(dim=dim, keepdim=True)


def _qr_cca(x, y):
    q_1, r_1 = x.qr()
    q_2, r_2 = y.qr()
    u, diag, v = (q_1.t() @ q_2).svd()
    a = r_1.inv() @ u
    b = r_2.inv() @ v
    return a, b, diag


def _svd_cca(x, y):
    u_1, s_1, v_1 = x.svd(some=True)
    u_2, s_2, v_2 = y.svd(some=True)
    u, diag, v = (u_1.t() @ u_2).svd()
    a = v_1 @ (1 / s_1).diag() @ u
    b = v_2 @ (1 / s_2).diag() @ v
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
    assert method in ("svd", "qr"), "Unknown method"

    x = zero_mean(x, dim=0)
    y = zero_mean(y, dim=0)
    return _svd_cca(x, y) if method == "svd" else _qr_cca(x, y)


def svcca_distance(x, y, method="svd"):
    """
    SVCCA distance proposed in Raghu et al. 2017
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd" (default) or "qr"
    """
    x = svd_reduction(x)
    y = svd_reduction(y)
    a, b, diag = cca(x, y, method=method)
    return 1 - diag.mean()


def pwcca_distance(x, y, method="svd"):
    """
    Project Weighting CCA proposed in Marcos et al. 2018
    :param x: data matrix [data, neurons]
    :param y: data matrix [data, neurons]
    :param method: computational method "svd" (default) or "qr"
    """
    a, b, diag = cca(x, y, method=method)
    pw = (x @ a).abs().sum(dim=0)
    pw /= pw.sum()
    return 1 - pw @ diag


if __name__ == '__main__':
    a = torch.randn(10, 30)
    b = torch.randn(10, 30)
    c = torch.randn(10, 20)
    svcca_distance(a, b)
    pwcca_distance(a, c)
