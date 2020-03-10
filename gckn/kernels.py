import torch


def exp(x, alpha):
    return torch.exp(alpha*(x - 1.))

def linear(x, alpha):
    return x

def d_exp(x, alpha):
    return alpha * exp(x, alpha)

kernels = {
    "exp": exp,
    "linear": linear
}

d_kernels = {
    "exp": d_exp,
    "linear": linear
}
