import torch


class NanException(Exception):
    pass

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException