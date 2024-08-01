import torch
import torch.nn as nn
import torch.nn.init as init
def get_initializer(method):
    if method == 'xavier':
        return xavier_uniform_init
    elif method == 'kaiming':
        return kaiming_uniform_init
    elif method == 'normal':
        return normal_init
    elif method == 'orthogonal':
        return orthogonal_init
    elif method == 'constant':
        return lambda m: constant_init(m, value=0.1)

def xavier_uniform_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def kaiming_uniform_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def normal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=0, std=1)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def orthogonal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.orthogonal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def constant_init(m, value=0.1):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.constant_(m.weight, value)
        if m.bias is not None:
            init.constant_(m.bias, value)