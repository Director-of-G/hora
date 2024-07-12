from torch import nn

def get_activation(act='gleu'):
    print(f'=-' * 20)
    print(f'Using activation:{act}')
    if act == 'tanh':
        return nn.Tanh
    elif act == 'elu':
        return nn.ELU
    elif act == 'relu':
        return nn.ReLU
    elif act == 'leakyrelu':
        return nn.LeakyReLU
    elif act == 'sigmoid':
        return nn.Sigmoid
    elif act == 'swish':
        return nn.SiLU
    elif act == 'gelu':
        return nn.GELU
    