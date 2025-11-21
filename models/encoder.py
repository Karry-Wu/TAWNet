import torch
from torch import nn
from models.p2t import p2t_base

class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=True, group=1, dilation=1,
                 act=nn.ReLU()):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias, groups=group,
                      dilation=dilation),
            nn.BatchNorm2d(out_channel),
            act)

    def forward(self, x):
        return self.conv(x)


class P2tBackbone(nn.Module):
    def __init__(self, p2t_path):
        super(P2tBackbone, self).__init__()
        self.p2t_backbone1 = p2t_base()
        self.p2t_backbone2 = p2t_base()
        if p2t_path is not None:
            self.p2t_backbone1.init_weights(p2t_path)
            self.p2t_backbone2.init_weights(p2t_path)

    def forward(self, x, y):
        x_out = self.p2t_backbone1(x)
        y_out = self.p2t_backbone2(y)
        return x_out, y_out
