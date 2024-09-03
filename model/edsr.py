from model import common
import torch.nn as nn
import torch


def make_model(scale, upsampler, input_dim, config=None):
    return EDSR(
        kernel_size=3,
        n_resblocks=16,
        n_feats=64,
        res_scale=1,
        scale=scale,
        upsampler=upsampler,
        input_dim=input_dim,
    )


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        conv,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        bn=False,
        act=nn.ReLU(inplace=True),
    ):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(
        self,
        conv=default_conv,
        kernel_size=3,
        n_resblocks=16,
        n_feats=64,
        res_scale=1,
        scale=1,
        upsampler="none",
        input_dim=1,
    ):
        super(EDSR, self).__init__()
        act = nn.ReLU(True)
        self.upsampler = upsampler
        # define head module
        if input_dim == 9:  # sinogram unfolded
            m_head = [conv(input_dim, n_feats, 1)]
        else:
            m_head = [conv(input_dim, n_feats, 3)]

        # define body module
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))  # type: ignore

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        # define tail module
        if upsampler == "none":
            self.out_dim = n_feats
        elif upsampler == "pixelshuffle":
            self.out_dim = 1
            m_tail = [
                common.Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, 1, kernel_size),
            ]
            self.tail = nn.Sequential(*m_tail)
        elif upsampler == "conv":
            self.out_dim = 1
            m_tail = [conv(n_feats, 1, kernel_size)]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        if not self.upsampler == "none":
            res = self.tail(res)
        return res
