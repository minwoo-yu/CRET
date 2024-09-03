from model import common
import torch
import torch.nn as nn


def make_model(scale, upsampler, input_dim, config=None):
    return RDN(RDNkSize=3, RDNconfig="A", G0=64, scale=scale, upsampler=upsampler, input_dim=input_dim)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1), nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, RDNkSize=3, RDNconfig="B", G0=64, scale=4, upsampler="none", input_dim=1):
        super(RDN, self).__init__()
        r = scale
        kSize = RDNkSize
        self.upsampler = upsampler

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            "A": (20, 6, 32),
            "B": (16, 8, 64),
        }[RDNconfig]

        # Shallow feature extraction net
        if input_dim == 9:  # sinogram unfolded
            self.SFENet1 = nn.Conv2d(9, G0, kernel_size=1, padding=0, stride=1)
        else:
            self.SFENet1 = nn.Conv2d(1, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))

        # Global Feature Fusion
        self.GFF = nn.Sequential(
            *[nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)]
        )

        # define upsampling module
        if upsampler == "none":
            self.out_dim = G0
        elif upsampler == "pixelshuffle":
            self.out_dim = 1
            # Up-sampling net
            if r == 1:
                self.UPNet = nn.Sequential(*[nn.Conv2d(G0, 1, kSize, padding=(kSize - 1) // 2, stride=1)])
            elif r == 2 or r == 3:
                self.UPNet = nn.Sequential(
                    *[
                        nn.Conv2d(G0, G * r, kSize, padding=(kSize - 1) // 2, stride=1),
                        common.PixelShuffle1D(r),
                        nn.Conv2d(G, 1, kSize, padding=(kSize - 1) // 2, stride=1),
                    ]
                )
            elif r == 4:
                self.UPNet = nn.Sequential(
                    *[
                        nn.Conv2d(G0, G * 2, kSize, padding=(kSize - 1) // 2, stride=1),
                        common.PixelShuffle1D(2),
                        nn.Conv2d(G, G * 2, kSize, padding=(kSize - 1) // 2, stride=1),
                        common.PixelShuffle1D(2),
                        nn.Conv2d(G, 1, kSize, padding=(kSize - 1) // 2, stride=1),
                    ]
                )
        elif upsampler == "conv":
            self.out_dim = 1
            self.UPNet = nn.Sequential(
                *[
                    nn.Conv2d(G0, 1, kSize, padding=(kSize - 1) // 2, stride=1),
                ]
            )

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        if self.upsampler == "none":
            return x
        else:
            return self.UPNet(x)
