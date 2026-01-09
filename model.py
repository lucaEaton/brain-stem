import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, d_prob=0.4):
        super(UpBlock, self).__init__()
        self.u = nn.ConvTranspose2d(in_c, out_c, kernel_size=5,
                                    stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.d = nn.Dropout(d_prob)
        self.r = ConvBlock(out_c * 2, out_c)

    def forward(self, x, skip):
        x = F.relu(self.bn(self.u(x)), inplace=True)
        x = self.d(x)
        dH = skip.size()[2] - x.size()[2]
        dW = skip.size()[3] - x.size()[3]
        if dH != 0 or dW != 0:
            x = F.pad(x, [dW // 2, dW - dW // 2, dH // 2, dH - dH // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.r(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_c=1, out_c=2):
        super(UNet, self).__init__()
        self.cb0 = ConvBlock(in_c, 32)
        self.cb1 = ConvBlock(32, 64)
        self.cb2 = ConvBlock(64, 128)
        self.cb3 = ConvBlock(128, 256)
        self.cb4 = ConvBlock(256, 512)
        self.ub4 = UpBlock(512, 256)
        self.ub3 = UpBlock(256, 128)
        self.ub2 = UpBlock(128, 64)
        self.ub1 = UpBlock(64, 32)
        self.fc = nn.Conv2d(32, out_channels=out_c, kernel_size=1)
        self.mp = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x0 = self.cb0(x)
        xp0 = self.mp(x0)
        x1 = self.cb1(xp0)
        xp1 = self.mp(x1)
        x2 = self.cb2(xp1)
        xp2 = self.mp(x2)
        x3 = self.cb3(xp2)
        xp3 = self.mp(x3)

        x4 = self.cb4(xp3)

        xd4 = self.ub4(x4, x3)
        xd3 = self.ub3(xd4, x2)
        xd2 = self.ub2(xd3, x1)
        xd1 = self.ub1(xd2, x0)

        return torch.relu(self.fc(xd1))
