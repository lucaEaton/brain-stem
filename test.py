import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import musdb
from torch.nn import MaxPool2d

# 1. We get raw data y using librosa or something
# 2. I call my stft on my raw data, which gives us phase and magnitude information
# 3. I feed the magnitude to my encoder/decoder
# 4. Whatever the model outputs, we use inverse stft to get a "song"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()
        self.c0 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.b0 = nn.BatchNorm2d(out_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = torch.relu(self.max0(self.b0(self.c0(x))))
        x = torch.relu(self.b1(self.c1(x)))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, d_prob=0.4):
        super(UpBlock, self).__init__()
        self.upsample0 = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, padding=1)
        self.b0 = nn.BatchNorm2d(out_c)
        self.upsample1 = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(out_c)
        self.upsample2 = nn.ConvTranspose2d(in_c, out_c, kernel_size=5, padding=2)
        self.b2 = nn.BatchNorm2d(out_c)
        self.d2 = nn.Dropout(d_prob)

    def forward(self, x):
        x = torch.relu(self.b0(self.u0(x)))
        x = torch.relu(self.b1(self.u1(x)))
        x = torch.relu(self.d2(self.b2(self.u2(x))))
        return x


class UNet(nn.Module):
    def __init__(self, in_c, out_c, d_prob=0.4):
        super(UNet, self).__init__()
        # beginning block
        self.c0 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.b0 = nn.BatchNorm2d(out_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(out_c)

        # encode/decode blocks
        self.encode_block = ConvBlock(in_c, out_c)
        self.decode_block = UpBlock(out_c, out_c, d_prob)

        # beginning decode upsampling block
        self.upsample0 = nn.ConvTranspose2d(in_c, out_c, kernel_size=5, padding=2)
        self.b2 = nn.BatchNorm2d(out_c)
        self.d0 = nn.Dropout(d_prob)

        # ending decode block
        self.c2 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=1)

        # deconv2d ~ 3x3@32 ~ NOTE: repeat twice
        self.upsample1 = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(out_c)
        self.d1 = nn.Dropout(d_prob)

    def forward(self, x):
        # First two conv2d blocks ~ 3x3@32
        x_begin = torch.relu(self.b0(self.c0(torch.relu(self.b1(self.c1(x))))))

        # encode ~ x0-x4 = each encode block provided by Fig.3
        x0 = self.encode_block(x_begin)
        x1 = self.encode_block(x0)
        x2 = self.encode_block(x1)
        x3 = self.encode_block(x2)
        x4 = self.encode_block(x3)

        # decode ~ y0-y5 = each decode block provided by Fig.3
        y0 = torch.relu(self.d0(self.b2(self.upsample0(x4))))
        cat0 = torch.cat((y0, x4), dim=1)
        y1 = self.decode_block(cat0)
        cat1 = torch.cat((y1, x3), dim=1)
        y2 = self.decode_block(cat1)
        cat2 = torch.cat((y2, x2), dim=1)
        y3 = self.decode_block(cat2)
        cat3 = torch.cat((y3, x1), dim=1)
        y4 = self.decode_block(cat3)

        # last 3 ending blocks ~ 2x deconv2d~3x3@32 & final conv 1x1@k
        deconv0 = torch.relu(self.d1(self.b3(self.upsample1(y4))))
        deconv1 = torch.relu(self.d1(self.b3(self.upsample1(deconv0))))

        y5 = torch.relu(self.c2(deconv1))

        return y5


# Change in n out, I believe we should hard code the in channels
# and out channels as it provides us them in the paper
model_stem_splitter = UNet(in_c=0, out_c=0, d_prob=0.4)
model_stem_splitter = model_stem_splitter.to(device)  # either cuda (for derek) or mps(for me(luca))

# FOR DEREK :
# later within training loop I believe we (1/4)
# use this loss function to calc predicted n target (2/4)
# in order to calc total loss as displayed (3/4)
# in the Fig on Pg.3, Section 2.2 (4/4)
loss_fn = nn.L1Loss()

# learning rate = 1e^-3 (first 20 epochs) ~ weight decay (1e^-6) Pg.3, Section 2.2
optimizer = torch.optim.Adam(model_stem_splitter.parameters(), lr=0.001, weight_decay=0.000001)

torch.manual_seed(67)

epochs = 40

for epoch in range(epochs):
    model_stem_splitter.train()
