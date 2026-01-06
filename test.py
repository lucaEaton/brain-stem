import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        x = F.relu(self.b0(self.c0(x)))
        x = F.relu(self.b1(self.c1(x)))
        return x

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, d_prob):
        super(UpBlock, self).__init__()
        self.upsample0 = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, padding=1)
        self.b0 = nn.BatchNorm2d(out_c)
        self.upsample1 = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(out_c)
        self.upsample2 = nn.ConvTranspose2d(in_c, out_c, kernel_size=5, padding=2)
        self.b2 = nn.BatchNorm2d(out_c)
        self.d2 = nn.Dropout(d_prob)

    def forward(self, x):
        x = F.relu(self.b0(self.upsample0(x)))
        x = F.relu(self.b1(self.upsample1(x)))
        x = F.relu(self.d2(self.b2(self.upsample2(x))))
        return x

class UNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(UNet, self).__init__()
        #String together my ConvBlock(encoder) and UpBlock(decoder) with a concat

    def forward(self, x):

