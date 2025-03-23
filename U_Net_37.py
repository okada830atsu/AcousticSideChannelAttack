
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1DSlice(nn.Module):
    def __init__(self, num_classes=37):
        super(UNet1DSlice, self).__init__()

        self.enc1 = self.conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.enc3 = self.conv_block(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bottleneck = self.conv_block(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(64 + 64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(32 + 32, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(16 + 16, 16)

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        e3_crop = e3[..., :u3.shape[2], :u3.shape[3]]  # crop if needed
        d3 = self.dec3(torch.cat([u3, e3_crop], dim=1))

        u2 = self.up2(d3)
        e2_crop = e2[..., :u2.shape[2], :u2.shape[3]]
        d2 = self.dec2(torch.cat([u2, e2_crop], dim=1))

        u1 = self.up1(d2)
        e1_crop = e1[..., :u1.shape[2], :u1.shape[3]]
        d1 = self.dec1(torch.cat([u1, e1_crop], dim=1))

        out = self.final_conv(d1)
        out = out.mean(dim=2)  # average over mel-band (height)
        out = out.permute(0, 2, 1)  # (B, T, C)

        return out