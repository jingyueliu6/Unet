import tabnanny

from unet.unet_part import *

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.output(x)
        x = self.act(x)
        logits = self.act(x)
        return logits
