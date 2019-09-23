# pylint: disable=missing-docstring, bad-whitespace, line-too-long,
# pylint: disable=invalid-name, bad-continuation, arguments-differ,
# pylint: disable=too-many-instance-attributes, no-self-use, too-many-arguments
from torch.nn import Module
import layers

class UNet(Module):
    '''
     U-Net network from
        Ronneberger, Olaf, Philipp Fischer, and Thomas Brox.
        “U-Net: Convolutional Networks for Biomedical Image Segmentation.”
        ArXiv:1505.04597 [Cs], May 18, 2015. http://arxiv.org/abs/1505.04597.

    '''
    def __init__(self):
        super().__init__()
        self.down1 = layers.Down1()
        self.down2 = layers.Down2()
        self.down3 = layers.Down3()
        self.down4 = layers.Down4()

        self.up5 = layers.Up5()
        self.up4 = layers.Up4()
        self.up3 = layers.Up3()
        self.up2 = layers.Up2()
        self.up1 = layers.Up1()

    def forward(self,x):
        x,y1 = self.down1(x)
        x,y2 = self.down2(x)
        x,y3 = self.down3(x)
        x,y4 = self.down4(x)
        x = self.up5(x)
        x = self.up4(x,y4)
        x = self.up3(x,y3)
        x = self.up2(x,y2)
        x = self.up1(x,y1)

        return x
