import torch
import torch.nn as nn
import torch.nn.functional as F

def create_layer(Nin,Nout,kernel=3,padding=0,stride=1):

    return nn.Sequential(
        nn.Conv2d(Nin,Nout,kernel,padding=padding,stride=stride),
        nn.ReLU()
    )

def create_layer_t(Nin, Nout, stride=2, output_padding=0, padding=0, kernel=2):

    return nn.Sequential(
        nn.ConvTranspose2d(Nin, Nout,
                           kernel_size=kernel,
                           stride=stride,
                           padding=padding,
                           output_padding=output_padding),
        nn.ReLU()
    )

NGF = 64

class Down1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Sequential( # <-N x 572 x 572
            create_layer(3,NGF),
            create_layer(NGF,NGF),
        ) # -> N x 568 x 568

    def forward(self,x):
        '''
            x input dim: 1 x 572 x 572

            x output dim: 284 x 284
            y (cropped) output dim: 392 x 392
        '''
        x = self.layer(x)

        return  F.max_pool2d(x,2), x[:,:,88:-88,88:-88]

class Down2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential( # <-N x 284 x 284
            create_layer(NGF,NGF*2),
            create_layer(NGF*2,NGF*2),
        ) # -> N x 280 x 280

    def forward(self,x):
        '''
            x input dim: 284 x 284

            x output dim: 140 x 140
            y (cropped) output dim: 200 x 200
        '''
        x = self.layer(x)

        return  F.max_pool2d(x,2), x[:,:,40:-40,40:-40]

class Down3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential( # <-N x 140 x 140
            create_layer(NGF*2,NGF*4),
            create_layer(NGF*4,NGF*4),
        ) # -> N x 136 x 136

    def forward(self,x):
        '''
            x input dim: 140 x 140

            x output dim: 68 x 68
            y (cropped) output dim: 104 x 104
        '''
        x = self.layer(x)

        return  F.max_pool2d(x,2), x[:,:,16:-16,16:-16]

class Down4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential( # <-N x 68 x 68
            create_layer(NGF*4,NGF*8),
            create_layer(NGF*8,NGF*8),
        ) # -> N x 64 x 64

    def forward(self,x):
        '''
            x input dim: 68 x 68

            x output dim: 32 x 32
            y (cropped) output dim: 56 x 56
        '''
        x = self.layer(x)

        return  F.max_pool2d(x,2), x[:,:,4:-4,4:-4]

class Up5(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential( # <-N x 32 x 32
            create_layer(NGF*8,NGF*16),
            create_layer(NGF*16,NGF*16),
            create_layer_t(NGF*16,NGF*8)
        ) # -> N x 64 x 64

    def forward(self,x):
        '''
            x input dim: 32 x 32

            x output dim: 56 x 56
        '''
        return self.layer(x)

class Up4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential( # <-(Nx2) x 56 x 56
            create_layer(NGF*16,NGF*8),
            create_layer(NGF*8,NGF*8),
            create_layer_t(NGF*8,NGF*4)
        ) # -> N x 64 x 64

    def forward(self,x,y):
        '''
            x,y input dim: 56 x 56

            x output dim: 104 x 104
        '''
        return self.layer(torch.cat((x,y),dim=1))

class Up3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential( # <-(Nx2) x 104 x 104
            create_layer(NGF*8,NGF*4),
            create_layer(NGF*4,NGF*4),
            create_layer_t(NGF*4,NGF*2)
        ) # -> N x 100 x 100

    def forward(self,x,y):
        '''
            x,y input dim: 104 x 104

            x output dim: 200 x 200
        '''
        return self.layer(torch.cat((x,y),dim=1))

class Up2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential( # <-(Nx2) x 200 x 200
            create_layer(NGF*4,NGF*2),
            create_layer(NGF*2,NGF*2),
            create_layer_t(NGF*2,NGF)
        ) # -> N x 196 x 196

    def forward(self,x,y):
        '''
            x,y input dim: 200 x 200

            x output dim: 392 x 392
        '''
        return self.layer(torch.cat((x,y),dim=1))

class Up1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential( # <-(Nx2) x 200 x 200
            create_layer(NGF*2,NGF),
            create_layer(NGF,NGF),
            create_layer(NGF,1,kernel=1,stride=1),
            nn.Sigmoid()
        ) # -> 1 x 388 x 388

    def forward(self,x,y):
        '''
            x,y input dim: 392 x 392

            x output dim: 388 x 388
        '''
        return self.layer(torch.cat((x,y),dim=1))