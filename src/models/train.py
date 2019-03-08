# pylint: disable=missing-docstring, invalid-name,  bad-indentation,
# pylint: disable=bad-continuation, bad-whitespace
# pylint: disable=too-many-locals, line-to-long, too-many-instance-attributes,
import math
import itertools
import torch
from torch import optim
import torch.nn.functional as F
from pytorch_utils import batcher, vis
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torchvision.datasets import DatasetFolder

import utils
import models

EPS = 1e-8

class Train():

    def __init__(self):
        '''
            Initialize Training
        '''
        #
        # Argument Parsing
        #
        args = utils.parse_args()
        print(args)
        use_cuda = torch.cuda.is_available() and not args.no_cuda
        device = torch.device('cuda' if use_cuda else 'cpu')
        torch.manual_seed(args.seed)
        if use_cuda:
          torch.cuda.manual_seed(args.seed)
        self.args = args
        self.device = device

        #
        # Data loading
        #
        if(args.dataset == 'VEINS_100'):
            train_loader, test_loader = utils.VEINS_100_loaders(args)
            self.train_loader = train_loader
            self.test_loader = test_loader
        else:
            exit("Invalid --data set option")

        #
        # Model loading
        #
        self.Net = models.UNet().to(device)
        if args.resume:
            # TODO: Load models here
            pass

        #
        # Optimizers
        #
        self.optimizer = optim.Adam(self.Net.parameters(), lr=args.lr)
        if args.resume:
            #TODO: Load optimizers here
            #optimizer.load_state_dict(torch.load('optimiser.pth'))
            pass

        #
        # Logging setup
        #
        self.log = SummaryWriter(args.logdir,args.comment) # Logging
        self.batch = batcher() # Logging

    def train(self,epoch):
        '''
            Perform one training epoch
        '''
        self.Net.train()
        for i, (x, target) in enumerate(self.train_loader):
          self.batch.batch()
          target = target.to(device=self.device, non_blocking=True).unsqueeze(1)
          x = x.to(device=self.device, non_blocking=True) # p(x,y)
          y = self.Net(x)

          # print(target.size())
          #loss = torch.sum((y - target)**2,dim=1)
          #loss = torch.mean(loss)
          loss = F.binary_cross_entropy(y,target)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          self.batch.add('loss',loss.item())
          self.batch.add('gradients', utils.grad_norm(self.Net.up1.parameters()))

          #
          # Progress reporting
          #
          if i % 125 == 0:
            self.log.add_image('images/real', x.to('cpu').detach()[0],  epoch * len(self.train_loader) + i)
            self.log.add_image('images/target', target.to('cpu')[0],  epoch * len(self.train_loader) + i)
            self.log.add_image('images/segmented', y.to('cpu').detach()[0],  epoch * len(self.train_loader) + i)

            print('Epoch: %d [%d/%d]: ' %
                   (
                     epoch,
                     i * len(x),
                     len(self.train_loader.dataset),
                   ),
                  end = '')
            self.batch.report()
            print('', flush=True)

            self.batch.write(self.log, epoch * len(self.train_loader) + i)

    def test(self,epoch):
        '''
            Generate some logging information that is too computationally
            intensive to log on each batch.

            :param epoch (int): number of epochs trained, used for logging
        '''
        pass

    def main(self):
        '''
            Perform training
        '''
        for epoch in range(int(self.args.epochs)):
            self.train(epoch)
            self.test(epoch)

            if self.args.save_interval != 0 and epoch % self.args.save_interval == 0:
                # TODO: Save models here
                # torch.save(optimizer.state_dict(), 'optimizer.pth')
                # torch.save(model.state_dict(), 'model.pth')
                pass

if __name__ == '__main__':
    trainer = Train()
    trainer.main()
