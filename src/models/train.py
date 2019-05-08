# pylint: disable=missing-docstring, invalid-name,  bad-indentation,
# pylint: disable=bad-continuation, bad-whitespace
# pylint: disable=too-many-locals, line-to-long, too-many-instance-attributes,
import math
import itertools
import torch
from torch import optim
import torch.nn.functional as F
from pytorch_utils import batcher, vis, util
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torchvision.datasets import DatasetFolder
from skimage import draw

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
        # Model loading
        #
        self.Net = models.UNet().to(device)

        #
        # Resume if necessary
        #
        if args.resume is not None:
            largs, optimizer = utils.load_nets(args.resume,self.Net)
            args = largs
            self.args = largs
            print("Args over ridden by --resume: %s" % (self.args,))

            self.optimizer = optimizer
        else:
            #
            # Optimizers
            #
            self.optimizer = optim.Adam(self.Net.parameters(), lr=args.lr)

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

          if self.args.weighted_ce:
            weight = target
            weight[weight == 0] = 0.1
            loss = F.binary_cross_entropy(y,target, weight=weight)
          else:
            loss = F.binary_cross_entropy(y,target)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          self.batch.add('loss',loss.item())
          self.batch.add('gradients', utils.grad_norm(self.Net.parameters()))

          #
          # Progress reporting
          #
          if i % 12 == 0:
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

    def test_average(self, loader, iters=5):
        loss = 0.0
        for i in range(iters):
            x, target = next(loader.__iter__())
            x = x.to(device=self.device, non_blocking=True) # p(x,y)
            target = target.to(device=self.device, non_blocking=True).unsqueeze(1)
            y = self.Net(x)
            loss += F.binary_cross_entropy(y,target).item()

        return loss/iters, x, target, y

    def test(self,epoch):
        '''
            Generate some logging information that is too computationally
            intensive to log on each batch.

            :param epoch (int): number of epochs trained, used for logging
        '''
        loss, x, target, y = self.test_average(self.test_loader)
        self.log.add_scalar('test/test_loss',loss,epoch)

        x = x.to('cpu').detach()[0]
        x = (x - torch.min(x[:])) / (torch.max(x[:]) - torch.min(x[:]))
        r,c = draw.polygon_perimeter([92,92,    92+388,92+388],
                                     [92,92+388,92+388,92])
        x[1,r,c] = 1
        self.log.add_image('images/real', x,  epoch)
        self.log.add_image('images/target', target.to('cpu')[0], epoch)
        self.log.add_image('images/segmented', y.to('cpu').detach()[0], epoch)


        loss, _, _, _ = self.test_average(self.train_loader)
        self.log.add_scalar('test/train_loss',loss,epoch)

    def save(self,epoch):
        folder_name = './%s_epoch%d'%(self.args.comment,epoch)
        nets = {
            'Net': self.Net
        }
        to_pickle = {
            'args': self.args,
            'optimizers':{
                'inference': self.optimizer
            }
        }
        util.save_nets('saves/' + folder_name,nets,to_pickle)

    def main(self):
        '''
            Perform training
        '''
        for epoch in range(int(self.args.epochs)):
            self.train(epoch)
            self.test(epoch)

            if self.args.save_interval != 0 and epoch % self.args.save_interval == 0:
                self.save(epoch)

        self.save(int(self.args.epochs))

if __name__ == '__main__':
    trainer = Train()
    trainer.main()
