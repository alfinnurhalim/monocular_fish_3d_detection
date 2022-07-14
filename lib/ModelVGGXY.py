import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg,resnet50

# def OrientationLoss(orient_batch, orientGT_batch, confGT_batch,bins):

#     batch_size = orient_batch.size()[0]
#     indexes = torch.max(confGT_batch, dim=1)[1]

#     # extract just the important bin
#     orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
#     orient_batch = orient_batch[torch.arange(batch_size), indexes]

#     theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
#     estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

#     error = (theta_diff - estimated_theta_diff)
#     return -1 * torch.cos(error).mean()

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch,bins):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    error = (theta_diff - estimated_theta_diff)
    return (2 - (2/batch_size * torch.cos(error).sum()))/bins

class Model(nn.Module):
    def __init__(self, bins=2,backbone = 'resnet50'):
        super(Model, self).__init__()
        
        self.bins = bins
        self.backbone = self._get_backbone(backbone)
        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) # to get sin and cos
                )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins),
                )
        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )

        self.depth = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 1)
                )

    def forward(self, x,rx=False):
        x = self.backbone(x) # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)

        orientX = self.orientation(x)
        orientX = orientX.view(-1, self.bins, 2)

        orientY = self.orientation(x)
        orientY = orientY.view(-1, self.bins, 2)

        if rx:
            for param in orientY.parameters():
                param.requires_grad = False
                
        confX = self.confidence(x)
        confY = self.confidence(x)

        dimension = self.dimension(x)

        depth = self.depth(x)

        return orientX, confX, orientY, confY, dimension, depth

    def _get_backbone(self,name):

        # will support different backbone later
        backbone = vgg.vgg19_bn(pretrained=True)
        backbone = backbone.features

        # resnet
        # modules = list(backbone.children())[:-1]
        # backbone = nn.Sequential(*modules)

        return backbone

