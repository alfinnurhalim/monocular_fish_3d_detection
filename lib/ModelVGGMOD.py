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

    # print('the shape is',orientGT_batch[:,1])
    theta_diff_X = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    theta_diff_Y = torch.atan2(orientGT_batch[:,3], orientGT_batch[:,2])

    estimated_theta_diff_X = torch.atan2(orient_batch[:,1], orient_batch[:,0])
    estimated_theta_diff_Y = torch.atan2(orient_batch[:,3], orient_batch[:,2])

    error_X = (theta_diff_X - estimated_theta_diff_X)
    error_Y = (theta_diff_Y - estimated_theta_diff_Y)

    error_X = (2 - (2/batch_size * torch.cos(error_X).sum()))/bins
    error_Y = (2 - (2/batch_size * torch.cos(error_Y).sum()))/bins

    return error_Y+error_X

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
                    nn.Linear(256, bins*2*2*2) # to get sin and cos for rx and ry. number of bins doubled
                )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2),
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

    def forward(self, x):
        x = self.backbone(x) # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)

        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins*2, 2*2)

        confidence = self.confidence(x)

        dimension = self.dimension(x)

        depth = self.depth(x)

        return orientation, confidence, dimension, depth

    def _get_backbone(self,name):

        # will support different backbone later
        backbone = vgg.vgg19_bn(pretrained=True)
        backbone = backbone.features

        # resnet
        # modules = list(backbone.children())[:-1]
        # backbone = nn.Sequential(*modules)

        return backbone

