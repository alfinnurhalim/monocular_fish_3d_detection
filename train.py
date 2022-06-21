import sys
import os
import cv2
import torch
import wandb

import numpy as np
import pandas as pd
import torch.nn as nn

from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from torch.utils import data
from torch.autograd import Variable
from torchvision.models import vgg,resnet50

BASE_DIR = os.path.dirname(os.path.abspath('.'))
sys.path.append(BASE_DIR) 

from lib.KITTIDataloader import KITTIDataloader
from lib.Dataset import Dataset,Bin
from lib.Model import Model, OrientationLoss

# ============================== Config ==============================
# dataset path
dataset_path = os.path.join('dataset','20220618_qd3dt_512')
output_path = os.path.join('weights')

# number of bins
bins = 2

# number of epoch
epochs = 40

# batch size
batch_size = 64

# lr
lr = 0.001

# Hyper-params
alpha = 0.6
w = 2.1

# =========================== End of Config ==============================
wandb.init(project="3d_det")

dataloader = KITTIDataloader(os.path.join(dataset_path,'KITTI/detection/training/'))
dataloader.load_from_file()

dataset = Dataset(dataloader,bin_num=bins)


params = {'batch_size': batch_size,
		'shuffle': True,
		'num_workers': 6}

wandb.config = {
		'learning_rate': lr,
		'epochs': epochs,
		'batch_size': batch_size,
		'bins' : bins}

generator = data.DataLoader(dataset, **params)

model = Model(bins=bins,backbone='resnet50').cuda()
opt_SGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

orient_loss_func = OrientationLoss
conf_loss_func = nn.CrossEntropyLoss().cuda()
dim_loss_func = nn.MSELoss().cuda()
depth_loss_func = nn.SmoothL1Loss().cuda()

total_step = int(len(dataset)/batch_size)

for epoch in range(epochs):
	step = 0
	for local_batch, local_labels in generator:
		truth_orient = local_labels['orientX'].float().cuda()
		truth_conf = local_labels['confX'].long().cuda()
		truth_dim = local_labels['dimension'].float().cuda()
		truth_depth = local_labels['depth'].float().cuda()

		local_batch=local_batch.float().cuda()
		[orient, conf, dim, depth] = model(local_batch)

		orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
		dim_loss = dim_loss_func(dim, truth_dim)
		depth_loss = depth_loss_func(depth, truth_depth)

		truth_conf = torch.max(truth_conf, dim=1)[1]
		conf_loss = conf_loss_func(conf, truth_conf)

		loss_theta = conf_loss + w * orient_loss
		loss = alpha * dim_loss + loss_theta + depth_loss

		opt_SGD.zero_grad()
		loss.backward()
		opt_SGD.step()

		if step % 60 == 0:
			wandb.log({"loss": loss,
				"orient_loss": loss_theta,
				"dim_loss":dim_loss,
				"depth_loss":depth_loss})
			print('Epoch {} | {}/{}: Loss {}, Dim Loss {}, Orient Loss {}, Depth Loss {}'.format(epoch,step,total_step,loss,dim_loss,loss_theta,depth_loss))
		    
		step +=1
	    
	if (epoch) % 1 == 0:
		name = os.path.join(output_path,'epoch_{}.pkl'.format(epoch))
		print("====================")
		print ("Done with epoch %s!" % epoch)
		print ("Saving weights as %s ..." % name)
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': opt_SGD.state_dict(),
			'loss': loss
			}, name)
		print("====================")