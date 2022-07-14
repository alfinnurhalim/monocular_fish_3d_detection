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
from matplotlib.pyplot import figure

from torch.utils import data
from torch.autograd import Variable
from torchvision.models import vgg,resnet50

BASE_DIR = os.path.dirname(os.path.abspath('.'))
sys.path.append(BASE_DIR) 

import lib.Dataset_utils as utils
from lib.KITTIDataloader import KITTIDataloader
from lib.DatasetMOD import Dataset,Bin
from lib.ModelVGGMOD import Model, OrientationLoss

# ============================== Config ==============================
# dataset path
dataset_path = os.path.join('../fish_conversion/data','20220707_qd3dt_512')
weight_path = os.path.join('weights','mod_epoch_10.pkl')

# number of bins
bin_num = 2

# batch size
batch_size = 32 #for testing

interval = np.pi*2/bin_num
angle_overlap = np.pi/4
# =========================== End of Config ==============================

dataloader = KITTIDataloader(os.path.join(dataset_path,'KITTI/detection/training/'))
dataloader.load_from_file()

dataset = Dataset(dataloader,bin_num=bin_num,overlap=angle_overlap,mode='eval')
params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': 6}
generator = data.DataLoader(dataset, **params)

model = Model(bins=bin_num,backbone='resnet50').cuda()
checkpoint = torch.load(weight_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# create bin
bins = list()

# create BinX and BinY
binsX = [Bin(angle_min=i*interval,angle_max=(i+1)*interval,overlap=angle_overlap) for i in range(bin_num)]
binsY = [Bin(angle_min=i*interval,angle_max=(i+1)*interval,overlap=angle_overlap) for i in range(bin_num)]

for binx in binsX:
    for biny in binsY:
        bins.append([binx,biny])

res = list()

for local_batch, local_labels in tqdm(generator):
	truth_orient = local_labels['orient'].float().cuda()
	truth_conf = local_labels['conf'].long().cuda()
	truth_dim = local_labels['dimension'].float().cuda()
	truth_depth = local_labels['depth'].float().cuda()

	local_batch=local_batch.float().cuda()
	[orient, conf, dim, depth] = model(local_batch)

	orient = orient.cpu().data.numpy()[0, :, :]
	conf = conf.cpu().data.numpy()[0, :]
	dim = dim.cpu().data.numpy()[0, :]
	depth = depth.cpu().data.numpy()[0,:]

	max_conf = np.argmax(conf)
	alphax = bins[max_conf][0].get_angle(orient[max_conf][:2])
	alphay = bins[max_conf][1].get_angle(orient[max_conf][2:])

	max_conf_gt = np.argmax(truth_conf.cpu().data.numpy()[0])
	alpha_gt_x = bins[max_conf_gt][0].get_angle(truth_orient.cpu().data.numpy()[0][max_conf_gt][:2])
	alpha_gt_y = bins[max_conf_gt][1].get_angle(truth_orient.cpu().data.numpy()[0][max_conf_gt][2:])

	res.append([alphay,alpha_gt_y])

mae,mse = utils.get_eval_metric(res)
print('mae',mae,'\nmse',mse)

utils.plot_error(res)