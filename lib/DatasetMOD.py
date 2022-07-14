import torch
from torchvision import transforms
from torch.utils.data import Dataset

import os
import numpy as np
import cv2
import pandas as pd

from lib.Dataset_utils import compute_hwl_mean,crop_img

class Dataset(Dataset):
	def __init__(self,dataloader,bin_num=2, overlap=np.pi/32,mode='training'):

		self.mode = mode
		self.dataloader = dataloader
		self.id_mapping = self._get_id_mapping(dataloader)

		# creating bin
		interval = np.pi*2/bin_num

		# create BinX and BinY
		binsX = [Bin(angle_min=i*interval,angle_max=(i+1)*interval,overlap=overlap) for i in range(bin_num)]
		binsY = [Bin(angle_min=i*interval,angle_max=(i+1)*interval,overlap=overlap) for i in range(bin_num)]

		# add combination of binx and biny
		self.bins = list()
		for binx in binsX:
			for biny in binsY:
				self.bins.append([binx,biny])

		# compute dimension mean
		self.dim_mean = compute_hwl_mean(os.path.join(dataloader.base_dir,'label_2'))

	def _get_id_mapping(self,dataloader):
		mapping = dict()
		idx = 0

		files = dataloader.data
		for i,file in enumerate(files):
			for j,obj in enumerate(file.obj):
				mapping[idx] = {'file_id':i,'obj_id':j}
				idx += 1

		return mapping

	# should return (Input, Label)
	def __getitem__(self, idx):
		file = self.dataloader.data[self.id_mapping[idx]['file_id']]
		obj = file.obj[self.id_mapping[idx]['obj_id']]

		box_2d = (obj.xmin,obj.ymin,obj.xmax,obj.ymax)

		full_img = cv2.imread(file.img_path)
		cropped_img = crop_img(full_img.copy(),box_2d,self.mode)

		orient,conf = self.get_orientation(obj.alphax,obj.alphay)

		label = dict()
		label['box_2d'] = box_2d

		label['dimension'] = np.array([obj.h,obj.w,obj.l])

		label['alphax'] = obj.alphax
		label['alphay'] = obj.alphay

		label['rx'] = obj.rx
		label['ry'] = obj.ry

		label['orient'] = orient
		label['conf'] = conf

		label['depth'] = obj.z

		return cropped_img,label

	def __len__(self):
		return len(self.id_mapping)

	def get_orientation(self,alphax,alphay):
		orient = list()
		conf = list()

		for binx,biny in self.bins:
			orient.append(binx.get_orientation(alphax) + biny.get_orientation(alphay)) # [cos(x),sin(x),cos(y),sin(y)]
			conf.append(int(binx.is_between(alphax) and biny.is_between(alphay)))

		return np.array(orient),np.array(conf)

class Bin(object):
	"""docstring for Bin"""
	def __init__(self, angle_min, angle_max,overlap=0.0001):
		self.overlap = overlap

		self.min = angle_min - overlap
		self.max = angle_max + overlap

		self.range = self._get_range()
		self.center = self._get_center()

	def _get_center(self):
		return self.min + abs(self.max - self.min)/2

	def _get_range(self):
		return abs(self.max - self.min)%(np.pi*2)

	def is_between(self,angle):
		angle = angle%(np.pi*2)

		return (angle > self.min) and (angle < self.max)

	def get_orientation(self,angle):
		angle = angle%(np.pi*2)

		if not self.is_between(angle):
			return [0,0]

		angle_diff = angle - self.center

		orient = [np.cos(angle_diff), np.sin(angle_diff)]

		return orient

	def get_angle(self,orient):
		cos = orient[0]
		sin = orient[1]
		
		if sin == 0 and cos == 0:
			return 0 

		angle = np.arctan2(sin, cos)
		angle = angle + self.center

		return angle