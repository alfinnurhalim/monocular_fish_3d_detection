import os
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

class KITTIDataloader(object):
	"""docstring for KITTIDataloader"""
	def __init__(self, base_dir):

		self.base_dir = base_dir
		self.data_path = self._get_data_path(base_dir)
		self.data = []

	def _get_data_path(self,base_dir):
		img_dir = os.path.join(base_dir,'image_2/')
		ann_dir = os.path.join(base_dir,'label_2/')
		calib_dir = os.path.join(base_dir,'calib/')
		
		data_path = [{'img':img_dir+img,'ann':ann_dir+ann,'calib':calib_dir+calib} for img,ann,calib in
								zip(sorted(os.listdir(img_dir)),
								sorted(os.listdir(ann_dir)),
								sorted(os.listdir(calib_dir))
							)]
		return data_path

	def load_from_file(self):
		print('loading dataset to memory')
		for path in tqdm(self.data_path):
			kitti_file = KITTI_File()
			kitti_file.load_from_file(path)
			self.data.append(kitti_file)


class KITTI_File(object):
	def __init__(self):
		self.data = None

		self.filename = None

		self.ann_path = None
		self.img_path = None
		self.calib_path = None

		self.camera = None
		self.obj = []

	def load_from_file(self,path):

		self.ann_path = path['ann']
		self.img_path = path['img']
		self.calib_path = path['calib']

		self.filename = os.path.splitext(os.path.basename(self.ann_path))[0]

		self.data = self._load_annotation(self.ann_path)

		# load cam
		self.camera = KITTI_Camera()
		self.camera.load_from_file(self.calib_path)

		for i in range(len(self.data)):
			kitti_obj = KITTI_Object()
			kitti_obj.load_from_file(self.data.iloc[i])
			self.obj.append(kitti_obj)

	def _image_transform(self,img):
		img = cv2.resize(img,(1024,1024))
		return img

	def _load_annotation(self,path):
		header = ['class','trunc','occlusion','alphax','xmin','ymin','xmax','ymax','h','w','l','x','y','z','rx','ry','rz','alphay']
		ann = pd.read_csv(self.ann_path,sep = ' ',names=header)

		return ann


class KITTI_Camera(object):
	def __init__(self):
		self.intrinsic = None
		self.extrinsic = None

	def _load_calib(self,path):
		with open(path, 'r') as f:
			lines = f.readlines()
			P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
			# P2 = [list(P2[i]) for i in range(len(P2))]
		return P2

	def load_from_file(self,calib_path):

		self.intrinsic = self._load_calib(calib_path)

		# convert to 4x3 mattrix
		# self.intrinsic[0].append(0)
		# self.intrinsic[1].append(0)
		# self.intrinsic[2].append(0)


		self.extrinsic = np.eye(3)

class KITTI_Object(object):
	def __init__(self):
		# class
		self.type = None
		self.truncated = 0
		self.occluded = 0

		self.alphax = None
		self.alphay = None

		# 2d bbox
		self.xmin = None
		self.ymin = None
		self.xmax = None
		self.ymax = None

		# 3d dimension
		self.h = None
		self.w = None
		self.l = None

		# 3d bbox
		self.x = None
		self.y = None
		self.z = None

		# 3d rotation
		self.rx = None
		self.ry = None
		self.rz = None

	def load_from_file(self,data):
		self.type = 'Car'

		self.alphax = data.alphax%(np.pi*2)
		self.alphay = data.alphay%(np.pi*2)

		self.xmin = int(data.xmin)
		self.ymin = int(data.ymin)
		self.xmax = int(data.xmax)
		self.ymax = int(data.ymax)

		self.h = data.h 
		self.w = data.w 
		self.l = data.l 

		self.x = data.x
		self.y = data.y
		self.z = data.z 

		self.rx = data.rx%(np.pi*2)
		self.ry = data.ry%(np.pi*2)
		self.rz = data.rz%(np.pi*2)