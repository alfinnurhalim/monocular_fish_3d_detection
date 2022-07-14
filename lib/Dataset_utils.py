import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import transforms
from matplotlib.pyplot import figure
from sklearn.metrics import average_precision_score,mean_absolute_error,mean_squared_error 
import imgaug.augmenters as iaa

def compose_transform(img,mode):

	transforms = list()
	h,w,_ = img.shape

	# transforms.append(iaa.PadToFixedSize(width=512, height=512))

	if h>w:
		transforms.append(iaa.Resize({"height": 224, "width": "keep-aspect-ratio"}))
	else:
		transforms.append(iaa.Resize({"height": "keep-aspect-ratio", "width": 224}))

	# if mode =='training':
	# 	transforms.append(iaa.PadToFixedSize(width=224, height=224))
	# else:
	transforms.append(iaa.PadToFixedSize(width=224, height=224, position="center"))

	# if mode == 'training':
	# transforms.append(iaa.SaltAndPepper(0.1))
	# transforms.append(iaa.GammaContrast((0.5, 2.0)))
	# transforms.append(iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.0), add=(-5, 5)))
	# transforms.append(iaa.AddToHueAndSaturation((-25, 25), per_channel=True))
		# transforms.append(iaa.BlendAlphaCheckerboard(nb_rows=2, nb_cols=(1, 4),foreground=iaa.AddToHue((-100, 100))))
		# transforms.append(iaa.Cutout(nb_iterations=2))
		# transforms.append(iaa.CoarseDropout(0.02, size_percent=0.10, per_channel=0.5))
		# transforms.append(iaa.Jigsaw(nb_rows=10, nb_cols=10))

	return transforms

def crop_img(img,box_2d,mode = 'training'):
	(x0,y0,x1,y1) = box_2d

	h = abs(y1-y0)
	w = abs(x1-x0)

	x0 = int(x0 - w*0.1)
	x1 = int(x1 + w*0.1)
	
	y0 = int(y0 - h*0.1)
	y1 = int(y1 + h*0.1)	

	cropped = img.copy()

	cropped = cropped[y0:y0+(abs(y1-y0)),x0:x0+(abs(x1-x0))]

	# do data augemntation 
	# if mode == 'training':
	seq = iaa.Sequential(compose_transform(cropped,mode))
	cropped = seq(image=cropped.copy())
	# cropped = cv2.resize(cropped,(224,224))

	# cropped = cropped/255
	
	process = transforms.Compose ([
            transforms.ToTensor()
        ])

	cropped = process(cropped)

	return cropped  

def compute_hwl_mean(label_dir):
	data = list()

	header = ['class','trunc','occlusion','alphax','xmin','ymin','xmax','ymax','h','w','l','x','y','z','rx','ry','rz','alphay']
	for filename in os.listdir(label_dir):
		ann = pd.read_csv(os.path.join(label_dir,filename),sep = ' ',names=header)
		data.append(ann)
	
	df =data[0]
	for d in (data[1:]):
		df = df.append(d,ignore_index=True)

	l = df['l'].mean()
	h = df['h'].mean()
	w = df['w'].mean()

	return np.array([h,w,l])

def get_eval_metric(res):
	df = pd.DataFrame(res,columns=['gt','pred'])

	df['gt'] = df['gt']%(np.pi*2)
	df['pred'] = df['pred']%(np.pi*2)

	result = df.values.tolist()

	y_true = np.array([x[1] for x in result])
	y_pred = np.array([x[0] for x in result])

	mae = np.degrees(mean_absolute_error(y_true, y_pred))
	mse = np.degrees(mean_squared_error(y_true, y_pred))

	return mae,mse 

def plot_error(res):
	df = pd.DataFrame(res,columns=['gt','pred'])

	df['gt'] = df['gt']%(np.pi*2)
	df['pred'] = df['pred']%(np.pi*2)

	df_mean = df
	df_mean['error'] = abs(df_mean['pred'] - df_mean['gt'])
	df.loc[df['error'] > np.pi, 'error'] = np.pi*2 - df['error'] 

	df_mean = df_mean[['gt','error']]
	df_mean = df.groupby(['gt']).mean().reset_index()

	df_mean['deg'] = df_mean['gt'].apply(np.degrees)
	df_mean['deg_err'] = df_mean['error'].apply(np.degrees)

	df_mean = df_mean.sort_values(['error'])
	df_mean = df_mean.iloc[:-1]
	df_mean = df_mean.sort_values(['deg'])

	error = df_mean['deg_err'].to_list()
	deg = df_mean['deg']

	figure(figsize=(8, 6), dpi=80)
	plt.plot(deg,error)

	plt.xlabel('RX Groundtruth')
	# naming the y axis
	plt.ylabel('Error')
	plt.ylim([0, 180])
	plt.xlim([0, 360])

	plt.show()