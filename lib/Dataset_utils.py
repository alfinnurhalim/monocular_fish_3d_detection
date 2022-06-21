import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import transforms
from matplotlib.pyplot import figure
from sklearn.metrics import average_precision_score,mean_absolute_error,mean_squared_error 

def crop_img(img,box_2d):
	(x0,y0,x1,y1) = box_2d
	cropped = img.copy()

	cropped = cropped[y0:y0+(abs(y1-y0)),x0:x0+(abs(x1-x0))]
	cropped = cropped/255

	cropped = cv2.resize(src = cropped, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
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
	mse = mean_squared_error(y_true, y_pred)

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

	plt.xlabel('Ry Groundtruth')
	# naming the y axis
	plt.ylabel('Error')
	plt.ylim([0, 180])

	plt.show()