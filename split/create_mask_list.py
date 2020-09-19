import cv2
import skimage.io
#from tqdm.notebook import tqdm
import zipfile
import skimage
from tqdm import tqdm
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import csv
import random

if __name__ == '__main__':
	file_rad='rad_with_mask.txt'
	file_kar='kar_with_mask.txt'
	fc=open('train.csv')
	data=[]
	with open('train.csv') as fn:
		f_csv = csv.reader(fn)
		for rows in f_csv:
			#print(rows)
			data.append([rows[0],rows[1],rows[2],rows[3]])
	data=data[1:]
	data_rad=[]
	data_kar=[]
	for i in tqdm(range(len(data))):
		rows=data[i]
		img_name=data[i][0]
		corp=data[i][1]
		label=data[i][2]
		if os.path.exists('../train_label_masks/'+img_name+'_mask.tiff'):
			if corp[0]=='r':
				data_rad.append([rows[0]+'\t'+rows[1]+'\t'+rows[2]+'\t'+rows[3]])
			else:
				data_kar.append([rows[0]+'\t'+rows[1]+'\t'+rows[2]+'\t'+rows[3]])
	with open(file_rad,'w') as ffn:
		for j in range(len(data_rad)):
			print(data_rad[j])
			ffn.writelines(data_rad[j][0]+'\n')
	with open(file_kar,'w') as ffn:
		for j in range(len(data_kar)):
			print(data_kar[j])
			ffn.writelines(data_kar[j][0]+'\n')
			