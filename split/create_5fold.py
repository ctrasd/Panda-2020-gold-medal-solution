from __future__ import print_function, absolute_import
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
	fc=open('train.csv')
	data=[]
	with open('train.csv') as fn:
		f_csv = csv.reader(fn)
		for rows in f_csv:
			#print(rows)
			data.append([rows[0]+'\t'+rows[1]+'\t'+rows[2]+'\t'+rows[3]])
	data=data[1:]
	random.shuffle(data)
	lens=len(data)
	fold_len=lens//5
	data_fold=[]
	for i in range(5):
		data_fold.append(data[i*fold_len:(i+1)*fold_len])
		file_name='fold_'+str(i)+'.txt'
		with open(file_name,'w') as ffn:
			for j in range(i*fold_len,(i+1)*fold_len):
				print(data[j])
				ffn.writelines(data[j][0]+'\n')
		file_name='train_flod_'+str(i)+'.txt'
		with open(file_name,'w') as ffn:
			for j in range(lens):
				if j>= i*fold_len and j<(i+1)*fold_len:
					continue
				#print(data[j])
				ffn.writelines(data[j][0]+'\n')