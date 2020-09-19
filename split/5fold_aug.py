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

	cnt=0
	for i in range(5):
		train_file='train_flod_'+str(i)+'.txt'
		test_file='fold_'+str(i)+'.txt'
		aug_train_file='all_aug_train_fold_'+str(i)+'.txt'
		aug_test_file='all_aug_fold_'+str(i)+'.txt'
		with open(test_file,'r') as fn:
			with open(aug_test_file,'w') as fn_w:
				for lines in fn:
					line=lines.split('\t')
					fn_w.writelines(lines)
					if(int(line[2])>=0):
						line[0]=line[0]+'_aug'
						s=''
						for ch in line:
							s=s+ch+'\t'
							#print(ch)
						s=s[:-1]
						fn_w.writelines(s)
						#print(s)
					'''
					cnt=cnt+1
					if cnt>10:
						break
					'''