import cv2
#import skimage.io
#from tqdm.notebook import tqdm
#import zipfile
#import skimage
from tqdm import tqdm
import os
import sys
import time
#import datetime
#import argparse
import os.path as osp
import numpy as np
import csv
import random


if __name__ == '__main__':
    file_rad='rad_with_mask.txt'
    img_name=[]
    imgs=[]
    with open(file_rad) as fh:
        for line in fh:
            line = line.rstrip()
            words = line.split()
            for i in range(36):
                img_name.append(words[0]+'_'+str(i))
            imgs.append((words[0], int(words[2])))
    data=img_name[1:]
    random.shuffle(data)
    lens=len(data)
    fold_len=lens//5
    data_fold=[]
    for i in range(5):
        data_fold.append(data[i*fold_len:(i+1)*fold_len])
        file_name='radmask_fold_'+str(i)+'.txt'
        with open(file_name,'w') as ffn:
            for j in range(i*fold_len,(i+1)*fold_len):
                print(data[j])
                ffn.writelines(data[j]+'\n')
        file_name='radmask_train_flod_'+str(i)+'.txt'
        with open(file_name,'w') as ffn:
            for j in range(lens):
                if j>= i*fold_len and j<(i+1)*fold_len:
                    continue
                #print(data[j])
                ffn.writelines(data[j]+'\n')
    '''
    file_rad_tile='rad_mask_tile.txt'
    with open(file_rad_tile,'w') as fn:
        for x in img_name:
            fn.writelines(x+'\n')
    '''