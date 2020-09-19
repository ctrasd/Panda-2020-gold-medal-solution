import os
import cv2
import skimage.io
#from tqdm.notebook import tqdm
import zipfile
import numpy as np
import skimage
import random
from tqdm import tqdm

tile_size = 256
image_size = 256
n_tiles = 36


def get_tiles(img, mode=0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], 'constant',constant_values=255)
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    )

    img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
    n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(img3,[[0,n_tiles-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img':img3[i], 'idx':i})
    return result, n_tiles_with_info >= n_tiles

'''
img=skimage.io.MultiImage('./train_images/50f2942a4d731666067eab3d0bf0011a.tiff',conserve_memory=False)[1]
print(img.shape)
result_img,flag=get_tiles(img)
print(type(result_img),result_img[0]['img'].shape)
cv2.imwrite('result.png',result_img[0]['img'])
'''


dir='./train_images/'

tif_list=os.listdir(dir)
for tif_name in tqdm(tif_list):
    if tif_name[-4:]!='tiff':
        continue
    img_name=tif_name[:-5]
    cnt=0
    img=skimage.io.MultiImage(dir+tif_name)[1]
    '''
    tiles,flag=get_tiles(img)
    idxes = list(range(n_tiles))
    n_row_tiles = int(np.sqrt(n_tiles))
    images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w
            if len(tiles) > idxes[i]:
                this_img = tiles[idxes[i]]['img']
            else:
                this_img = np.ones((image_size, image_size, 3)).astype(np.uint8) * 255
            this_img = 255 - this_img
            h1 = h * image_size
            w1 = w * image_size
            se=random.random()
            if se<=0.5:
                this_img=cv2.flip(this_img,0)
            se2=random.random()
            if se2<=0.5:
                this_img=cv2.flip(this_img,1)
            se3=random.random()
            if se3<=0.5:
                this_img=np.transpose(this_img,(1,0,2))

            images[h1:h1+image_size, w1:w1+image_size] = this_img

    images = 255 - images
    #images = images.astype(np.float32)
    #images /= 255
    #images = images.transpose(2, 0, 1)
    cv2.imwrite('./train_images_png_256/'+img_name+'.png',images)
    '''
    
    
    tiles2,flag2=get_tiles(img,2)
    idxes = list(range(n_tiles))
    n_row_tiles = int(np.sqrt(n_tiles))
    images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w
            if len(tiles2) > idxes[i]:
                this_img = tiles2[idxes[i]]['img']
            else:
                this_img = np.ones((image_size, image_size, 3)).astype(np.uint8) * 255
            this_img = 255 - this_img
            h1 = h * image_size
            w1 = w * image_size
            se=random.random()
            if se<=0.5:
                this_img=cv2.flip(this_img,0)
            se2=random.random()
            if se2<=0.5:
                this_img=cv2.flip(this_img,1)
            se3=random.random()
            if se3<=0.5:
                this_img=np.transpose(this_img,(1,0,2))

            images[h1:h1+image_size, w1:w1+image_size] = this_img

    images = 255 - images
    #images = images.astype(np.float32)
    #images /= 255
    #images = images.transpose(2, 0, 1)
    cv2.imwrite('./train_images_png_256/'+img_name+'_aug.png',images)