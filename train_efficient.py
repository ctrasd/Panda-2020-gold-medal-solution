from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random
from PIL import Image
import tqdm
import cv2

import torchvision as tv
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from efficientnet_pytorch import EfficientNet
from warmup_scheduler import GradualWarmupScheduler
from evaluation import *
from models.inception_v4 import *
#from models.resnet import *
from models.densenet import *
parser = argparse.ArgumentParser(description='Don\'t worry, be happy')
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int,
                    help="test batch size")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--fold', default='4', choices=['0','1','2','3','4'])
parser.add_argument('--size', default='256', choices=['154','140','256','192','combine'])
parser.add_argument('--max-epoch', default=50, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--save-dir', type=str, default='/data/ctr/tangwang_new/save_dir/')
parser.add_argument('--model', default='dense',choices=['dense','inception_resnet','efficientnet-b0','efficientnet-b1','efficientnet-b2',
    'efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7','efficientnet-b8','resnext50'])


parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=100, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.3, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')
args = parser.parse_args()
def cv_imread(file_path):
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img 

from torch.nn.parameter import Parameter

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

size2tile={'256':36,'192':64,'170':81,'154':100,'140':121}
cnt_t=0
#image_size=int(args.size)
def get_aug_img(this_img):
    se=random.random()
    if se<=0.7:
        this_img=cv2.flip(this_img,0)
    se2=random.random()
    if se2<=0.7:
        this_img=cv2.flip(this_img,1)
    se3=random.random()
    if se3<=0.7:
        this_img=np.transpose(this_img,(1,0,2))
    return this_img

class panda_dataset_random(Dataset):
    """docstring for data"""
    def __init__(self, txt_path,size='256',transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[2])))
        self.imgs = imgs 
        self.transform = transform
        self.size=size
    def __getitem__(self, index):
        fn, label1 = self.imgs[index]
        label=torch.tensor([0]*6).float()
        label[0:label1+1]=1
        label=label*0.9+0.05
        label[0]=1
        dir='./train_images_png_'+self.size+'/'
        if label1>=2:
            img=cv2.imread(dir+fn+'.png')
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            se=random.random()
            if se<=0.5:
                img=cv2.imread(dir+fn+'.png')
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                #img = Image.open('./train_images_png/'+fn+'.png').convert('RGB') 
            else:
                img=cv2.imread(dir+fn+'_aug.png')
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                #img = Image.open('./train_images_png/'+fn+'_aug.png').convert('RGB') 
        tiles=size2tile[self.size]
        flip_idx=np.random.choice(list(range(tiles)), 4, replace=False)
        for id in flip_idx:
            x=id%int(np.sqrt(tiles))
            y=id//int(np.sqrt(tiles))
            h1 = x * image_size
            w1 = y * image_size
            flip_img=img[h1:h1+image_size, w1:w1+image_size]
            flip_img=get_aug_img(flip_img)
            img[h1:h1+image_size, w1:w1+image_size]=flip_img
        #img= Image.open(fn).convert('L')
        #img = Image.open(fn).convert('RGB')
        #img=cv2.imread(fn)
        #img=load_ben_color(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (512, 512))
        #img=load_ben_yuan(img)

        cv2.imwrite('fanfan.jpg',img)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        #img = img.unsqueeze(0) 
        
        return fn,img, label,label1
    def __len__(self):
        return len(self.imgs)


class panda_dataset(Dataset):
    """docstring for data"""
    def __init__(self, txt_path,size='256',transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[2])))
        self.imgs = imgs 
        self.transform = transform
        self.size=size
    def __getitem__(self, index):
        fn, label1 = self.imgs[index]
        label=torch.tensor([0]*6).float()
        label[0:label1+1]=1
        label=label*0.9+0.05
        label[0]=1
        dir='./train_images_png_'+self.size+'/'
        if label1>=2:
            img=cv2.imread(dir+fn+'.png')
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            se=random.random()
            if se<=0.5:
                img=cv2.imread(dir+fn+'.png')
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                #img = Image.open('./train_images_png/'+fn+'.png').convert('RGB') 
            else:
                img=cv2.imread(dir+fn+'_aug.png')
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                #img = Image.open('./train_images_png/'+fn+'_aug.png').convert('RGB') 
        #img= Image.open(fn).convert('L')
        #img = Image.open(fn).convert('RGB')
        #img=cv2.imread(fn)
        #img=load_ben_color(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (512, 512))
        #img=load_ben_yuan(img)

        #cv2.imwrite('./save_test/test_'+str(index)+'.jpg',img)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        #img = img.unsqueeze(0) 
        
        return fn,img, label,label1
    def __len__(self):
        return len(self.imgs)

class panda_dataset_nob(Dataset):
    """docstring for data"""
    def __init__(self, txt_path,size='256',transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1], int(words[2])))
        self.imgs = imgs 
        self.transform = transform
        self.size=size
    def __getitem__(self, index):
        fn, comp,label1 = self.imgs[index]
        label=torch.tensor([0]*6).float()
        label[0:label1+1]=1
        label=label*0.9+0.05
        label[0]=1
        dir='./train_images_png_'+self.size+'/'
        se=random.random()
        if se<=0.5:
            img=cv2.imread(dir+fn+'.png')
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #img = Image.open('./train_images_png/'+fn+'.png').convert('RGB') 
        else:
            img=cv2.imread(dir+fn+'_aug.png')
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #img = Image.open('./train_images_png/'+fn+'_aug.png').convert('RGB') 
        '''
        if comp[0]=='k':
            se=random.random()
            if se<=0.5:
                img=cv2.imread(dir+fn+'.png')
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                #img = Image.open('./train_images_png/'+fn+'.png').convert('RGB') 
            else:
                img=cv2.imread(dir+fn+'_aug.png')
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                #img = Image.open('./train_images_png/'+fn+'_aug.png').convert('RGB') 
        else:
            img=cv2.imread(dir+fn+'.png')
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
       '''
        #img=cv2.imread(dir+fn+'.png')
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #img= Image.open(fn).convert('L')
        #img = Image.open(fn).convert('RGB')
        #img=cv2.imread(fn)
        #img=load_ben_color(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (512, 512))
        #img=load_ben_yuan(img)

        #cv2.imwrite('./save_test/test_'+str(index)+'.jpg',img)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        #img = img.unsqueeze(0) 
        
        return fn,img, label,label1
    def __len__(self):
        return len(self.imgs)    

    
class panda_dataset_nob_all(Dataset):
    """docstring for data"""
    def __init__(self, txt_path,size='256',transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1], int(words[2])))
        self.imgs = imgs 
        self.transform = transform
        self.size=size
    def __getitem__(self, index):
        fn, comp,label1 = self.imgs[index]
        label=torch.tensor([0]*6).float()
        label[0:label1+1]=1
        label=label*0.9+0.05
        label[0]=1
        dir='./train_images_png_'+self.size+'/'
        img=cv2.imread(dir+fn+'.png')
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #img = Image.open('./train_images_png/'+fn+'.png').convert('RGB') 


        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        #img = img.unsqueeze(0) 
        
        return fn,img, label,label1
    def __len__(self):
        return len(self.imgs)    
    

class panda_dataset_test(Dataset):
    """docstring for data"""
    def __init__(self, txt_path,size='256',transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1],int(words[2])))
        self.imgs = imgs 
        self.transform = transform
        self.size=size
    def __getitem__(self, index):
        fn, comp,label1 = self.imgs[index]
        label=torch.tensor([0]*6).float()
        label[0:label1+1]=1
        label=label*0.9+0.05
        label[0]=1
        img = Image.open('./train_images_png_'+self.size+'/'+fn+'.png').convert('RGB') 
        #img= Image.open(fn).convert('L')
        #img = Image.open(fn).convert('RGB')
        #img=cv2.imread(fn)
        #img=load_ben_color(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (512, 512))
        #img=load_ben_yuan(img)

        #cv2.imwrite('./save_test/test_'+str(index)+'.jpg',img)
        #img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        #img = img.unsqueeze(0) 
        
        return fn,img, label,label1,comp
    def __len__(self):
        return len(self.imgs)


def get_preds(arr):
    mask = arr == 0
    return np.clip(np.where(mask.any(1), mask.argmax(1), 6) - 1, 0, 5)


warmup_factor = 10
warmup_epo=1
if __name__ == '__main__':
    max_loss=100000000
    torch.set_num_threads(5) 
    runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-')
    args.save_dir = os.path.join(args.save_dir, runId)
    writer=SummaryWriter(comment=args.size+'_'+args.model+'_00001_adam_bce_fold_'+args.fold+'_1152_naugt_nbal')
    #if not os.path.exists(args.save_dir):
        #os.mkdir(args.save_dir)
    print("==========\nArgs:{}\n==========".format(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        #torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")


    transform_dense = transforms.Compose([
            #transforms.CenterCrop(512)
            #transforms.RandomResizedCrop(512),
            #transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05,contrast=0.2,saturation=0.1,hue=0.1),
            transforms.Resize((1344, 1344)),
            transforms.ToTensor(), # 转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ]) 
    transform_eff = transforms.Compose([
            #transforms.CenterCrop(512)
            #transforms.RandomResizedCrop(456),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(360),
            transforms.ColorJitter(brightness=0.05,contrast=0.1,saturation=0.1,hue=0.1),
            transforms.Resize((1152, 1152)),
            transforms.ToTensor(), # 转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             ])
    transform_resnext = transforms.Compose([
            #transforms.CenterCrop(512)
            #transforms.RandomResizedCrop(456),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(360),
            #transforms.ColorJitter(brightness=0.1,contrast=0.05,saturation=0.05,hue=0.05),
            #transforms.Resize((1024, 1024)),
            transforms.ToTensor(), # 转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             ])
    transform_test = transforms.Compose([
            #transforms.CenterCrop(512)
            #transforms.RandomResizedCrop(512),
            #transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),
            transforms.Resize((1152, 1152)),
            transforms.ToTensor(), # 转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             ])
    
    if args.model=='dense':
        transform2=transform_dense
        transform_test = transforms.Compose([
            #transforms.CenterCrop(512)
            #transforms.RandomResizedCrop(512),
            #transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),
            transforms.Resize((1248, 1248)),
            transforms.ToTensor(), # 转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             ])
    elif args.model=='resnext50':
        transform2=transform_resnext
        transform_test = transforms.Compose([
            #transforms.CenterCrop(512)
            #transforms.RandomResizedCrop(512),
            #transforms.ColorJitter(brightness=20,contrast=0.2,saturation=20,hue=0.1),
            #transforms.Resize((1024, 1024)),
            transforms.ToTensor(), # 转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             ])
        
    else:    
        transform2=transform_eff

    train_data=panda_dataset_nob('./split/train_fold_'+args.fold+'.txt',args.size,transform2)
    test_data=panda_dataset_test('./split/fold_'+args.fold+'.txt',args.size,transform_test)
    if args.model=='dense':
        net=Baseline_single(num_classes=6)
        #net.load_state_dict(torch.load('./save_models/model_yuan456_'+args.model+'_00001_adam_combine_'+args.dataset+'_bce_maxest.pkl'))
    elif args.model=='resnext50':
        net=torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        net.avgpool=GeM(p=3)
        net.fc=nn.Linear(in_features=2048,out_features=6,bias=True)
    else:
        net = EfficientNet.from_pretrained(args.model)
        #net = EfficientNet.from_name('efficientnet-b0')
        feature = net._fc.in_features
        net._fc = nn.Linear(in_features=feature,out_features=6,bias=True)
        net._avg_pooling=GeM(p=3)
        
    if use_gpu:
        net= nn.DataParallel(net).cuda()
    #net.load_state_dict(torch.load('./save_models/model_256_resnext50_0.00032_adam_4_bce_newest_gem_1024_nob.pkl'))
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.SmoothL1Loss()
    criterion=nn.BCEWithLogitsLoss()
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=args.lr)
    #scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[10,20,30,40,50,60], gamma=0.7, last_epoch=-1)
    #print(epoch, 'lr={:.6f}'.format(scheduler.get_lr()[0]))
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)
    #scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=9, verbose=True, eps=1e-8,min_lr=5e-6)
    #scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine) 
    start_epoch = args.start_epoch
    dataloader = DataLoader(
        train_data,batch_size= args.train_batch, shuffle = True, num_workers= 8)
    dataloader_test=DataLoader(
        test_data,batch_size= args.test_batch, shuffle = False, num_workers= 8)
    optimizer.zero_grad()
    idx=0
    max_correct=0
    max_kar_kappa=0
    for epoch in range(start_epoch,args.max_epoch):
        print('lr:',optimizer.state_dict()['param_groups'][0]['lr'])
        writer.add_scalar('scalar/learning_rate',optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        #print(epoch, 'lr={:.7f}'.format(scheduler.get_lr()[0]))
        #print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        #if epoch==1:
        #   for param_group in optimizer.param_groups:
        #       param_group['lr']=param_group['lr']*0.1
        #       print('lr',param_group['lr'])
        total_loss=0
        total_correct=0
        total=0
        net.train()
        feature_show_flag=0
        for id,item in tqdm(enumerate(dataloader)):
            #print(id)
            idx=idx+1
            fn,data,label,label_num=item
            #print(fn,label_num)
            if use_gpu:
                data=data.cuda()
                label=label.float().cuda()
                label_num=label_num.long().cuda()

            else:
                #label=label.float()
                label_num=label_num.long()
            print(data.size())
            
            #out,aux=net(data)
            out=net(data)
            #print(feat_map[0,9,:,:].shape)
            #out=net(data)
            #print(out,type(out))
            #print(type(out))
            #print('')
            #print(out)
            #print(label)

            #_, predicted = torch.max(out, 1)
            predicted=get_preds((torch.sigmoid(out) > 0.5).cpu().numpy())
            total += len(label_num)
            #print('')
            #print(predicted)
            #print(label_num.cpu().numpy())
            correct = (predicted == label_num.cpu().numpy()).sum()
            total_correct=total_correct+correct
            loss=criterion(out,label)
            total_loss=total_loss+loss.item()
            #print(torch.sigmoid(out),'\n',label)
            loss.backward()
            #if id%3==0:
            optimizer.step()
            optimizer.zero_grad()
            print('id:%d loss:%f correct:%d'%(id,loss.item(),correct))
            writer.add_scalar('scalar/running_loss',loss.item(), idx)
        print("loss:%f"%(total_loss))
        writer.add_scalar('scalar/running_correct',total_correct.astype('float32')/total, epoch)
        #f = open(args.save_dir+'/train_log.txt','a')
        #f.writelines([str(id),str(total_loss)])
        #f.close()
        feature_show_flag=0
        if epoch%1==0:
            ratera=np.array([])
            raterb=np.array([])
            ratera_kar=np.array([])
            raterb_kar=np.array([])
            ratera_red=np.array([])
            raterb_red=np.array([])
            with torch.no_grad():
                net.eval()
                total=0
                total_loss=0
                correct=0
                for id,item in tqdm(enumerate(dataloader_test)):
                    fn,data,label,label_num,comp=item
                    if use_gpu:
                        data=data.cuda()
                        label=label.float().cuda()
                        label_num=label_num.long().cuda()
                    else:
                        label=label.float()
                        label_num=label_num.long()
                    #print(data.size())
                    
                    #out,aux=net(data)
                    out=net(data)
                    '''
                    if feature_show_flag<=5:
                        for i in range(3):
                            writer.add_image(fn[0].split('/')[-1]+'_'+str(epoch)+'_'+str(label_num),feat_map[0,i,:,:].reshape((-1,feat_map[0,i,:,:].shape[0],feat_map[0,i,:,:].shape[1])))
                        feature_show_flag+=1                            
                    '''
                    #out=net(data)
                    #_, predicted = torch.max(out, 1)
                    predicted=get_preds((torch.sigmoid(out) > 0.5).cpu().numpy())
                    total += label.size(0)
                    correct += (predicted == label_num.cpu().numpy()).sum()
                    loss1=criterion(out, label)
                    total_loss+=loss1.item()
                    #predicted=predicted.cpu().numpy()
                    #label_num=predicted.cpu().numpy()
                    #print('comp:',comp[0])
                    if comp[0][0]=='r':
                        ratera_red=np.hstack((ratera_red,predicted))
                        raterb_red=np.hstack((raterb_red,label_num.cpu().numpy()))
                    else:
                        ratera_kar=np.hstack((ratera_kar,predicted))
                        raterb_kar=np.hstack((raterb_kar,label_num.cpu().numpy()))

                    ratera=np.hstack((ratera,predicted))
                    raterb=np.hstack((raterb,label_num.cpu().numpy()))
                print(ratera,raterb)
                kappa=quadratic_weighted_kappa(ratera,raterb)
                kappa_red=quadratic_weighted_kappa(ratera_red,raterb_red)
                kappa_kar=quadratic_weighted_kappa(ratera_kar,raterb_kar)
                #np.save('./save_predict/'+str(epoch)+'.npy',ratera)
                #np.save('./save_predict/gt.npy',raterb)
                print('测试准确率为： %.2f %%  kappa为： %.4f'%((100*correct.astype('float32')/total),kappa))
                print('red_kappa为： %.4f, kar_kappa为： %.4f'%(kappa_red,kappa_kar))
                #print('red数目: %f,kar数目: %f' %(ratera_red.shape[0],ratera_kar.shape[0]))
                writer.add_scalar('scalar/test_correct',correct.astype('float32')/total, epoch)
                writer.add_scalar('scalar/test_loss',total_loss/total, epoch)
                writer.add_scalar('scalar/test_kappa',kappa, epoch)
                writer.add_scalar('scalar/test_kappa_red',kappa_red, epoch)
                writer.add_scalar('scalar/test_kappa_kar',kappa_kar, epoch)
        scheduler_cosine.step(epoch)
        #scheduler.step(total_loss)
        state_dict = net.state_dict()
        torch.save(state_dict, './save_models/model_'+args.size+'_'+args.model+'_'+str(args.lr)+'_adam_'+args.fold+'_bce_newest_gem_1152_naugt_nbal.pkl')
        if kappa>max_correct:
            max_correct=kappa
            print('saved kappa:',max_correct)
            state_dict = net.state_dict()
            torch.save(state_dict, './save_models/model_'+args.size+'_'+args.model+'_'+str(args.lr)+'_adam_'+args.fold+'_bce_maxest_gem_1152_naugt_nbal.pkl')
        if total_loss/total<max_loss:
            max_loss=total_loss/total
            print('saved min_loss:',max_loss)
            state_dict = net.state_dict()
            torch.save(state_dict, './save_models/model_'+args.size+'_'+args.model+'_'+str(args.lr)+'_adam_'+args.fold+'_bce_min_loss_gem_1152_naugt_nbal.pkl')
        if kappa_kar>max_kar_kappa:
            max_kar_kappa=kappa_kar
            state_dict = net.state_dict()
            torch.save(state_dict, './save_models/model_'+args.size+'_'+args.model+'_'+str(args.lr)+'_adam_'+args.fold+'_bce_karmax_gem_1152_naugt_nbal.pkl')
