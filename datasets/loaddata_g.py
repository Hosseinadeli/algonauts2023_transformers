# loaddata_g.py


from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch

import glob
from sklearn.utils import shuffle
from PIL import Image, ImageOps
from tqdm import tqdm
import random
import cv2

import pandas as pd
import skimage.io as io

import torch
from torchvision.ops import nms

import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

import datasets.transforms as T

from skimage.draw import disk
import copy


        

def plot_dots(im, coords):
    
    h, w, _ = im.shape
    
    img = im.copy()

    first_dot_x = int(coords[0][0])  # * w 
    first_dot_y = int(coords[0][1])   # * h 

    second_dot_x = int(coords[1][0])  # * w 
    second_dot_y = int(coords[1][1]) # * h

    rr, cc = disk((first_dot_y, first_dot_x), 14, shape=None)
    in_bound = [(rr>-1) & (rr<h) & (cc>-1) & (cc<w)]
    cc = cc[in_bound]
    rr = rr[in_bound]
    img[rr, cc] = [255, 0, 0]

    rr, cc = disk((first_dot_y, first_dot_x), 7, shape=None)
    in_bound = [(rr>-1) & (rr<h) & (cc>-1) & (cc<w)]
    cc = cc[in_bound]
    rr = rr[in_bound]
    img[rr, cc] = [255, 255, 255]

    rr, cc = disk((second_dot_y, second_dot_x), 14, shape=None)
    in_bound = [(rr>-1) & (rr<h) & (cc>-1) & (cc<w)]
    cc = cc[in_bound]
    rr = rr[in_bound]
    img[rr, cc] = [255, 0, 0]

    rr, cc = disk((second_dot_y, second_dot_x), 7, shape=None)
    in_bound = [(rr>-1) & (rr<h) & (cc>-1) & (cc<w)]
    cc = cc[in_bound]
    rr = rr[in_bound]
    img[rr, cc] = [255, 255, 255]

    return img
    #         img_name = '{}{}_{}_{}_{}.png'.format(img_dir, image_id, img_id, same_different[same_diff], close_far[k])
    #         io.imsave(img_name, img)

    # print('the dots are on (the) {} object(s)'.format(same_different[1-y['labels'][0][0]]))

    # fig = plt.figure(figsize=(8,10)) #  (figsize=figsize)
    # plt.imshow(img); plt.axis('off')
    # plt.show()
    
def load_image(df, dataType, img_dir):
    
    label = int(df['same_diff'] == 'same')
    
    first_dot_xy = np.fromstring(df['first_dot_xy'][1:-1], dtype=float, sep=',')
    second_dot_xy = np.fromstring(df['second_dot_xy'][1:-1], dtype=float, sep=',')

    path = '%s/%s/%s'%(img_dir,dataType,df['img_name'])
    #img = cv2.imread(path)
    img = Image.open(path).convert('RGB')
    w, h = img.size 
    
    dots_coords = np.stack((first_dot_xy,second_dot_xy))
    #dots_coords[:,[1, 0]] =  dots_coords[:,[0, 1]]  # swapping x and y positions to match image size format
    
    #img = Image.fromarray(plot_dots(np.asarray(img), dots_coords))
    
    
    dots_coords[:,0] = dots_coords[:,0]/w
    dots_coords[:,1] = dots_coords[:,1]/h
        
    target = {'labels':torch.tensor([label])}
    dots_coords = {'coords':torch.tensor(dots_coords)}    
    
    return img, dots_coords, target
    
    
class grouping_dataset(Dataset):
    def __init__(self, args, is_train='train', transforms=None):
        super(grouping_dataset, self).__init__()
        self.is_train = is_train
        
        if self.is_train in ['train', 'val']: 
        
            df = pd.read_excel( '{}{}'.format(args.dataset_grouping_dir, 'train_data_grouping.xls'), index_col=0)  
            self.df_train = df[:26500] #.sample(frac=1)   # :25000
            self.df_val = df[26500:] #.sample(frac=1) # 28000:     
            self.dataType='train2017'
            
        elif self.is_train == 'test': 
            self.df_test = pd.read_excel( '{}{}'.format(args.dataset_grouping_dir, 'test_data_grouping.xls'), index_col=0)  
            self.dataType='val2017'   
            
            
        self.transforms = transforms
        self.img_dir = args.coco_path + '/images'
        
        if self.is_train == 'train':
            self.length = len(self.df_train)
            
        elif self.is_train == 'val':
            self.length = len(self.df_val)
            
        elif self.is_train == 'test':
            self.length = len(self.df_test)

    def __getitem__(self, idx):
        
        if self.is_train == 'train':
            df = self.df_train.iloc[idx]
            img, dots_coords, target = load_image(df, self.dataType, self.img_dir)
            
        elif self.is_train == 'val':
            df = self.df_val.iloc[idx]
            img, dots_coords, target = load_image(df, self.dataType, self.img_dir)
    
        elif self.is_train == 'test':
            df = self.df_test.iloc[idx]
            img, dots_coords, target = load_image(df, self.dataType, self.img_dir)
            
#             ind = self.test_ids[idx]
#             img, dots_coords, target = load_image(self.image_paths[ind], dataType, self.img_dir)

        img, _ = self.transforms(img, None)
        return img, dots_coords, target
    
    
    def __len__(self):
        return self.length
        
        

def make_coco_transforms():

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])

    raise ValueError(f'unknown {image_set}')

def fetch_dataloader(args, batch_size, train='train', shuffle=True, train_val_split='none', download=True):
    """
    load dataset depending on the task
    currently implemented tasks:
        -svhn
        -cifar10
        -mnist
        -multimnist, multimnist_cluttered 
    args
        -args
        -batch size
        -train: if True, load train dataset, else test dataset
        -train_val_split: 
            'none', load entire train dataset
            'train', load first 90% as train dataset
            'val', load last 10% as val dataset
            'train-val', load 90% train, 10% val dataset
    """
    kwargs = {'num_workers': 0, 'pin_memory': False} if torch.cuda.is_available() else {}
    
    
#     img_folder = '../data/svrt_dataset/a128_results_problem_1'   # svrt_task1_64x64' #a128_results_problem_1'
    transforms = T.Compose([T.ToTensor()])

    svrt_data = grouping_dataset(args, is_train=train, transforms=make_coco_transforms())
    dataloader = torch.utils.data.DataLoader(dataset=svrt_data, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)        

    return dataloader 
