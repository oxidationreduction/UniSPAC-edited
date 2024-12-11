import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from skimage.measure import label
import logging
import zarr
from funlib.evaluate import rand_voi

import matplotlib.pyplot as plt
# import pyvoi


import numpy as np
from torch.utils.data import DataLoader
# import imageio

NUM_SAMPLES = 1

def set_seed(seed = 1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True
    
set_seed()

from scipy.stats import multivariate_normal
class Dataset_2D_Zebrafinch_Training():
    def __init__(
        self,
        data_dir='./data/funke/zebrafinch/training/',  # 数据的路径
        data_name = 'gt_z255-383_y1407-1663_x1535-1791.zarr',
    ):

        ###装载FIB-25的训练数据
        self.images = list()
        self.masks = list()
        self.labels = list()  

        zarr_path = data_dir + data_name
        f = zarr.open(zarr_path, mode='r')
        volumes = f['volumes']
        raw = volumes['raw'] #zyx
        labels = volumes['labels']['neuron_ids']  #zyx

        raw_patch = raw[100:(100+labels.shape[0]),200:(200+labels.shape[1]),200:(200+labels.shape[2])]
        labels = label(labels).astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域

        for n in range(len(labels)):
            if np.max(labels[n])==0:
                continue
            # self.images.append(raw_patch[n])
            # self.masks.append(labels[n])

            test_patch = np.array(raw_patch[n])
            pad_size_x = (8-(test_patch.shape[0]%8))%8
            pad_size_y = (8-(test_patch.shape[1]%8))%8
            # print(test_patch.shape)
            test_patch = np.pad(test_patch,((0,pad_size_x),(0,pad_size_y)),'constant',constant_values = (0,0)) 
            self.images.append(test_patch)


            mask_patch = (np.array(labels[n]) != 0 ) + 0
            # print(mask_patch.shape)
            mask_patch = np.pad(mask_patch,((0,pad_size_x),(0,pad_size_y)),'constant',constant_values = (0,0)) 
            self.masks.append(mask_patch)

            self.labels.append(labels[n])

    #展示一下求Affinity
    def getAffinity(self,labels):
        '''
        labels为2维: 长*宽
        Return: Affinity为3维, 2*长*宽,其中2对应了长(x)、宽(y)两个方向上的梯度
        '''
        label_shift = np.pad(labels,((0,1),(0,1)),'edge')


        affinity_x = np.expand_dims(((labels-label_shift[1:,:-1])!=0)+0,axis=0)
        affinity_y = np.expand_dims(((labels-label_shift[:-1,1:])!=0)+0,axis=0)

        affinity = np.concatenate([affinity_x,affinity_y],axis=0).astype('float32')
        
        
        return affinity
                
    def get_prompt(self, labels):
        ##
        Points_pos = list()
        Points_lab = list()
        Boxes = list()
        # mask = np.zeros_like(labels,bool)
        
        ##
        total_unique_labels = np.unique(labels).tolist()
        # point_style = random.choice(['+','-','+-'])
        point_style = '+'
        labels_contain = total_unique_labels
        labels_exclude = list()
                
        ##Get Points(+) and boxes
        for label in labels_contain:
            mask_label = (labels == label)
            
            ##Mask
            # mask = np.logical_or(mask,mask_label)
            
            # idx_label = np.sort(np.where(mask_label))
            idx_label = np.where(mask_label)
            y_list = idx_label[0]
            x_list = idx_label[1]
            
#             ##Point(+)
#             for times in range(NUM_SAMPLES):
#                 # idx = random.choice(np.arange(len(x_list)).tolist())
#                 idx = int(len(x_list)/(NUM_SAMPLES+1)*(times+1))
                
#                 if '+' in point_style:
#                     Points_pos.append([x_list[idx],y_list[idx]])
#                     Points_lab.append(1)
                

            Points_pos.append([int(np.mean(x_list)),int(np.mean(y_list))])
            Points_lab.append(1)
            
            
            ##Box
            # if Get_box:
            box = [np.min(x_list),np.min(y_list),np.max(x_list),np.max(y_list)]
            Boxes.append(box)


        ##Get Points(-) 
        for label in labels_exclude:
            mask_label = (labels == label)
            idx_label = np.sort(np.where(mask_label))
            y_list = idx_label[0]
            x_list = idx_label[1]
            
            # idx = random.choice(np.arange(len(x_list)).tolist())
            idx = int(len(x_list)/2)
            if '-' in point_style:
                Points_pos.append([x_list[idx],y_list[idx]])
                Points_lab.append(0)

            
        # return Points_pos,Points_lab,Boxes,mask
        return Points_pos,Points_lab,Boxes

    def generate_gaussian_matrix(self, Points_pos,Points_lab, H, W, theta=10):
        if Points_pos==None:
            total_matrix = np.ones((H,W))
            return total_matrix
        
        total_matrix = np.zeros((H,W))

        record_list = list()
        for n,(X,Y) in enumerate(Points_pos):
            if (X,Y) not in record_list:
                record_list.append((X,Y))
            else:
                continue

            # 生成坐标网格
            x, y = np.meshgrid(np.arange(W), np.arange(H))
            pos = np.dstack((x, y))

            # 以 (X, Y) 为中心，theta 为标准差生成高斯分布
            rv = multivariate_normal(mean=[X, Y], cov=[[theta, 0], [0, theta]])

            # 计算高斯分布在每个像素上的值
            matrix = rv.pdf(pos)
            ##normalize
            matrix = matrix * (1/np.max(matrix))

            total_matrix = total_matrix + matrix*(Points_lab[n]*2-1)
            # total_matrix = total_matrix + matrix
            
        if np.max(Points_lab) == 0:
            total_matrix = total_matrix * 2 + 1

        return total_matrix

    
    def __len__(self):
        return len(self.images)


    # normalize raw data between 0 and 1
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)

    def __getitem__(self, idx):

        raw = self.normalize(self.images[idx]) #获得第idx张图像
        # raw = self.images[idx] #获得第idx张图像
        mask = self.masks[idx]
        label = self.labels[idx]
        
        Points_pos,Points_lab,Boxes =  self.get_prompt(label)
        
        point_map = self.generate_gaussian_matrix(Points_pos,Points_lab,raw.shape[0],raw.shape[1],theta=30)
        
        raw = np.expand_dims(raw, axis=0) # 1* 图像维度
        
        affinity = self.getAffinity(label)

        return raw, label, mask, point_map, Points_pos,Points_lab, affinity


def collate_fn_2D_hemi_test(batch):
    
    raw = np.array([item[0] for item in batch]).astype(np.float32)

    labels = np.array([item[1] for item in batch]).astype(np.uint64)
    
    mask = np.array([item[2] for item in batch]).astype(np.uint8)
    
    point_map = np.array([item[3] for item in batch]).astype(np.float32)
    
    affinity = np.array([item[6] for item in batch]).astype(np.float32)


    max_Point_num = max([len(item[4]) for item in batch])
    Points_pos = list()
    Points_lab = list()
    for item in batch:
        Points_pos.append(np.pad(item[4],((0,max_Point_num-len(item[4])),(0,0)),'edge'))
        Points_lab.append(np.pad(item[5],((0,max_Point_num-len(item[5]))),'edge') )
    Points_pos = np.array(Points_pos)
    Points_lab = np.array(Points_lab)

    return raw, labels, mask, point_map, Points_pos,Points_lab, affinity
    
# import pyvoi
###   sudo apt install libboost-dev
###   pip install git+https://github.com/funkey/waterz.git
import waterz
import scipy
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed
from tqdm import tqdm
import numpy as np
from scipy.ndimage import label as wz_label

#@title watershed wrappers
def watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = wz_label(maxima)

    print(f"Found {n} fragments")

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances,
        seeds,
        mask=boundary_mask)

    ret = (fragments.astype(np.uint64), n + id_offset)

    return ret

def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        id_offset=0,
        min_seed_distance=10):

    mean_affs = 0.5*(affs[1] + affs[2])

    fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
  
    boundary_mask = mean_affs>0.9*max_affinity_value
    boundary_distances = distance_transform_edt(boundary_mask)

    ret = watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        id_offset=id_offset,
        min_seed_distance=min_seed_distance)

    return ret

#@title segmentation wrapper
def get_segmentation(affinities, threshold):

    fragments = watershed_from_affinities(affinities)[0]
    thresholds = [threshold]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds,
    )

    segmentation = next(generator)

    return segmentation




    
from models.segNeuro import segneuro2d,segEM2d
# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
###创建模型
model = segEM2d()
model_path = './output/checkpoints/segEM2d(hemi+fib25)_Best_in_val.model'
# model_path = './output/checkpoints/segEM2d(hemi+fib25)wloss-1_Best_in_val.model'
# model_path = './output/checkpoints/segEM2d(hemi+fib25)_zebrafinch1-8_joint_Best_in_val.model'

weights = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(weights)

model = model.to(device)


##装载数据
data_list = ['gt_z255-383_y1407-1663_x1535-1791.zarr','gt_z2559-2687_y4991-5247_x4863-5119.zarr',
    'gt_z2815-2943_y5631-5887_x4607-4863.zarr','gt_z2834-2984_y5311-5461_x5077-5227.zarr',
    'gt_z2868-3018_y5744-5894_x5157-5307.zarr','gt_z2874-3024_y5707-5857_x5304-5454.zarr',
    'gt_z2934-3084_y5115-5265_x5140-5290.zarr','gt_z3096-3246_y5954-6104_x5813-5963.zarr',
    'gt_z3118-3268_y6538-6688_x6100-6250.zarr','gt_z3126-3276_y6857-7007_x5694-5844.zarr',
    'gt_z3436-3586_y599-749_x2779-2929.zarr','gt_z3438-3588_y2775-2925_x3476-3626.zarr',
    'gt_z3456-3606_y3188-3338_x4043-4193.zarr','gt_z3492-3642_y7888-8038_x8374-8524.zarr',
    'gt_z3492-3642_y841-991_x381-531.zarr','gt_z3596-3746_y3888-4038_x3661-3811.zarr',
    'gt_z3604-3754_y4101-4251_x3493-3643.zarr','gt_z3608-3758_y3829-3979_x3423-3573.zarr',
    'gt_z3702-3852_y9605-9755_x2244-2394.zarr','gt_z3710-3860_y8691-8841_x2889-3039.zarr',
    'gt_z3722-3872_y4548-4698_x2879-3029.zarr','gt_z3734-3884_y4315-4465_x2209-2359.zarr',
    'gt_z3914-4064_y9035-9185_x2573-2723.zarr','gt_z4102-4252_y6330-6480_x1899-2049.zarr',
    'gt_z4312-4462_y9341-9491_x2419-2569.zarr','gt_z4440-4590_y7294-7444_x2350-2500.zarr',
    'gt_z4801-4951_y10154-10304_x1972-2122.zarr','gt_z4905-5055_y928-1078_x1729-1879.zarr',
    'gt_z4951-5101_y9415-9565_x2272-2422.zarr','gt_z5001-5151_y9426-9576_x2197-2347.zarr',
    'gt_z5119-5247_y1023-1279_x1663-1919.zarr','gt_z5405-5555_y10490-10640_x3406-3556.zarr',
    'gt_z734-884_y9561-9711_x563-713.zarr']


voi_dict = dict()
voi_dict_ACRLSD = dict()
for data_name in data_list:
    voi_dict[data_name] = list()
    voi_dict_ACRLSD[data_name] = list()
    
    print("Start testing {}".format(data_name))
    
    ##装载数据
    test_dataset = Dataset_2D_Zebrafinch_Training(data_name = data_name)
    test_loader = DataLoader(test_dataset,batch_size=1, shuffle=False, collate_fn=collate_fn_2D_hemi_test)
    
    ###################Test###################
    model.eval()
    np.random.seed(100)
    for n,data in enumerate(test_loader):
        raw, labels, mask, point_map, Points_pos,Points_lab, affinity = data

        ##Transfer to tensors
        input_image = torch.as_tensor(raw,dtype=torch.float, device= device) #(batch, 1, height, width)
        point_map = torch.as_tensor(point_map, dtype=torch.float, device=device) #(batch, height, width)
        point_map = torch.ones(point_map.shape).to(device)
        mask = torch.as_tensor(mask, dtype=torch.float, device=device) #(batch, 1, height, width)

        with torch.no_grad():
            y_mask,y_affinity = model(input_image,point_map)

        labels = labels.squeeze()
        affinity = affinity.squeeze()

        input_image = input_image.detach().cpu().numpy().squeeze()
        binary_y_mask = (y_mask>0.5).detach().cpu().numpy().squeeze()
        y_seg = label(binary_y_mask).astype(np.uint64)[:labels.shape[0],:labels.shape[1]]
        y_affinity = y_affinity.detach().cpu().numpy().squeeze()
        
        
        
        ##计算VOI
        rand_voi_report = rand_voi(
                        labels[labels!=0],
                        y_seg[labels!=0],
                        return_cluster_scores=False)

        metrics = rand_voi_report.copy()
        voi_dict[data_name].append([metrics['voi_split'],metrics['voi_merge']])
        
        
        
        ######计算ACRLSD的性能
        print('Start watershed process...')
        # watershed assumes 3d arrays, create fake channel dim (call these watershed affs - ws_affs)
        ws_affs = np.stack([
                np.ones_like(y_affinity[0]),
                1- y_affinity[0],
                1- y_affinity[1]]
            )
        # affs shape: 3, h, w

        # waterz agglomerate requires 4d affs (c, d, h, w) - add fake z dim
        ws_affs = np.expand_dims(ws_affs, axis = 1)

        # higher thresholds will merge more, lower thresholds will split more
        y_seg_ACRLSD = np.array(label(get_segmentation(ws_affs, threshold=0.5)).squeeze(),dtype=np.uint64)
        y_seg_ACRLSD = y_seg_ACRLSD[:labels.shape[0],:labels.shape[1]]
        
        rand_voi_report = rand_voi(
                        labels[labels!=0],
                        y_seg_ACRLSD[labels!=0],
                        return_cluster_scores=False)

        metrics = rand_voi_report.copy()
        voi_dict_ACRLSD[data_name].append([metrics['voi_split'],metrics['voi_merge']])



    print("Mean VOI for SegNeuro-2D = {:.4f}".format(np.mean(np.sum(voi_dict[data_name],axis=1))))
    print("Mean VOI for ACRLSD-2D = {:.4f}".format(np.mean(np.sum(voi_dict_ACRLSD[data_name],axis=1))))



# np.save('./voi_dict_SegNeuro2D(star).npy', voi_dict)
np.save('./voi_dict_SegNeuro2D.npy', voi_dict)
np.save('./voi_dict_ACRLSD-2D.npy', voi_dict_ACRLSD)
