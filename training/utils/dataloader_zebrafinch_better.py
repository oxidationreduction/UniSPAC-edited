import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# import h5py
from skimage.measure import label
import albumentations as A  ##一种数据增强库
from scipy.ndimage import binary_erosion
# import matplotlib.pyplot as plt
import torch
from scipy.stats import multivariate_normal
import zarr
from lsd.train import local_shape_descriptor



class Dataset_2D_zebrafinch_Train_CL(Dataset):
    def __init__(
        self,
        data_dir='./data/funke/zebrafinch/training/',  # 数据的路径
        split='train', # 划分方式
        crop_size=None, #切割尺寸
        padding_size=8,
        require_lsd = False,
        require_xz_yz = False,
        data_idxs = [0,1,2],):


        self.split = split
        self.crop_size = crop_size
        self.padding_size = padding_size
        self.split = split
        self.require_lsd = require_lsd

        ###装载zebrafinch的训练数据
        self.images = list()
        self.masks = list()
        ##装载数据
        data_list = ['gt_z2834-2984_y5311-5461_x5077-5227.zarr',
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
            'gt_z734-884_y9561-9711_x563-713.zarr','gt_z255-383_y1407-1663_x1535-1791.zarr',
            'gt_z2559-2687_y4991-5247_x4863-5119.zarr','gt_z2815-2943_y5631-5887_x4607-4863.zarr',]
        
        data_list = [data_list[idx] for idx in data_idxs]
        
        # ##Debug
        # data_list = ['gt_z255-383_y1407-1663_x1535-1791.zarr','gt_z2559-2687_y4991-5247_x4863-5119.zarr']
        
        
        for data_name in data_list:
            zarr_path = data_dir + data_name
            f = zarr.open(zarr_path, mode='r')
            volumes = f['volumes']
            raw = volumes['raw'] #zyx
            labels = volumes['labels']['neuron_ids']  #zyx
            
            labels = label(labels).astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域
            
            print('data {}: raw shape={}, label shape = {}'.format(data_name, raw.shape,labels.shape))
    
            if split == 'train':
                for n in range(8,len(labels)):
                    if np.max(labels[n])==0:
                        continue
                    self.images.append(raw[100+n,200:(200+labels.shape[1]),200:(200+labels.shape[2])])
                    self.masks.append(labels[n])
            elif split == 'val':
                for n in range(8):
                    if np.max(labels[n])==0:
                        continue
                    self.images.append(raw[100+n,200:(200+labels.shape[1]),200:(200+labels.shape[2])])
                    self.masks.append(labels[n])
            
            if require_xz_yz:
                if split == 'train':
                    for n in range(8,labels.shape[1]):
                        if np.max(labels[:,n,:])==0:
                            continue
                        self.images.append(raw[100:(100+labels.shape[0]),200+n,200:(200+labels.shape[2])])
                        self.masks.append(labels[:,n,:])
                    for n in range(8,labels.shape[2]):
                        if np.max(labels[:,:,n])==0:
                            continue
                        self.images.append(raw[100:(100+labels.shape[0]),200:(200+labels.shape[1]),200+n])
                        self.masks.append(labels[:,:,n])
                elif split == 'val':
                    for n in range(8):
                        if np.max(labels[:,n,:])==0:
                            continue
                        self.images.append(raw[100:(100+labels.shape[0]),200+n,200:(200+labels.shape[2])])
                        self.masks.append(labels[:,n,:])
                        if np.max(labels[:,:,n])==0:
                            continue
                        self.images.append(raw[100:(100+labels.shape[0]),200:(200+labels.shape[1]),200+n])
                        self.masks.append(labels[:,:,n])
        
    def __len__(self):
        return len(self.images)

    # function to erode label boundaries，即腐蚀边界
    def erode(self, labels, iterations, border_value):

        foreground = np.zeros_like(labels, dtype=bool) #和标签维度相同的全False矩阵

        # loop through unique labels
        for label in np.unique(labels): # 遍历标签信息（切割块）中所有的连通域的值

            # skip background
            if label == 0:
              continue

            # mask to label
            label_mask = labels == label # 当前连通域对应的标签（背景为False，当前连通域为True）

            # erode labels
            eroded_mask = binary_erosion(
                  label_mask,
                  iterations=iterations,
                  border_value=border_value) #获得iterations轮腐蚀后的当前连通域的标签

            # get foreground
            foreground = np.logical_or(eroded_mask, foreground) #这个前景相当于所有连通域经过边界腐蚀后得到的前景，前景用True标出

        # and background...
        background = np.logical_not(foreground)

        # set eroded pixels to zero
        labels[background] = 0

        return labels

    # takes care of padding
    def get_padding(self, crop_size, padding_size):
    
        # quotient
        q = int(crop_size / padding_size)
    
        if crop_size % padding_size != 0:
            padding = (padding_size * (q + 1))
        else:
            padding = crop_size
    
        return padding

        # sample augmentations (see https://albumentations.ai/docs/examples/example_kaggle_salt)
    def augment_data(self, raw, mask, padding):
        
        transform = A.Compose([
              A.RandomCrop(
                  width=self.crop_size,
                  height=self.crop_size), #1. 随机切割成 crop_size * crop_size 的尺寸
              A.PadIfNeeded(
                  min_height=padding,
                  min_width=padding,
                  p=1,
                  border_mode=0),  #2. 填充成 padding * padding 的尺寸，border_mode=0似乎是常数填充（参考：https://wenku.csdn.net/answer/b124adffba28441daf7b260623a28d87）
              A.HorizontalFlip(p=0.3), #3. 以0.3的概率进行水平翻转
              A.VerticalFlip(p=0.3),   #4. 以0.3的概率进行垂直翻转
              A.RandomRotate90(p=0.3), #5. 以0.3的概率进行垂随机旋转90度
              A.Transpose(p=0.3),      #6. 以0.3的概率进行转置
              A.RandomBrightnessContrast(p=0.3) # 7. 以0.3的概率随机改变输入图像的亮度和对比度。
            ],
            is_check_shapes=False)  ## 数据增强

        # print("raw shape = {}".format(raw.shape))
        # print("mask shape = {}".format(mask.shape))
        transformed = transform(image=raw, mask=mask)

        raw, mask = transformed['image'], transformed['mask']

        return raw, mask ## 图像和标签的维度都是： padding * padding

    # normalize raw data between 0 and 1
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)
    
    def get_prompt(self, labels):
        
        ##
        Points_pos = list()
        Points_lab = list()
        Boxes = list()
        mask = np.zeros_like(labels,bool)
        p_default = np.random.rand()
        
        ##
        total_unique_labels = np.unique(labels).tolist()
        point_style = random.choice(['+','-','+-'])
        if p_default<0.5:
            mask = (labels != 0)
            return None, None, None, mask
        else:
            # p_label_contain = random.choice(['1','1','1','2','2','2','3','3','3',0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            p_label_contain = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
        
        ###产生Points
        # if isinstance(p_label_contain,str):
        #     temp_list = deepcopy(total_unique_labels)
        #     temp_list.remove(0)
        #     random.shuffle(temp_list)
        #     labels_contain = total_unique_labels[:int(p_label_contain)]
        #     labels_exclude = total_unique_labels[int(p_label_contain):]
        # else:
        while(1):
            labels_contain = list()
            labels_exclude = list()
            for label in total_unique_labels:
                p_label = np.random.rand()
                if p_label<p_label_contain and label!=0:
                    labels_contain.append(label)
                else:
                    labels_exclude.append(label)
            if len(labels_contain)!= 0:
                break
            
                
        ##Get Points(+) and boxes
        for label in labels_contain:
            mask_label = (labels == label)
            
            ##Mask
            mask = np.logical_or(mask,mask_label)
            
            idx_label = np.where(mask_label)
            y_list = idx_label[0]
            x_list = idx_label[1]
            
            ##Point(+)
            idx = random.choice(np.arange(len(x_list)).tolist())
            # idx = int(len(x_list)/2)
            if '+' in point_style or len(labels_contain)==1:
                Points_pos.append([x_list[idx],y_list[idx]])
                Points_lab.append(1)
            
            ##Box
            # if Get_box:
            box = [np.min(x_list),np.min(y_list),np.max(x_list),np.max(y_list)]
            Boxes.append(box)
            
            
            
        ##Get Points(-) 
        for label in labels_exclude:
            mask_label = (labels == label)
            idx_label = np.where(mask_label)
            y_list = idx_label[0]
            x_list = idx_label[1]
            
            idx = random.choice(np.arange(len(x_list)).tolist())
            # idx = int(len(x_list)/2)
            if '-' in point_style:
                Points_pos.append([x_list[idx],y_list[idx]])
                Points_lab.append(0)

            
        return Points_pos,Points_lab,Boxes,mask
    
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

    def __getitem__(self, idx):

        raw = self.images[idx] #获得第idx张图像
        labels = self.masks[idx] #获得第idx张图像的标签信息
        
        # if self.split == 'val':
        #     seed = 1000
        #     np.random.seed(seed)
        #     random.seed(seed)
            # os.environ['PYTHONHASHSEED'] = str(seed)

        raw = self.normalize(raw) # 所有像素值归一化

        # relabel connected components
        # labels = label(labels).astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域
        # labels = labels.astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域

        padding = self.get_padding(self.crop_size, self.padding_size) # padding_size的整数倍，>= crop_size
        raw, labels = self.augment_data(raw, labels, padding)
        
        raw = np.expand_dims(raw, axis=0) # 1* 图像维度


        # if train/val, generate our gt labels
        ## 非测试数据的话，进行一轮边界腐蚀
        labels = self.erode(
            labels,
            iterations=1, #腐蚀的迭代次数
            border_value=1) # 边界值
        
        affinity = self.getAffinity(labels)

        if self.require_lsd:
            lsds = local_shape_descriptor.get_local_shape_descriptors(
                    segmentation=labels,
                    sigma=(5,)*2,
                    voxel_size=(1,)*2) ##获得lsd标签，维度为：6*图像维度
            
            lsds = lsds.astype(np.float32)    # 6* 图像维度

        Points_pos,Points_lab,Boxes,mask = self.get_prompt(labels)
        
        point_map = self.generate_gaussian_matrix(Points_pos,Points_lab,self.crop_size,self.crop_size,theta=30)
   
        if self.require_lsd:
            # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity,lsds
            return raw, labels,point_map,mask,affinity,lsds
        else:
            # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity
            return raw, labels,point_map,mask,affinity



def collate_fn_2D_zebrafinch_Train(batch):
    
    raw = np.array([item[0] for item in batch]).astype(np.float32)

    labels = np.array([item[1] for item in batch]).astype(np.int32)


#     max_Point_num = max([len(item[2]) for item in batch])
#     max_Box_num = max([len(item[4]) for item in batch])
#     Points_pos = list()
#     Points_lab = list()
#     Boxes = list()
#     for item in batch:
#         Points_pos.append(np.pad(item[2],((0,max_Point_num-len(item[2])),(0,0)),'edge') )
#         Points_lab.append(np.pad(item[3],((0,max_Point_num-len(item[3]))),'edge') )
#         Boxes.append(np.pad(item[4],((0,max_Box_num-len(item[4])),(0,0)),'edge') )
#     Points_pos = np.array(Points_pos)
#     Points_lab = np.array(Points_lab)
#     Boxes = np.array(Boxes)
    
    # point_map = np.array([item[5] for item in batch]).astype(np.float32)
    # mask = np.array([item[6] for item in batch]).astype(np.uint8)
    # affinity = np.array([item[7] for item in batch]).astype(np.float32)
    # lsds = np.array([item[8] for item in batch]).astype(np.float32)
    
    point_map = np.array([item[2] for item in batch]).astype(np.float32)
    mask = np.array([item[3] for item in batch]).astype(np.uint8)
    affinity = np.array([item[4] for item in batch]).astype(np.float32)
    if len(batch[0]) == 6:
        lsds = np.array([item[5] for item in batch]).astype(np.float32)
        return raw, labels,point_map,mask,affinity,lsds
    else:
        return raw, labels,point_map,mask,affinity
    # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity,lsds





class Dataset_2D_zebrafinch_Train_CL_avalanche(Dataset):
    def __init__(
        self,
        data_dir='./data/funke/zebrafinch/training/',  # 数据的路径
        split='train', # 划分方式
        crop_size=None, #切割尺寸
        padding_size=8,
        require_lsd = False,
        require_xz_yz = False,
        data_idxs = [0,1,2],):


        self.split = split
        self.crop_size = crop_size
        self.padding_size = padding_size
        self.split = split
        self.require_lsd = require_lsd

        ###装载zebrafinch的训练数据
        self.images = list()
        self.masks = list()
        self.targets = list()
        ##装载数据
        data_list = ['gt_z2834-2984_y5311-5461_x5077-5227.zarr',
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
            'gt_z734-884_y9561-9711_x563-713.zarr','gt_z255-383_y1407-1663_x1535-1791.zarr',
            'gt_z2559-2687_y4991-5247_x4863-5119.zarr','gt_z2815-2943_y5631-5887_x4607-4863.zarr',]
        
        data_list = [data_list[idx] for idx in data_idxs]
        
        # ##Debug
        # data_list = ['gt_z255-383_y1407-1663_x1535-1791.zarr','gt_z2559-2687_y4991-5247_x4863-5119.zarr']
        
        
        for idx,data_name in enumerate(data_list):
            zarr_path = data_dir + data_name
            f = zarr.open(zarr_path, mode='r')
            volumes = f['volumes']
            raw = volumes['raw'] #zyx
            labels = volumes['labels']['neuron_ids']  #zyx
            
            labels = label(labels).astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域
            
            print('data {}: raw shape={}, label shape = {}'.format(data_name, raw.shape,labels.shape))

            if split == 'train':
                for n in range(8,len(labels)):
                    if np.max(labels[n])==0:
                        continue
                    self.images.append(raw[100+n,200:(200+labels.shape[1]),200:(200+labels.shape[2])])
                    self.masks.append(labels[n])
                    self.targets.append(idx)
            elif split == 'val':
                for n in range(8):
                    if np.max(labels[n])==0:
                        continue
                    self.images.append(raw[100+n,200:(200+labels.shape[1]),200:(200+labels.shape[2])])
                    self.masks.append(labels[n])
                    self.targets.append(idx)
            
            if require_xz_yz:
                if split == 'train':
                    for n in range(8,labels.shape[1]):
                        if np.max(labels[:,n,:])==0:
                            continue
                        self.images.append(raw[100:(100+labels.shape[0]),200+n,200:(200+labels.shape[2])])
                        self.masks.append(labels[:,n,:])
                        self.targets.append(idx)
                    for n in range(8,labels.shape[2]):
                        if np.max(labels[:,:,n])==0:
                            continue
                        self.images.append(raw[100:(100+labels.shape[0]),200:(200+labels.shape[1]),200+n])
                        self.masks.append(labels[:,:,n])
                        self.targets.append(idx)
                elif split == 'val':
                    for n in range(8):
                        if np.max(labels[:,n,:])==0:
                            continue
                        self.images.append(raw[100:(100+labels.shape[0]),200+n,200:(200+labels.shape[2])])
                        self.masks.append(labels[:,n,:])
                        self.targets.append(idx)
                        if np.max(labels[:,:,n])==0:
                            continue
                        self.images.append(raw[100:(100+labels.shape[0]),200:(200+labels.shape[1]),200+n])
                        self.masks.append(labels[:,:,n])
                        self.targets.append(idx)
        
        # ##适配avalanche
        # self.targets = [1 for i in range(len(self.images))]  # 标签交替赋值为1
        
    def __len__(self):
        return len(self.images)

    # function to erode label boundaries，即腐蚀边界
    def erode(self, labels, iterations, border_value):

        foreground = np.zeros_like(labels, dtype=bool) #和标签维度相同的全False矩阵

        # loop through unique labels
        for label in np.unique(labels): # 遍历标签信息（切割块）中所有的连通域的值

            # skip background
            if label == 0:
              continue

            # mask to label
            label_mask = labels == label # 当前连通域对应的标签（背景为False，当前连通域为True）

            # erode labels
            eroded_mask = binary_erosion(
                  label_mask,
                  iterations=iterations,
                  border_value=border_value) #获得iterations轮腐蚀后的当前连通域的标签

            # get foreground
            foreground = np.logical_or(eroded_mask, foreground) #这个前景相当于所有连通域经过边界腐蚀后得到的前景，前景用True标出

        # and background...
        background = np.logical_not(foreground)

        # set eroded pixels to zero
        labels[background] = 0

        return labels

    # takes care of padding
    def get_padding(self, crop_size, padding_size):
    
        # quotient
        q = int(crop_size / padding_size)
    
        if crop_size % padding_size != 0:
            padding = (padding_size * (q + 1))
        else:
            padding = crop_size
    
        return padding

        # sample augmentations (see https://albumentations.ai/docs/examples/example_kaggle_salt)
    def augment_data(self, raw, mask, padding):
        
        transform = A.Compose([
              A.RandomCrop(
                  width=self.crop_size,
                  height=self.crop_size), #1. 随机切割成 crop_size * crop_size 的尺寸
              A.PadIfNeeded(
                  min_height=padding,
                  min_width=padding,
                  p=1,
                  border_mode=0),  #2. 填充成 padding * padding 的尺寸，border_mode=0似乎是常数填充（参考：https://wenku.csdn.net/answer/b124adffba28441daf7b260623a28d87）
              A.HorizontalFlip(p=0.3), #3. 以0.3的概率进行水平翻转
              A.VerticalFlip(p=0.3),   #4. 以0.3的概率进行垂直翻转
              A.RandomRotate90(p=0.3), #5. 以0.3的概率进行垂随机旋转90度
              A.Transpose(p=0.3),      #6. 以0.3的概率进行转置
              A.RandomBrightnessContrast(p=0.3) # 7. 以0.3的概率随机改变输入图像的亮度和对比度。
            ],
            is_check_shapes=False)  ## 数据增强

        # print("raw shape = {}".format(raw.shape))
        # print("mask shape = {}".format(mask.shape))
        transformed = transform(image=raw, mask=mask)

        raw, mask = transformed['image'], transformed['mask']

        return raw, mask ## 图像和标签的维度都是： padding * padding

    # normalize raw data between 0 and 1
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)
    
    def get_prompt(self, labels):
        
        ##
        Points_pos = list()
        Points_lab = list()
        Boxes = list()
        mask = np.zeros_like(labels,bool)
        p_default = np.random.rand()
        
        ##
        total_unique_labels = np.unique(labels).tolist()
        point_style = random.choice(['+','-','+-'])
        if p_default<0.5:
            mask = (labels != 0)
            return None, None, None, mask
        else:
            # p_label_contain = random.choice(['1','1','1','2','2','2','3','3','3',0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            p_label_contain = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
        
        ###产生Points
        # if isinstance(p_label_contain,str):
        #     temp_list = deepcopy(total_unique_labels)
        #     temp_list.remove(0)
        #     random.shuffle(temp_list)
        #     labels_contain = total_unique_labels[:int(p_label_contain)]
        #     labels_exclude = total_unique_labels[int(p_label_contain):]
        # else:
        while(1):
            labels_contain = list()
            labels_exclude = list()
            for label in total_unique_labels:
                p_label = np.random.rand()
                if p_label<p_label_contain and label!=0:
                    labels_contain.append(label)
                else:
                    labels_exclude.append(label)
            if len(labels_contain)!= 0:
                break
            
                
        ##Get Points(+) and boxes
        for label in labels_contain:
            mask_label = (labels == label)
            
            ##Mask
            mask = np.logical_or(mask,mask_label)
            
            idx_label = np.where(mask_label)
            y_list = idx_label[0]
            x_list = idx_label[1]
            
            ##Point(+)
            idx = random.choice(np.arange(len(x_list)).tolist())
            # idx = int(len(x_list)/2)
            if '+' in point_style or len(labels_contain)==1:
                Points_pos.append([x_list[idx],y_list[idx]])
                Points_lab.append(1)
            
            ##Box
            # if Get_box:
            box = [np.min(x_list),np.min(y_list),np.max(x_list),np.max(y_list)]
            Boxes.append(box)
            
            
            
        ##Get Points(-) 
        for label in labels_exclude:
            mask_label = (labels == label)
            idx_label = np.where(mask_label)
            y_list = idx_label[0]
            x_list = idx_label[1]
            
            idx = random.choice(np.arange(len(x_list)).tolist())
            # idx = int(len(x_list)/2)
            if '-' in point_style:
                Points_pos.append([x_list[idx],y_list[idx]])
                Points_lab.append(0)

            
        return Points_pos,Points_lab,Boxes,mask
    
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

    def __getitem__(self, idx):

        raw = self.images[idx] #获得第idx张图像
        labels = self.masks[idx] #获得第idx张图像的标签信息
        
        # if self.split == 'val':
        #     seed = 1000
        #     np.random.seed(seed)
        #     random.seed(seed)
            # os.environ['PYTHONHASHSEED'] = str(seed)

        raw = self.normalize(raw) # 所有像素值归一化

        # relabel connected components
        # labels = label(labels).astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域
        # labels = labels.astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域

        padding = self.get_padding(self.crop_size, self.padding_size) # padding_size的整数倍，>= crop_size
        raw, labels = self.augment_data(raw, labels, padding)
        
        raw = np.expand_dims(raw, axis=0) # 1* 图像维度


        # if train/val, generate our gt labels
        ## 非测试数据的话，进行一轮边界腐蚀
        labels = self.erode(
            labels,
            iterations=1, #腐蚀的迭代次数
            border_value=1) # 边界值
        
        affinity = self.getAffinity(labels)

        if self.require_lsd:
            lsds = local_shape_descriptor.get_local_shape_descriptors(
                    segmentation=labels,
                    sigma=(5,)*2,
                    voxel_size=(1,)*2) ##获得lsd标签，维度为：6*图像维度
            
            lsds = lsds.astype(np.float32)    # 6* 图像维度

        Points_pos,Points_lab,Boxes,mask = self.get_prompt(labels)
        
        point_map = self.generate_gaussian_matrix(Points_pos,Points_lab,self.crop_size,self.crop_size,theta=30)
   
        if self.require_lsd:
            # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity,lsds
            return raw.astype(np.float32),1,labels.astype(np.int32),point_map.astype(np.float32),mask.astype(np.float32),affinity.astype(np.float32),lsds.astype(np.float32)
        else:
            # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity
            return raw.astype(np.float32),1,labels.astype(np.int32),point_map.astype(np.float32),mask.astype(np.float32),affinity.astype(np.float32)



# def collate_fn_2D_zebrafinch_Train(batch):
    
#     raw = np.array([item[0] for item in batch]).astype(np.float32)

#     labels = np.array([item[1] for item in batch]).astype(np.int32)

#     point_map = np.array([item[2] for item in batch]).astype(np.float32)
#     mask = np.array([item[3] for item in batch]).astype(np.uint8)
#     affinity = np.array([item[4] for item in batch]).astype(np.float32)
#     if len(batch[0]) == 6:
#         lsds = np.array([item[5] for item in batch]).astype(np.float32)
#         return raw, labels,point_map,mask,affinity,lsds
#     else:
#         return raw, labels,point_map,mask,affinity
#     # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity,lsds
