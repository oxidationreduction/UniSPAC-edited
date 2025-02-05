import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from skimage.measure import label as label_fun
import logging
import zarr

import matplotlib.pyplot as plt
import pyvoi


import numpy as np
from torch.utils.data import DataLoader
# import imageio

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


'''
图2b的代码
'''

NUM_SAMPLES = 1
NUM_NEURONS = 1

import argparse
def get_args():
    parser = argparse.ArgumentParser(description='Test the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--NUM_SAMPLES', dest='NUM_SAMPLES', type=int, default=1,
                        help='NUM_SAMPLES',metavar='E')
    
    parser.add_argument('--NUM_NEURONS', dest='NUM_NEURONS', type=int, default=1,
                        help='NUM_NEURONS',metavar='E')

    return parser.parse_args()
args = get_args()
NUM_SAMPLES = args.NUM_SAMPLES
NUM_NEURONS = args.NUM_NEURONS
print("####################")
print("NUM_NEURONS={},NUM_SAMPLES={}".format(NUM_NEURONS,NUM_SAMPLES))


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
class Dataset_2D_hemi_Test():
    def __init__(
        self,
        roi= 1):

        roi1_raw = zarr.open('./data/funke/hemi/testing/ground_truth/data.zarr/volumes/roi_1/raw', mode='r')
        roi2_raw = zarr.open('./data/funke/hemi/testing/ground_truth/data.zarr/volumes/roi_2/raw', mode='r')
        roi3_raw = zarr.open('./data/funke/hemi/testing/ground_truth/data.zarr/volumes/roi_3/raw', mode='r')


        roi1_label_consolidated = zarr.open('./data/funke/hemi/testing/ground_truth/data.zarr/volumes/roi_1/consolidated_ids', mode='r')
        roi2_label_consolidated = zarr.open('./data/funke/hemi/testing/ground_truth/data.zarr/volumes/roi_2/consolidated_ids', mode='r')
        roi3_label_consolidated = zarr.open('./data/funke/hemi/testing/ground_truth/data.zarr/volumes/roi_3/consolidated_ids', mode='r')

        if roi==1:
            raw = roi1_raw
            label = roi1_label_consolidated
        elif roi==2:
            raw = roi2_raw
            label = roi2_label_consolidated
        elif roi==3:
            raw = roi3_raw
            label = roi3_label_consolidated
            
        # raw = np.array(raw)
        # label = np.array(label_fun(label))


        self.images = list()
        self.masks = list()
        self.labels = list()
        

        # for idx in range(1): ##Debug
        # for idx in range(raw.shape[0]-400):
        for idx in range(0,1000,10):
            # test_patch = np.array(raw[idx:(idx+num_slices),:,:])
            test_patch = np.array(raw[(200+idx),200:-200,200:-200])
            pad_size_x = 8-(test_patch.shape[0]%8)
            pad_size_y = 8-(test_patch.shape[1]%8)
            # print(test_patch.shape)
            test_patch = np.pad(test_patch,((0,pad_size_x),(0,pad_size_y)),'constant',constant_values = (0,0)) 
            self.images.append(test_patch)
            
            mask_patch = (np.array(label[idx,:,:]) != 0 ) + 0
            # print(mask_patch.shape)
            mask_patch = np.pad(mask_patch,((0,pad_size_x),(0,pad_size_y)),'constant',constant_values = (0,0))
            self.masks.append(mask_patch)
            
            self.labels.append(label[idx,:,:])
    
        # # crop_size = 256
        # crop_size = 1024
        # for idx in range(0,1000,100):
        # # for idx in range(raw.shape[0]-400):
        #     # test_patch = np.array(raw[idx:(idx+num_slices),:,:])
        #     test_patch = np.array(raw[(200+idx),200:(200+crop_size),200:(200+crop_size)])
        #     self.images.append(test_patch)
            
        #     mask_patch = (np.array(label[idx,:crop_size,:crop_size]) != 0 ) + 0
        #     self.masks.append(mask_patch)
            
        #     self.labels.append(label[idx,:crop_size,:crop_size])
            
    def get_prompt(self, labels):
        ##
        Points_pos = list()
        Points_lab = list()
        Boxes = list()
        mask = np.zeros_like(labels,bool)
        
        ##
        total_unique_labels = np.unique(labels).tolist()
        # point_style = random.choice(['+','-','+-'])
        point_style = '+'
        total_unique_labels.remove(0)
        random.shuffle(total_unique_labels)
        labels_contain = total_unique_labels[:NUM_NEURONS]
        labels_exclude = list()
        # labels_exclude = total_unique_labels[NUM_NEURONS:]


        ##Get Points(+) and boxes
        for label in labels_contain:
            mask_label = (labels == label)
            
            #Mask
            mask = np.logical_or(mask,mask_label)
            
            # idx_label = np.sort(np.where(mask_label))
            idx_label = np.where(mask_label)
            y_list = idx_label[0]
            x_list = idx_label[1]
            
            ##Point(+)
            for times in range(NUM_SAMPLES):
                idx = random.choice(np.arange(len(x_list)).tolist())
                # idx = int(len(x_list)/5*times)
                # idx = int(len(x_list)/(NUM_SAMPLE+1)*(times+1))
                if '+' in point_style:
                    Points_pos.append([x_list[idx],y_list[idx]])
                    Points_lab.append(1)
            
            
            ##Box
            # if Get_box:
            box = [np.min(x_list),np.min(y_list),np.max(x_list),np.max(y_list)]
            Boxes.append(box)
            
            # Points_pos.append([int(np.mean(x_list)),int(np.mean(y_list))])
            # Points_lab.append(1)



        ##Get Points(-) 
        for label in labels_exclude:
            mask_label = (labels == label)
            idx_label = np.where(mask_label)
            y_list = idx_label[0]
            x_list = idx_label[1]
            
            for times in range(NUM_SAMPLES):
                idx = random.choice(np.arange(len(x_list)).tolist())
                # idx = int(len(x_list)/2)
                if '-' in point_style:
                    Points_pos.append([x_list[idx],y_list[idx]])
                    Points_lab.append(0)

            
        return Points_pos,Points_lab,Boxes,mask
        # return Points_pos,Points_lab,Boxes
    
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
        # mask = self.masks[idx]
        label = self.labels[idx]
        
        Points_pos,Points_lab,Boxes,mask =  self.get_prompt(label)
        
        point_map = self.generate_gaussian_matrix(Points_pos,Points_lab,raw.shape[0],raw.shape[1],theta=30)
        
        raw = np.expand_dims(raw, axis=0) # 1* 图像维度

        return raw, label, mask, point_map, Points_pos,Points_lab


def collate_fn_2D_hemi_test(batch):
    
    raw = np.array([item[0] for item in batch]).astype(np.float32)

    labels = np.array([item[1] for item in batch]).astype(np.uint64)
    
    mask = np.array([item[2] for item in batch]).astype(np.uint8)
    
    point_map = np.array([item[3] for item in batch]).astype(np.float32)
    
    # affinity = np.array([item[7] for item in batch]).astype(np.float32)


    max_Point_num = max([len(item[5]) for item in batch])
    Points_pos = list()
    Points_lab = list()
    for item in batch:
        Points_pos.append(np.pad(item[4],((0,max_Point_num-len(item[4])),(0,0)),'edge'))
        Points_lab.append(np.pad(item[5],((0,max_Point_num-len(item[5]))),'edge') )
    Points_pos = np.array(Points_pos)
    Points_lab = np.array(Points_lab)

    return raw, labels, mask, point_map, Points_pos,Points_lab
    


# matplotlib uses a default shader
# we need to recolor as unique objects

def create_lut(labels):
    np.random.seed(100)
    ##labels标记了图像中所有的连通域，维度对应图像的维度
    labels = label_fun(labels)
    max_label = np.max(labels)
    ## 下面这个lut是给每个连通域随机设定rgb值
    lut = np.random.randint(
            low=0,
            high=255,
            size=(int(max_label + 1), 3),
            dtype=np.uint8)
    lut[0] = 0
    ## 除了rgb外，增加一个通道，lut的维度为：连通域数量*4
    lut = np.append(
            lut,
            np.zeros(
                (int(max_label + 1), 1),
                dtype=np.uint8) + 255,
            axis=1)

    lut[0] = 0 ##第0个连通域为背景，所有通道值设为0
    colored_labels = lut[labels] ## 获得所有像素点的各通道值，维度为：图像维度*通道数（4）

    return colored_labels
def show_points(coords, labels, ax, marker_size=500):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
    
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.functional import threshold, normalize



class SAMneuro2D(nn.Module):
    def __init__(self,model_type = "vit_h",sam_checkpoint = None):
        super(SAMneuro2D, self).__init__()
        ###装载预训练好的模型
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_model = SamPredictor(sam)
        
        self.sigmoid = torch.nn.Sigmoid()
        
        
    def forward(self,input_image, point_coords, point_labels, device):
        '''
        input_image: numpy, shape=(Batch=1 * channel=1 * Height * Width)
        point_coords: numpy, shape=(Batch=1 * n_points * 2)
        point_labels: numpy, shape=(Batch=1 * 1)
        '''
        ## Tranfer to SAM's input shape
        input_image = input_image.transpose(2,3,1,0).repeat(3,axis=2).squeeze() ## H * W * C=3
        point_coords = point_coords[0]
        point_labels = point_labels[0]
        
        # print("input_image shape = {}".format(input_image.shape))
        # print("point_coords shape = {}".format(point_coords.shape))
        # print("point_labels shape = {}".format(point_labels.shape))
        
        self.sam_model.set_image(input_image)
        
        
        y_mask, scores, low_res_masks = self.sam_model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
                return_logits = False,
            )
        
        # y_mask = self.sigmoid(y_mask)
        
        '''
        y_mask: tensor, shape=(Height * Width)
        scores: tensor, shape=(1)
        logits: tensor, shape=(H_down * W_down)
        '''

        return y_mask, scores, low_res_masks




# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

model_SAM = SAMneuro2D(sam_checkpoint = "sam_vit_h_4b8939.pth")

# # model_path = './output/checkpoints/SAM-finetuneDec(hemi+fib25)batch1_Best_in_val.model'
# model_path = './output/checkpoints/SAM-finetunePmpDec(hemi+fib25)batch1_Best_in_val.model'
# weights = torch.load(model_path,map_location=torch.device('cpu'))
# model_SAM.sam_model.model.load_state_dict(weights)

model_SAM.sam_model.model = model_SAM.sam_model.model.to(device)



from models.segNeuro import segneuro2d,segEM2d

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
###创建模型
# model = segEM_3d()
# model_path = './output/checkpoints/segEM_3d(hemi+fib25)_Best_in_val.model'


model = segEM2d()
model_path = './output/checkpoints/segEM2d(hemi+fib25)_Best_in_val.model'
# model_path = './output/checkpoints/segEM2d(hemi+fib25)wloss-1_Best_in_val.model'
weights = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(weights)

model = model.to(device)


##装载数据
test_dataset = Dataset_2D_hemi_Test(roi=1)
test_loader = DataLoader(test_dataset,batch_size=1, shuffle=False, collate_fn=collate_fn_2D_hemi_test)


VOI_segneuro2d = list()
VOI_SAM = list()
###################Test###################
model.eval()
np.random.seed(100)
for data in tqdm(test_loader):
    raw, labels, mask, point_map, Points_pos,Points_lab = data
    
    ##Transfer to tensors
    input_image = torch.as_tensor(raw,dtype=torch.float, device= device) #(batch, 1, height, width)
    point_map = torch.as_tensor(point_map, dtype=torch.float, device=device) #(batch, height, width)
    # point_map = torch.ones(point_map.shape).to(device)
    # mask = torch.as_tensor(mask, dtype=torch.float, device=device) #(batch, 1, height, width)

    with torch.no_grad():
        y_mask,_,y_affinity = model(input_image,point_map)
        
        y_mask_SAM,_,_ = model_SAM(np.array((raw*255),dtype=np.uint8), Points_pos, Points_lab, device)

        
    # labels = labels.squeeze()
    mask = mask.squeeze()
    labels = np.array(label_fun(mask),np.uint64)
        
    input_image = input_image.detach().cpu().numpy().squeeze()
    binary_y_mask = (y_mask>0.5).detach().cpu().numpy().squeeze()
    y_seg = label_fun(binary_y_mask).astype(np.uint64)
    
    binary_y_mask_SAM = (y_mask_SAM>0.5).detach().cpu().numpy().squeeze()
    y_seg_SAM = label_fun(binary_y_mask_SAM).astype(np.uint64)
    
    
    # ###plot
    # fig, axes = plt.subplots(1,6,figsize=(20, 20),sharex=True,sharey=True,squeeze=False)
    

    # axes[0][0].imshow(input_image, cmap='gray')
    # # axes[0][0].title.set_text('Raw image')
    # show_points(Points_pos, Points_lab, axes[0][0])
    # axes[0][0].axis('off')

    # axes[0][1].imshow(binary_y_mask, cmap='binary')
    # # axes[0][1].title.set_text('Predicted Mask')
    # axes[0][1].axis('off')

    # axes[0][2].imshow(input_image, cmap='gray')
    # axes[0][2].imshow(create_lut(y_seg), alpha=0.5)
    # axes[0][2].axis('off')
    
    # axes[0][3].imshow(input_image, cmap='gray')
    # # axes[0][3].imshow(create_lut(labels.squeeze()), alpha=0.5)
    # axes[0][3].imshow(create_lut(label(mask)), alpha=0.5)
    # axes[0][3].axis('off')
    
    # axes[0][4].imshow(binary_y_mask_SAM, cmap='binary')
    # # axes[0][1].title.set_text('Predicted Mask')
    # axes[0][4].axis('off')
    
    # axes[0][5].imshow(input_image, cmap='gray')
    # axes[0][5].imshow(create_lut(y_seg_SAM), alpha=0.5)
    # axes[0][5].axis('off')

    y_seg = y_seg[:labels.shape[0],:labels.shape[1]]
    from funlib.evaluate import rand_voi
    rand_voi_report = rand_voi(
                    labels[labels!=0],
                    y_seg[labels!=0],
                    return_cluster_scores=False)

    metrics = rand_voi_report.copy()
    # print("SegNeuro-2D: voi split = {}, voi merge = {}".format(metrics['voi_split'],metrics['voi_merge']))
    VOI_segneuro2d.append(metrics['voi_split']+ metrics['voi_merge'])


    y_seg_SAM = y_seg_SAM[:labels.shape[0],:labels.shape[1]]
    from funlib.evaluate import rand_voi
    rand_voi_report = rand_voi(
                    labels[labels!=0],
                    y_seg_SAM[labels!=0],
                    return_cluster_scores=False)

    metrics = rand_voi_report.copy()
    # print("SAM: voi split = {}, voi merge = {}".format(metrics['voi_split'],metrics['voi_merge']))
    VOI_SAM.append(metrics['voi_split']+ metrics['voi_merge'])

# plt.show()
# plt.savefig('./figure2a.pdf', dpi=300) 
print("Average VOI segneuro2D = {:.4f}".format(np.mean(VOI_segneuro2d)))
print("Average VOI SAM = {:.4f}".format(np.mean(VOI_SAM)))


