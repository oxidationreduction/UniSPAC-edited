from multiprocessing.pool import Pool
import os.path
import random
from time import sleep

import albumentations as A  ##一种数据增强库
import joblib
import numpy as np
import tqdm
import zarr
from PIL import Image
from lsd.train import local_shape_descriptor
from scipy.ndimage import binary_erosion
# import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# import h5py
from skimage.measure import label
from torch.utils.data import Dataset


ninanjie_data = '/home/liuhongyu2024/sshfs_share/liuhongyu2024/project/unispac/UniSPAC-edited/data/ninanjie'
ninanjie_save = '/home/liuhongyu2024/sshfs_share/liuhongyu2024/project/unispac/UniSPAC-edited/data/ninanjie-save'

def load_dataset(dataset_name: str, split: str='train', from_temp=True, require_lsd=True, require_xz_yz=True):
    temp_name = f'{dataset_name}_{from_temp}_{require_lsd}_{require_xz_yz}'
    DATASET = Dataset_3D_ninanjie_Train if '3d' in dataset_name.lower() else Dataset_2D_ninanjie_Train
    if os.path.exists(os.path.join(ninanjie_save, temp_name)) and from_temp:
        print(f"Load {dataset_name} from disk...")
        train_dataset = joblib.load(os.path.join(ninanjie_save, temp_name))
    else:
        train_dataset = DATASET(data_dir=os.path.join(ninanjie_data), split=split, crop_size=256,
                                                  require_lsd=require_lsd, require_xz_yz=require_xz_yz)
        joblib.dump(train_dataset, os.path.join(ninanjie_save, temp_name))
    return train_dataset


class Dataset_2D_ninanjie_Train(Dataset):
    def __init__(
            self,
            data_dir='./data/ninanjie',  # 数据的路径
            split='train',  # 划分方式
            crop_size=None,  # 切割尺寸
            padding_size=8,
            require_lsd=False,
            require_xz_yz=False):

        self.split = split
        self.crop_size = crop_size
        self.padding_size = padding_size
        self.split = split
        self.require_lsd = require_lsd

        self.images = list()
        self.masks = list()

        data_list = ['first', 'fourth']
        data_list = data_list * 8

        # ##Debug
        # data_list = ['trvol-250-1.zarr']

        raw, labels = list(), list()
        for data in data_list:
            raw_dir = os.path.join(data_dir, data, 'raw')
            labels_dir = os.path.join(data_dir, data, 'labels')

            for image in os.listdir(raw_dir):
                # read image from disk, tiff
                raw_img = Image.open(os.path.join(raw_dir, image))
                raw.append(raw_img)
                label_img = Image.open(os.path.join(labels_dir, image))
                labels.append(label_img)
        raw = np.array(raw)
        labels = np.array(labels).astype(np.uint16)

        raw_crop = []
        labels_crop = []
        lines, rows = (1, 1)
        cropped_size = (raw.shape[1] // lines, raw.shape[2] // rows)
        for line in range(lines):
            for row in range(rows):
                for raw_, labels_ in zip(raw, labels):
                    raw_crop.append(raw_[line * cropped_size[0]:(line + 1) * cropped_size[0], row * cropped_size[1]:(row + 1) * cropped_size[1]])
                    labels_crop.append(labels_[line * cropped_size[0]:(line + 1) * cropped_size[0], row * cropped_size[1]:(row + 1) * cropped_size[1]])
        raw = np.array(raw_crop)
        labels = label(np.array(labels_crop)).astype(np.uint16)

        print('raw shape={}, label shape = {}'.format(raw.shape, labels.shape))

        val_size = 8 # if len(raw) > 40 else max(2, len(raw) // 5)

        if split == 'train':
            self.images.extend(raw[val_size:])
            self.masks.extend(labels[val_size:])
        elif split == 'val':
            self.images.extend(raw[:val_size])
            self.masks.extend(labels[:val_size])

        if require_xz_yz:
            if split == 'train':
                for n in range(val_size, raw.shape[1]):
                    self.images.append(raw[:, n, :])
                    self.masks.append(labels[:, n, :])
                for n in range(val_size, labels.shape[2]):
                    self.images.append(raw[:, :, n])
                    self.masks.append(labels[:, :, n])
            elif split == 'val':
                for n in range(val_size):
                    self.images.append(raw[:, n, :])
                    self.masks.append(labels[:, n, :])
                    self.images.append(raw[:, :, n])
                    self.masks.append(labels[:, :, n])

        self.data_pack = []
        for idx in tqdm.tqdm(range(len(self.images))):
            sub_data = self.prework(idx)
            if sub_data is not None:
                self.data_pack.append(sub_data)
            else:
                print(f"data index={idx} is invalid.")
                # sleep(1)

        # pool = Pool(8)
        # results = []
        # pbar = tqdm.tqdm(total=len(self.images))
        # for idx in range(len(self.images)):
        #     results.append(pool.apply_async(self.prework, args=(idx,pbar)))
        # pool.close()
        # pool.join()
        # self.data_pack = []
        # for result in tqdm.tqdm(results):
        #     self.data_pack.append(result.get())
        # pbar.close()


    def __len__(self):
        return len(self.data_pack)

    # function to erode label boundaries，即腐蚀边界
    def erode(self, labels, iterations, border_value):

        foreground = np.zeros_like(labels, dtype=bool)  # 和标签维度相同的全False矩阵

        # loop through unique labels
        for label in np.unique(labels):  # 遍历标签信息（切割块）中所有的连通域的值

            # skip background
            if label == 0:
                continue

            # mask to label
            label_mask = labels == label  # 当前连通域对应的标签（背景为False，当前连通域为True）

            # erode labels
            eroded_mask = binary_erosion(
                label_mask,
                iterations=iterations,
                border_value=border_value)  # 获得iterations轮腐蚀后的当前连通域的标签

            # get foreground
            foreground = np.logical_or(eroded_mask, foreground)  # 这个前景相当于所有连通域经过边界腐蚀后得到的前景，前景用True标出

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
                height=self.crop_size),  # 1. 随机切割成 crop_size * crop_size 的尺寸
            A.PadIfNeeded(
                min_height=padding,
                min_width=padding,
                p=1,
                border_mode=0),
            # 2. 填充成 padding * padding 的尺寸，border_mode=0似乎是常数填充（参考：https://wenku.csdn.net/answer/b124adffba28441daf7b260623a28d87）
            A.HorizontalFlip(p=0.3),  # 3. 以0.3的概率进行水平翻转
            A.VerticalFlip(p=0.3),  # 4. 以0.3的概率进行垂直翻转
            A.RandomRotate90(p=0.3),  # 5. 以0.3的概率进行垂随机旋转90度
            A.Transpose(p=0.3),  # 6. 以0.3的概率进行转置
            A.RandomBrightnessContrast(p=0.3)  # 7. 以0.3的概率随机改变输入图像的亮度和对比度。
        ])  ## 数据增强

        transformed = transform(image=raw, mask=mask)

        raw, mask = transformed['image'], transformed['mask']

        return raw, mask  ## 图像和标签的维度都是： padding * padding

    # normalize raw data between 0 and 1
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)

    def get_prompt(self, labels):
        ##
        Points_pos = list()
        Points_lab = list()
        Boxes = list()
        mask = np.zeros_like(labels, bool)
        p_default = np.random.rand()

        ##
        total_unique_labels = np.unique(labels).tolist()
        point_style = random.choice(['+', '-', '+-'])
        if p_default < 0.5:
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
        while (1):
            labels_contain = list()
            labels_exclude = list()
            for label in total_unique_labels:
                p_label = np.random.rand()
                if p_label < p_label_contain and label != 0:
                    labels_contain.append(label)
                else:
                    labels_exclude.append(label)
            if len(labels_contain) != 0:
                break

        ##Get Points(+) and boxes
        for label in labels_contain:
            mask_label = (labels == label)

            ##Mask
            mask = np.logical_or(mask, mask_label)

            idx_label = np.where(mask_label)
            y_list = idx_label[0]
            x_list = idx_label[1]

            ##Point(+)
            idx = random.choice(np.arange(len(x_list)).tolist())
            # idx = int(len(x_list)/2)
            if '+' in point_style or len(labels_contain) == 1:
                Points_pos.append([x_list[idx], y_list[idx]])
                Points_lab.append(1)

            ##Box
            # if Get_box:
            box = [np.min(x_list), np.min(y_list), np.max(x_list), np.max(y_list)]
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
                Points_pos.append([x_list[idx], y_list[idx]])
                Points_lab.append(0)

        return Points_pos, Points_lab, Boxes, mask

    def generate_gaussian_matrix(self, Points_pos, Points_lab, H, W, theta=10):
        if Points_pos == None:
            total_matrix = np.ones((H, W))
            return total_matrix

        total_matrix = np.zeros((H, W))

        record_list = list()
        for n, (X, Y) in enumerate(Points_pos):
            if (X, Y) not in record_list:
                record_list.append((X, Y))
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
            matrix = matrix * (1 / np.max(matrix))

            total_matrix = total_matrix + matrix * (Points_lab[n] * 2 - 1)
            # total_matrix = total_matrix + matrix

        if np.max(Points_lab) == 0:
            total_matrix = total_matrix * 2 + 1

        return total_matrix

    # 展示一下求Affinity
    def getAffinity(self, labels):
        '''
        labels为2维: 长*宽
        Return: Affinity为3维, 2*长*宽,其中2对应了长(x)、宽(y)两个方向上的梯度
        '''
        label_shift = np.pad(labels, ((0, 1), (0, 1)), 'edge')

        affinity_x = np.expand_dims(((labels - label_shift[1:, :-1]) != 0) + 0, axis=0)
        affinity_y = np.expand_dims(((labels - label_shift[:-1, 1:]) != 0) + 0, axis=0)

        affinity = np.concatenate([affinity_x, affinity_y], axis=0).astype('float32')

        return affinity

    def prework(self, idx, pbar=None):
        raw = self.images[idx]  # 获得第idx张图像
        labels = self.masks[idx]  # 获得第idx张图像的标签信息

        # if self.split == 'val':
        #     seed = 1000
        #     np.random.seed(seed)
        #     random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)

        raw = self.normalize(raw)  # 所有像素值归一化

        # relabel connected components
        # labels = label(labels).astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域
        # labels = labels.astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域

        padding = self.get_padding(self.crop_size, self.padding_size)  # padding_size的整数倍，>= crop_size
        raw, labels = self.augment_data(raw, labels, padding)

        raw = np.expand_dims(raw, axis=0)  # 1* 图像维度

        # if train/val, generate our gt labels
        ## 非测试数据的话，进行一轮边界腐蚀
        # labels = self.erode(
        #     labels,
        #     iterations=1,  # 腐蚀的迭代次数
        #     border_value=1)  # 边界值

        affinity = self.getAffinity(labels)

        if self.require_lsd:
            lsds = local_shape_descriptor.get_local_shape_descriptors(
                segmentation=labels,
                sigma=(5,) * 2,
                voxel_size=(1,) * 2)  ##获得lsd标签，维度为：6*图像维度

            lsds = lsds.astype(np.float32)  # 6* 图像维度

        if len(np.unique(labels)) == 1:
            if pbar is not None:
                pbar.update()
            return None

        Points_pos, Points_lab, Boxes, mask = self.get_prompt(labels)

        point_map = self.generate_gaussian_matrix(Points_pos, Points_lab, self.crop_size, self.crop_size, theta=30)
        if pbar is not None:
            pbar.update()

        if self.require_lsd:
            # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity,lsds
            return raw, labels, point_map, mask, affinity, lsds
        else:
            # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity
            return raw, labels, point_map, mask, affinity

    def __getitem__(self, idx):
        return self.data_pack[idx]


def collate_fn_2D_fib25_Train(batch):
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
        return raw, labels, point_map, mask, affinity, lsds
    else:
        return raw, labels, point_map, mask, affinity
    # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity,lsds


class Dataset_3D_ninanjie_Train(Dataset):
    def __init__(
            self,
            data_dir='./data/ninanjie/',  # 数据的路径
            split='train',  # 划分方式
            crop_size=None,  # 切割尺寸
            num_slices=8,
            padding_size=8,
            require_lsd=False,
            require_xz_yz=False):

        self.split = split
        self.crop_size = crop_size
        self.num_slices = num_slices
        self.padding_size = padding_size
        self.split = split
        self.require_lsd = require_lsd

        ###装载FIB-25的训练数据
        self.images = list()
        self.masks = list()
        self.idxs = list()

        ##装载数据
        # data_list = ['trvol-250-1.zarr', 'trvol-250-2.zarr', 'tstvol-520-1.zarr', 'tstvol-520-2.zarr']
        # data_list = data_list * 8

        data_list = ['first', 'fourth']
        # data_list = data_list * 8

        # ##Debug
        # data_list = ['trvol-250-1.zarr']

        for data in data_list:
            raw_dir = os.path.join(data_dir, data, 'raw')
            labels_dir = os.path.join(data_dir, data, 'labels')

            raw_, labels_ = list(), list()
            for image in os.listdir(raw_dir):
                # read image from disk, tiff
                raw_img = Image.open(os.path.join(raw_dir, image))
                raw_.append(raw_img)
                label_img = Image.open(os.path.join(labels_dir, image))
                labels_.append(label_img)
            raw_ = np.array(raw_)
            labels_ = np.array(labels_)
            print('raw shape={}, label shape={}'.format(raw_.shape, labels_.shape))
            val_size = 8 # if len(raw_) > 40 else max(2, len(raw_) // 5)

            if split == 'train':
                self.images.append(raw_[val_size:])
                self.masks.append(labels_[val_size:])
            elif split == 'val':
                self.images.append(raw_[:val_size])
                self.masks.append(labels_[:val_size])

            if require_xz_yz:
                if split == 'train':
                    self.images.append(raw_.transpose(1, 0, 2)[val_size:])
                    self.masks.append(labels_.transpose(1, 0, 2)[val_size:])
                    self.images.append(raw_.transpose(1, 2, 0)[val_size:])
                    self.masks.append(labels_.transpose(1, 2, 0)[val_size:])
                elif split == 'val':
                    self.images.append(raw_.transpose(1, 0, 2)[:val_size])
                    self.masks.append(labels_.transpose(1, 0, 2)[:val_size])
                    self.images.append(raw_.transpose(1, 2, 0)[:val_size])
                    self.masks.append(labels_.transpose(1, 2, 0)[:val_size])

        # raw_crop = []
        # labels_crop = []
        # lines, rows = (1, 1)
        # cropped_size = (raw.shape[1] // lines, raw.shape[2] // rows)
        # for line in range(lines):
        #     for row in range(rows):
        #         for raw_, labels_ in zip(raw, labels):
        #             raw_crop.append(raw_[line * cropped_size[0]:(line + 1) * cropped_size[0],
        #                             row * cropped_size[1]:(row + 1) * cropped_size[1]])
        #             labels_crop.append(labels_[line * cropped_size[0]:(line + 1) * cropped_size[0],
        #                                row * cropped_size[1]:(row + 1) * cropped_size[1]])
        # raw = np.array(raw_crop)
        # labels = label(np.array(labels_crop)).astype(np.uint16)

        # for data_name in data_list:
        #     zarr_path = data_dir + data_name
        #     f = zarr.open(zarr_path, mode='r')
        #     volumes = f['volumes']
        #     raw = volumes['raw']  # zyx
        #     labels = volumes['labels']['neuron_ids']  # zyx
        #
        #     raw = np.array(raw)
        #     labels = label(labels).astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域
        #
        #     print('data {}: raw shape={}, label shape = {}'.format(data_name, raw.shape, labels.shape))
        #
        #     if split == 'train':
        #         self.images.append(raw[8:])
        #         self.masks.append(labels[8:])
        #     elif split == 'val':
        #         self.images.append(raw[:8])
        #         self.masks.append(labels[:8])
        #
        #     if require_xz_yz:
        #         if split == 'train':
        #             self.images.append(raw.transpose(1, 0, 2)[8:])
        #             self.masks.append(labels.transpose(1, 0, 2)[8:])
        #             self.images.append(raw.transpose(1, 2, 0)[8:])
        #             self.masks.append(labels.transpose(1, 2, 0)[8:])
        #         elif split == 'val':
        #             self.images.append(raw.transpose(1, 0, 2)[:8])
        #             self.masks.append(labels.transpose(1, 0, 2)[:8])
        #             self.images.append(raw.transpose(1, 2, 0)[:8])
        #             self.masks.append(labels.transpose(1, 2, 0)[:8])

        ##load all 3D patches
        for idx_img in range(len(self.images)):
            for idx_slice in range(self.images[idx_img].shape[0] - self.num_slices + 1):
                if not np.any(self.masks[idx_img][idx_slice]):
                    continue
                self.idxs.append([idx_img, idx_slice])

        self.data_pack = []
        for idx in tqdm.tqdm(range(len(self.idxs)), leave=False):
            sub_data = self.prework(idx)
            if sub_data is not None:
                self.data_pack.append(sub_data)
            else:
                print(f"data index={idx} is invalid.")

    def __len__(self):
        return len(self.idxs)

    # function to erode label boundaries，即腐蚀边界
    def erode(self, labels, iterations, border_value):

        foreground = np.zeros_like(labels, dtype=bool)  # 和标签维度相同的全False矩阵

        # loop through unique labels
        for label in np.unique(labels):  # 遍历标签信息（切割块）中所有的连通域的值

            # skip background
            if label == 0:
                continue

            # mask to label
            label_mask = labels == label  # 当前连通域对应的标签（背景为False，当前连通域为True）

            # erode labels
            eroded_mask = binary_erosion(
                label_mask,
                iterations=iterations,
                border_value=border_value)  # 获得iterations轮腐蚀后的当前连通域的标签

            # get foreground
            foreground = np.logical_or(eroded_mask, foreground)  # 这个前景相当于所有连通域经过边界腐蚀后得到的前景，前景用True标出

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
                height=self.crop_size),  # 1. 随机切割成 crop_size * crop_size 的尺寸
            A.PadIfNeeded(
                min_height=padding,
                min_width=padding,
                p=1,
                border_mode=0),
            # 2. 填充成 padding * padding 的尺寸，border_mode=0似乎是常数填充（参考：https://wenku.csdn.net/answer/b124adffba28441daf7b260623a28d87）
            A.HorizontalFlip(p=0.3),  # 3. 以0.3的概率进行水平翻转
            A.VerticalFlip(p=0.3),  # 4. 以0.3的概率进行垂直翻转
            A.RandomRotate90(p=0.3),  # 5. 以0.3的概率进行垂随机旋转90度
            A.Transpose(p=0.3),  # 6. 以0.3的概率进行转置
            A.RandomBrightnessContrast(p=0.3)  # 7. 以0.3的概率随机改变输入图像的亮度和对比度。
        ])  ## 数据增强

        transformed = transform(image=raw, mask=mask)

        raw, mask = transformed['image'], transformed['mask']

        return raw, mask  ## 图像和标签的维度都是： padding * padding

    # normalize raw data between 0 and 1
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)

    # 展示一下求Affinity
    def getAffinity(self, labels):
        '''
        功能:参考gunpowder的seg_to_affgraph函数,只计算前景的边缘!
        labels为3维: 长*宽*高
        Return: Affinity为4维, 3*长*宽*高,其中3对应了长(x)、宽(y)、高(z)三个方向上的梯度
        '''
        # dim_x,dim_y,dim_z = np.size(labels,0),np.size(labels,1),np.size(labels,2)
        label_shift = np.pad(labels, ((0, 1), (0, 1), (0, 1)), 'edge')

        affinity_x = np.expand_dims(((labels - label_shift[1:, :-1, :-1]) != 0) + 0, axis=0)
        affinity_y = np.expand_dims(((labels - label_shift[:-1, 1:, :-1]) != 0) + 0, axis=0)
        affinity_z = np.expand_dims(((labels - label_shift[:-1, :-1, 1:]) != 0) + 0, axis=0)

        background = (labels == 0)
        affinity_x[0][background] = 1
        affinity_y[0][background] = 1
        affinity_z[0][background] = 1

        affinity = np.concatenate([affinity_x, affinity_y, affinity_z], axis=0).astype('float32')

        return affinity

    def get_Mask(self, labels):
        ##
        Points_pos = list()
        Points_lab = list()
        Boxes = list()
        mask_3D = np.zeros_like(labels, bool)
        p_default = np.random.rand()

        ##
        total_unique_labels = np.unique(labels[:, :, 0]).tolist()
        if p_default < 0.5:
            p_label_contain = 1
        else:
            p_label_contain = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        ###选择保留哪些神经元, 得到mask
        count = 0
        while (1):
            count = count + 1
            labels_contain = list()
            labels_exclude = list()
            for label in total_unique_labels:
                p_label = np.random.rand()
                if p_label < p_label_contain and label != 0:
                    labels_contain.append(label)
                else:
                    labels_exclude.append(label)
            if len(labels_contain) != 0:
                break
            if count > 10:
                labels_contain = total_unique_labels
                break

        for label in labels_contain:
            mask_label = (labels == label)

            ##Mask
            mask_3D = np.logical_or(mask_3D, mask_label)

        if p_default < 0.5:
            Points_pos = None
            Points_lab = None
        else:
            ###选择第一张slice的mask，得到prompt
            Points_pos = list()  # position
            Points_lab = list()  # label
            point_style = random.choice(['+', '-', '+-'])
            ##Get Points(+)
            for label in labels_contain:
                mask_label = (labels[:, :, 0] == label)
                idx_label = np.where(mask_label)
                y_list = idx_label[0]
                x_list = idx_label[1]

                ##Point(+)
                idx = random.choice(np.arange(len(x_list)).tolist())
                # idx = int(len(x_list)/2)
                if '+' in point_style:
                    Points_pos.append([x_list[idx], y_list[idx]])
                    Points_lab.append(1)

            ##Get Points(-)
            for label in labels_exclude:
                mask_label = (labels[:, :, 0] == label)
                idx_label = np.where(mask_label)
                y_list = idx_label[0]
                x_list = idx_label[1]

                idx = random.choice(np.arange(len(x_list)).tolist())
                # idx = int(len(x_list)/2)
                if '-' in point_style:
                    Points_pos.append([x_list[idx], y_list[idx]])
                    Points_lab.append(0)

        return mask_3D, Points_pos, Points_lab

    def generate_gaussian_matrix(self, Points_pos, Points_lab, H, W, theta=10):
        if Points_pos == None:
            total_matrix = np.ones((H, W))
            return total_matrix

        total_matrix = np.zeros((H, W))

        record_list = list()
        for n, (X, Y) in enumerate(Points_pos):
            if (X, Y) not in record_list:
                record_list.append((X, Y))
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
            matrix = matrix * (1 / np.max(matrix))

            total_matrix = total_matrix + matrix * (Points_lab[n] * 2 - 1)
            # total_matrix = total_matrix + matrix

        try:
            if np.max(Points_lab) == 0:
                total_matrix = total_matrix * 2 + 1
        except:
            print(Points_pos)
            print(Points_lab)

        return total_matrix

    def prework(self, idx):
        idx_image = self.idxs[idx][0]
        idx_slice = self.idxs[idx][1]

        raw = self.images[idx_image]  # 获得第idx_image张图像
        labels = self.masks[idx_image]  # 获得第idx_image张图像的标签信息

        raw = raw[idx_slice:(idx_slice + self.num_slices)]
        labels = labels[idx_slice:(idx_slice + self.num_slices)]
        raw = raw.transpose(1, 2, 0)  # xyz
        labels = labels.transpose(1, 2, 0)  # xyz

        raw = self.normalize(raw)  # 所有像素值归一化

        # relabel connected components
        # labels = label(labels).astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域
        # labels = labels.astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域

        padding = self.get_padding(self.crop_size, self.padding_size)  # padding_size的整数倍，>= crop_size
        raw, labels = self.augment_data(raw, labels, padding)

        raw = np.expand_dims(raw, axis=0)  # 1* 图像维度

        # if train/val, generate our gt labels
        ## 非测试数据的话，进行一轮边界腐蚀
        labels = self.erode(
            labels,
            iterations=1,  # 腐蚀的迭代次数
            border_value=1)  # 边界值

        affinity = self.getAffinity(labels)

        if self.require_lsd:
            lsds = local_shape_descriptor.get_local_shape_descriptors(
                segmentation=labels,
                sigma=(5,) * 3,
                voxel_size=(1,) * 3)  ##获得lsd标签，维度为：10*图像维度

            lsds = lsds.astype(np.float32)  # 10* 图像维度

        mask_3D, Points_pos, Points_lab = self.get_Mask(labels)

        point_map = self.generate_gaussian_matrix(Points_pos, Points_lab, self.crop_size, self.crop_size, theta=30)

        if self.require_lsd:
            return raw, labels, mask_3D, affinity, point_map, lsds
        else:
            return raw, labels, mask_3D, affinity, point_map

    def __getitem__(self, idx):
        return self.data_pack[idx]


def collate_fn_3D_ninanjie_Train(batch):
    raw = np.array([item[0] for item in batch]).astype(np.float32)  # 注意normalize了，这里要用float

    labels = np.array([item[1] for item in batch]).astype(np.int32)

    mask_3D = np.array([item[2] for item in batch]).astype(np.uint8)

    affinity = np.array([item[3] for item in batch]).astype(np.float32)

    point_map = np.array([item[4] for item in batch]).astype(np.float32)

    # lsds = np.array([item[5] for item in batch]).astype(np.float32)
    if len(batch[0]) == 6:
        lsds = np.array([item[5] for item in batch]).astype(np.float32)
        return raw, labels, mask_3D, affinity, point_map, lsds
    else:
        return raw, labels, mask_3D, affinity, point_map

    # return raw, labels, mask_3D, affinity, point_map,lsds


###dataloader for SAM
class SAM_Dataset_2D_fib25_Train(Dataset):
    def __init__(
            self,
            data_dir='./data/fib25/training/',  # 数据的路径
            split='train',  # 划分方式
            crop_size=None,  # 切割尺寸
            padding_size=8,
            require_lsd=False,
            require_xz_yz=False, ):

        self.split = split
        self.crop_size = crop_size
        self.padding_size = padding_size
        self.split = split
        self.require_lsd = require_lsd

        ###装载FIB-25的训练数据
        self.images = list()
        self.masks = list()
        # 装载数据
        data_list = ['trvol-250-1.zarr', 'trvol-250-2.zarr', 'tstvol-520-1.zarr', 'tstvol-520-2.zarr']
        # data_list = data_list * 8

        # ##Debug
        # data_list = ['trvol-250-1.zarr']

        for data_name in data_list:
            zarr_path = data_dir + data_name
            f = zarr.open(zarr_path, mode='r')
            volumes = f['volumes']
            raw = volumes['raw']  # zyx
            labels = volumes['labels']['neuron_ids']  # zyx

            labels = label(labels).astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域

            print('data {}: raw shape={}, label shape = {}'.format(data_name, raw.shape, labels.shape))

            if split == 'train':
                for n in range(8, len(labels)):
                    self.images.append(raw[n])
                    self.masks.append(labels[n])
            elif split == 'val':
                for n in range(8):
                    self.images.append(raw[n])
                    self.masks.append(labels[n])

            if require_xz_yz:
                if split == 'train':
                    for n in range(8, labels.shape[1]):
                        self.images.append(raw[:, n, :])
                        self.masks.append(labels[:, n, :])
                    for n in range(8, labels.shape[2]):
                        self.images.append(raw[:, :, n])
                        self.masks.append(labels[:, :, n])
                elif split == 'val':
                    for n in range(8):
                        self.images.append(raw[:, n, :])
                        self.masks.append(labels[:, n, :])
                        self.images.append(raw[:, :, n])
                        self.masks.append(labels[:, :, n])

    def __len__(self):
        return len(self.images)

    # function to erode label boundaries，即腐蚀边界
    def erode(self, labels, iterations, border_value):

        foreground = np.zeros_like(labels, dtype=bool)  # 和标签维度相同的全False矩阵

        # loop through unique labels
        for label in np.unique(labels):  # 遍历标签信息（切割块）中所有的连通域的值

            # skip background
            if label == 0:
                continue

            # mask to label
            label_mask = labels == label  # 当前连通域对应的标签（背景为False，当前连通域为True）

            # erode labels
            eroded_mask = binary_erosion(
                label_mask,
                iterations=iterations,
                border_value=border_value)  # 获得iterations轮腐蚀后的当前连通域的标签

            # get foreground
            foreground = np.logical_or(eroded_mask, foreground)  # 这个前景相当于所有连通域经过边界腐蚀后得到的前景，前景用True标出

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
                height=self.crop_size),  # 1. 随机切割成 crop_size * crop_size 的尺寸
            A.PadIfNeeded(
                min_height=padding,
                min_width=padding,
                p=1,
                border_mode=0),
            # 2. 填充成 padding * padding 的尺寸，border_mode=0似乎是常数填充（参考：https://wenku.csdn.net/answer/b124adffba28441daf7b260623a28d87）
            A.HorizontalFlip(p=0.3),  # 3. 以0.3的概率进行水平翻转
            A.VerticalFlip(p=0.3),  # 4. 以0.3的概率进行垂直翻转
            A.RandomRotate90(p=0.3),  # 5. 以0.3的概率进行垂随机旋转90度
            A.Transpose(p=0.3),  # 6. 以0.3的概率进行转置
            A.RandomBrightnessContrast(p=0.3)  # 7. 以0.3的概率随机改变输入图像的亮度和对比度。
        ])  ## 数据增强

        transformed = transform(image=raw, mask=mask)

        raw, mask = transformed['image'], transformed['mask']

        return raw, mask  ## 图像和标签的维度都是： padding * padding

    # normalize raw data between 0 and 1
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)

    def get_prompt(self, labels):

        ##
        Points_pos = list()
        Points_lab = list()
        Boxes = list()
        mask = np.zeros_like(labels, bool)
        # p_default = np.random.rand()
        p_default = 1

        ##
        total_unique_labels = np.unique(labels).tolist()
        point_style = random.choice(['+', '-', '+-'])
        if p_default < 0.5:
            mask = (labels != 0)
            Points_pos = [[0, 0]]
            Points_lab = [0]
            Boxes = [[0, 0, 0, 0]]
            return Points_pos, Points_lab, Boxes, mask
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
        while (1):
            labels_contain = list()
            labels_exclude = list()
            for label in total_unique_labels:
                p_label = np.random.rand()
                if p_label < p_label_contain and label != 0:
                    labels_contain.append(label)
                else:
                    labels_exclude.append(label)
            if len(labels_contain) != 0:
                break

        ##Get Points(+) and boxes
        for label in labels_contain:
            mask_label = (labels == label)

            ##Mask
            mask = np.logical_or(mask, mask_label)

            idx_label = np.where(mask_label)
            y_list = idx_label[0]
            x_list = idx_label[1]

            ##Point(+)
            idx = random.choice(np.arange(len(x_list)).tolist())
            # idx = int(len(x_list)/2)
            if '+' in point_style or len(labels_contain) == 1:
                Points_pos.append([x_list[idx], y_list[idx]])
                Points_lab.append(1)

            ##Box
            # if Get_box:
            box = [np.min(x_list), np.min(y_list), np.max(x_list), np.max(y_list)]
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
                Points_pos.append([x_list[idx], y_list[idx]])
                Points_lab.append(0)

        return Points_pos, Points_lab, Boxes, mask

    def generate_gaussian_matrix(self, Points_pos, Points_lab, H, W, theta=10):
        if Points_pos == None:
            total_matrix = np.ones((H, W))
            return total_matrix

        total_matrix = np.zeros((H, W))

        record_list = list()
        for n, (X, Y) in enumerate(Points_pos):
            if (X, Y) not in record_list:
                record_list.append((X, Y))
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
            matrix = matrix * (1 / np.max(matrix))

            total_matrix = total_matrix + matrix * (Points_lab[n] * 2 - 1)
            # total_matrix = total_matrix + matrix

        if np.max(Points_lab) == 0:
            total_matrix = total_matrix * 2 + 1

        return total_matrix

    # 展示一下求Affinity
    def getAffinity(self, labels):
        '''
        labels为2维: 长*宽
        Return: Affinity为3维, 2*长*宽,其中2对应了长(x)、宽(y)两个方向上的梯度
        '''
        label_shift = np.pad(labels, ((0, 1), (0, 1)), 'edge')

        affinity_x = np.expand_dims(((labels - label_shift[1:, :-1]) != 0) + 0, axis=0)
        affinity_y = np.expand_dims(((labels - label_shift[:-1, 1:]) != 0) + 0, axis=0)

        affinity = np.concatenate([affinity_x, affinity_y], axis=0).astype('float32')

        return affinity

    def __getitem__(self, idx):

        raw = self.images[idx]  # 获得第idx张图像
        labels = self.masks[idx]  # 获得第idx张图像的标签信息

        # if self.split == 'val':
        #     seed = 1000
        #     np.random.seed(seed)
        #     random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)

        # raw = self.normalize(raw) # 所有像素值归一化

        # relabel connected components
        # labels = label(labels).astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域
        # labels = labels.astype(np.uint16)  # 读取通道0的标签信息，并设置好连通域

        padding = self.get_padding(self.crop_size, self.padding_size)  # padding_size的整数倍，>= crop_size
        raw, labels = self.augment_data(raw, labels, padding)

        raw = np.expand_dims(raw, axis=0)  # channel=1 * 2D图像维度

        # if train/val, generate our gt labels
        ## 非测试数据的话，进行一轮边界腐蚀
        labels = self.erode(
            labels,
            iterations=1,  # 腐蚀的迭代次数
            border_value=1)  # 边界值

        affinity = self.getAffinity(labels)

        if self.require_lsd:
            lsds = local_shape_descriptor.get_local_shape_descriptors(
                segmentation=labels,
                sigma=(5,) * 2,
                voxel_size=(1,) * 2)  ##获得lsd标签，维度为：6*图像维度

            lsds = lsds.astype(np.float32)  # 6* 图像维度

        Points_pos, Points_lab, Boxes, mask = self.get_prompt(labels)

        point_map = self.generate_gaussian_matrix(Points_pos, Points_lab, self.crop_size, self.crop_size, theta=30)

        if self.require_lsd:
            return raw, labels, Points_pos, Points_lab, Boxes, point_map, mask, affinity, lsds
            # return raw, labels,point_map,mask,affinity,lsds
        else:
            return raw, labels, Points_pos, Points_lab, Boxes, point_map, mask, affinity
            # return raw, labels,point_map,mask,affinity


def collate_fn_2D_fib25_Train_SAM(batch):
    # raw = np.array([item[0] for item in batch]).astype(np.float32)
    raw = np.array([item[0] for item in batch]).astype(np.uint8)

    labels = np.array([item[1] for item in batch]).astype(np.int32)

    max_Point_num = max([len(item[2]) for item in batch])
    max_Box_num = max([len(item[4]) for item in batch])
    Points_pos = list()
    Points_lab = list()
    Boxes = list()
    for item in batch:
        Points_pos.append(np.pad(item[2], ((0, max_Point_num - len(item[2])), (0, 0)), 'edge'))
        Points_lab.append(np.pad(item[3], ((0, max_Point_num - len(item[3]))), 'edge'))
        Boxes.append(np.pad(item[4], ((0, max_Box_num - len(item[4])), (0, 0)), 'edge'))
    Points_pos = np.array(Points_pos)
    Points_lab = np.array(Points_lab)
    Boxes = np.array(Boxes)

    point_map = np.array([item[5] for item in batch]).astype(np.float32)
    mask = np.array([item[6] for item in batch]).astype(np.uint8)
    affinity = np.array([item[7] for item in batch]).astype(np.float32)
    if len(batch[0]) == 9:
        lsds = np.array([item[8] for item in batch]).astype(np.float32)
        return raw, labels, Points_pos, Points_lab, Boxes, point_map, mask, affinity, lsds
    else:
        return raw, labels, Points_pos, Points_lab, Boxes, point_map, mask, affinity

    # point_map = np.array([item[2] for item in batch]).astype(np.float32)
    # mask = np.array([item[3] for item in batch]).astype(np.uint8)
    # affinity = np.array([item[4] for item in batch]).astype(np.float32)
    # if len(batch[0]) == 6:
    #     lsds = np.array([item[5] for item in batch]).astype(np.float32)
    #     return raw, labels,point_map,mask,affinity,lsds
    # else:
    #     return raw, labels,point_map,mask,affinity
    # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity,lsds
