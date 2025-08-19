import multiprocessing
from multiprocessing.pool import Pool
import os
import random
from time import sleep

import torch

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
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
import torch.nn.functional as F


ninanjie_data = '/home/liuhongyu2024/Documents/UniSPAC-edited/data/ninanjie'
ninanjie_save = '/home/liuhongyu2024/Documents/UniSPAC-edited/data/ninanjie-save'

def load_train_dataset(dataset_name: str, raw_dir=None, label_dir=None, from_temp=True, require_lsd=True, require_xz_yz=True,
                 crop_size=None, crop_xyz=None, chunk_position=None):
    temp_name = (f'{dataset_name}_{from_temp}_{require_lsd}_{require_xz_yz}_{crop_size}_'
                 f'{"".join((str(i) for i in crop_xyz))}_{"".join((str(i) for i in chunk_position))}_full')

    DATASET = Dataset_3D_ninanjie_Train_GPU if '_3d' in dataset_name.lower() else Dataset_2D_ninanjie_Train
    dataset_name = dataset_name.replace('_3d', '')
    if os.path.exists(os.path.join(ninanjie_save, temp_name)) and from_temp:
        print(f'load {dataset_name} from disk')
        train_dataset, val_dataset = joblib.load(os.path.join(ninanjie_save, temp_name))
    else:
        train_dataset = DATASET(data_dir=os.path.join(ninanjie_data, 'train'), batch_num=dataset_name,
                                split='train', crop_size=crop_size,
                                raw_dir=raw_dir, label_dir=label_dir,
                                require_lsd=require_lsd, require_xz_yz=require_xz_yz,
                                crop_xyz=crop_xyz, chunk_position=chunk_position)
        val_dataset = DATASET(data_dir=os.path.join(ninanjie_data, 'train'), batch_num=dataset_name,
                              split='val', crop_size=crop_size,
                              raw_dir=raw_dir, label_dir=label_dir,
                              require_lsd=require_lsd, require_xz_yz=require_xz_yz,
                              crop_xyz=crop_xyz, chunk_position=chunk_position)
        joblib.dump((train_dataset, val_dataset), os.path.join(ninanjie_save, temp_name))
    return train_dataset, val_dataset


def load_semantic_dataset(dataset_name: str, raw_dir=None, label_dir=None, from_temp=True, require_lsd=True,
                          require_xz_yz=True, crop_size=None, crop_xyz=None, chunk_position=None):
    temp_name = (f'semantic-{dataset_name}_{from_temp}_{require_lsd}_{require_xz_yz}_{crop_size}_'
                 f'{"".join((str(i) for i in crop_xyz))}_{"".join((str(i) for i in chunk_position))}')
    DATASET = Dataset_3D_ninanjie_Train if '3d' in dataset_name.lower() else Dataset_2D_semantic_Train
    if os.path.exists(os.path.join(ninanjie_save, temp_name)) and from_temp:
        train_dataset, val_dataset = joblib.load(os.path.join(ninanjie_save, temp_name))
    else:
        train_dataset = DATASET(data_dir=os.path.join(ninanjie_data, 'train'), batch_num=dataset_name,
                                split='train', crop_size=crop_size, raw_dir=raw_dir, label_dir=label_dir,
                                require_lsd=require_lsd, require_xz_yz=require_xz_yz,
                                crop_xyz=crop_xyz, chunk_position=chunk_position)
        val_dataset = DATASET(data_dir=os.path.join(ninanjie_data, 'train'), batch_num=dataset_name,
                              split='val', crop_size=crop_size, raw_dir=raw_dir, label_dir=label_dir,
                              require_lsd=require_lsd, require_xz_yz=require_xz_yz,
                              crop_xyz=crop_xyz, chunk_position=chunk_position)
        joblib.dump((train_dataset, val_dataset), os.path.join(ninanjie_save, temp_name))
    return train_dataset, val_dataset


def load_test_dataset(dataset_name: str, raw_dir='raw', label_dir='label', from_temp=True, crop_size=None,
                      crop_xyz=None, chunk_position=None, semantic_class_num=None):
    temp_name = (f'test_{dataset_name}_{from_temp}_{crop_size}_{"".join((str(i) for i in crop_xyz))}_'
                 f'{"".join((str(i) for i in chunk_position))}_{semantic_class_num}')
    DATASET = Dataset_3D_ninanjie_Train if '3d' in dataset_name.lower() else Dataset_2D_ninanjie_Test
    if os.path.exists(os.path.join(ninanjie_save, temp_name)) and from_temp:
        print(f"Load {dataset_name} from disk...")
        test_dataset = joblib.load(os.path.join(ninanjie_save, temp_name))
    else:
        test_dataset = DATASET(data_dir=os.path.join(ninanjie_data, 'train'), batch_num=dataset_name,
                               raw_dir=raw_dir, label_dir=label_dir, class_num=semantic_class_num,
                               crop_size=crop_size, crop_xyz=crop_xyz, chunk_position=chunk_position)
        joblib.dump(test_dataset, os.path.join(ninanjie_save, temp_name))
    return test_dataset


def _add_image(raw_dir_, labels_dir_, image_, crop_xy_, chunk_pos_):
    try:
        raw_img = Image.open(os.path.join(raw_dir_, image_))
        # label_img_ = str(int(image_.split('.')[0]) - 1).zfill(4) + '.tif'
        label_img = Image.open(os.path.join(labels_dir_, image_)) if labels_dir_ else None
        x_, y_ = raw_img.size
    except FileNotFoundError as e:
        return None, None
    left, right = chunk_pos_[0] * x_ // crop_xy_[0], (chunk_pos_[0]+1) * x_ // crop_xy_[0]
    upper, lower = chunk_pos_[1] * y_ // crop_xy_[1], (chunk_pos_[1]+1) * y_ // crop_xy_[1]

    cropped_raw = raw_img.crop((left, upper, right, lower))
    cropped_label = label_img.crop((left, upper, right, lower)) if labels_dir_ else None

    return cropped_raw, cropped_label


def _load_images(data_dir, data_list):
    with multiprocessing.Manager() as manager:
        raw, labels = manager.list(), manager.list()
        process_pool = Pool(processes=os.cpu_count() >> 1)
        images = list()

        for data in data_list:
            raw_dir = os.path.join(data_dir, data, 'raw')
            labels_dir = os.path.join(data_dir, data, 'label')

            for image in os.listdir(raw_dir):
                images.append((raw_dir, labels_dir, image))

        results = []
        for raw_dir, labels_dir, image in images:
            result = process_pool.apply_async(_add_image, args=(raw_dir, labels_dir, image))
            results.append(result)

        pbar = tqdm.tqdm(total=len(images), desc="load dataset", leave=False)
        for result in results:
            raw_img, label_img = result.get()
            if raw_img and label_img:
                raw.append(raw_img)
                labels.append(label_img)
            pbar.update(1)
        pbar.close()
        process_pool.close()
        process_pool.join()
        raw, labels = list(raw), list(labels)
        return raw, labels


def _dump_data(data_list, data_dir='./data/ninanjie'):
    raw, labels = [], []
    for data in data_list:
        raw_dir = os.path.join(data_dir, data, 'raw')
        labels_dir = os.path.join(data_dir, data, 'label')

        for image in tqdm.tqdm(os.listdir(raw_dir)):
            raw_img, label_img = _add_image(raw_dir, labels_dir, image)
            if raw_img and label_img:
                raw.append(raw_img)
                labels.append(label_img)
        joblib.dump((raw, labels), os.path.join(ninanjie_save, "_".join(data_list)))


def get_acc_prec_recall_f1(y_flat, gt_flat):
    # 计算真正例(TP)、假正例(FP)、真反例(TN)、假反例(FN)
    TP = np.sum((y_flat == 1) & (gt_flat == 1))
    TN = np.sum((y_flat == 0) & (gt_flat == 0))
    FP = np.sum((y_flat == 1) & (gt_flat == 0))
    FN = np.sum((y_flat == 0) & (gt_flat == 1))

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total != 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return accuracy, precision, recall, f1


import numpy as np


def variation_of_information(labels_true, labels_pred):
    """
    计算两个分割结果之间的变分信息(VOI)

    参数:
    labels_true: 真实分割标签数组
    labels_pred: 预测分割标签数组

    返回:
    voi: 变分信息值
    """
    # 确保输入是一维数组
    labels_true = np.asarray(labels_true).flatten()
    labels_pred = np.asarray(labels_pred).flatten()

    # 检查输入长度是否一致
    assert len(labels_true) == len(labels_pred)

    # 计算联合概率分布
    n_samples = len(labels_true)
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    n_classes_true = len(classes_true)
    n_classes_pred = len(classes_pred)

    # 构建混淆矩阵
    contingency = np.zeros((n_classes_true, n_classes_pred))
    for i, c in enumerate(classes_true):
        for j, d in enumerate(classes_pred):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == d))

    # 计算概率
    p_ij = contingency / n_samples
    p_i = np.sum(p_ij, axis=1)  # 真实标签的边缘概率
    p_j = np.sum(p_ij, axis=0)  # 预测标签的边缘概率

    # 计算熵
    def entropy(p):
        p = p[p > 0]  # 避免log(0)
        return -np.sum(p * np.log2(p))

    H_true = entropy(p_i)
    H_pred = entropy(p_j)

    # 计算互信息
    p_ij = p_ij[p_ij > 0]  # 避免log(0)
    MI = np.sum(p_ij * np.log2(p_ij / (np.outer(p_i, p_j).flatten()[p_ij > 0])))

    # 计算VOI
    voi = H_true + H_pred - 2 * MI
    return voi


if __name__ == "__main__":
    from skimage.metrics import variation_of_information as cal_voi
    pred = torch.tensor([[[2,2,2],[2,2,1]]]).flatten()
    gt = torch.tensor([[[0,0,0],[0,0,0]]]).flatten()
    print(cal_voi(np.asarray(pred), np.asarray(gt)), variation_of_information(pred, gt))

# TODO:
# 1. voi 放进分割模型评估部分
# 2. 老问题：为什么有的时候数据预处理后是全零
# 3. 新问题：如何应用POI，图像只针对雄蕊部分做了分割，其余部分虽然有细胞但是没有分割

class Dataset_2D_ninanjie_Train(Dataset):
    def __init__(
            self,
            data_dir='./data/ninanjie',  # 数据的路径
            batch_num='first',
            raw_dir=None,
            label_dir=None,
            split='train',  # 划分方式
            crop_size=None,  # 切割尺寸
            padding_size=8,
            require_lsd=False,
            require_xz_yz=False,
            crop_xyz=None, # 将体块沿着xyz三轴分别均匀分成多少份
            chunk_position=None, # 体块坐标
            **kwargs
    ):

        if crop_xyz is None:
            crop_xyz = [3, 3, 2]
        if chunk_position is None:
            chunk_position = [0, 0, 0]
        self.split = split
        self.crop_size = crop_size
        self.padding_size = padding_size
        self.split = split
        self.require_lsd = require_lsd

        self.images = list()
        self.masks = list()

        # ##Debug
        # data_list = ['trvol-250-1.zarr']
        # raw, labels = _load_images(data_dir, data_list)
        # print(f'unique: raw = {np.unique(raw)}, labels = {np.unique(labels)}')

        # raw_crop = []
        # labels_crop = []
        # lines, rows = (1, 1)
        # cropped_size = (raw.shape[1] // lines, raw.shape[2] // rows)
        # for line in range(lines):
        #     for row in range(rows):
        #         for raw_, labels_ in zip(raw, labels):
        #             raw_crop.append(raw_[line * cropped_size[0]:(line + 1) * cropped_size[0], row * cropped_size[1]:(row + 1) * cropped_size[1]])
        #             labels_crop.append(labels_[line * cropped_size[0]:(line + 1) * cropped_size[0], row * cropped_size[1]:(row + 1) * cropped_size[1]])
        # raw = np.array(raw_crop)
        # labels = label(np.array(labels_crop)).astype(np.uint16)

        # print('raw shape={}, label shape = {}'.format(raw.shape, labels.shape))

        raw, labels = [], []
        raw_dir = os.path.join(data_dir, batch_num, raw_dir if raw_dir else 'raw')
        labels_dir = os.path.join(data_dir, batch_num, label_dir if label_dir else 'label')

        z_start, z_end = chunk_position[2] * len(os.listdir(raw_dir)) // crop_xyz[2],\
                    (chunk_position[2] + 1) * len(os.listdir(raw_dir)) // crop_xyz[2]

        multi_pool = Pool(os.cpu_count() >> 1)
        results = []
        for image in os.listdir(raw_dir)[z_start:z_end]:
            results.append(multi_pool.apply_async(_add_image, args=(raw_dir, labels_dir, image, crop_xyz[:2], chunk_position[:2])))
        pbar = tqdm.tqdm(total=len(results), desc='load images', leave=False)
        for result in results:
            raw_img, label_img = result.get()
            if raw_img and label_img:
                raw.append(raw_img)
                labels.append(label_img)
            pbar.update(1)
        pbar.close()
        multi_pool.close()
        multi_pool.join()

        val_size = min(max(2, len(raw) // 5), 20) # 20% samples as val_set. 20 samples max if total samples over 100.
        raw = np.array(raw)
        labels = np.array(labels)

        print(f'image loaded, now shape: {raw.shape}')

        if self.crop_size is None:
            self.crop_size = min(raw.shape[1] // crop_xyz[0], raw.shape[2] // crop_xyz[1])

        assert split in ['train', 'val'], "invalid split mode"
        if split == 'train':
            self.images.extend(raw[val_size:])
            self.masks.extend(labels[val_size:])
        elif split == 'val':
            self.images.extend(raw[:val_size])
            self.masks.extend(labels[:val_size])

        if require_xz_yz:
            assert split in ['train', 'val']
            if split == 'train':
                img_start, img_end_x, img_end_y = val_size, raw[0].shape[0], raw[0].shape[1]
            elif split == 'val':
                img_start, img_end_x, img_end_y = 0, val_size, val_size

            for n in range(img_start, img_end_x):
                self.images.append(raw[:, n, :])
                self.masks.append(labels[:, n, :])
            for n in range(img_start, img_end_y):
                self.images.append(raw[:, :, n])
                self.masks.append(labels[:, :, n])
            print('require_xz_yz done.')

        self.data_pack = []
        invalid_count = 0

        pbar = tqdm.tqdm(range(len(self.images)), leave=False)
        for idx in pbar:
            sub_data = self.prework(idx)
            if sub_data is not None:
                self.data_pack.append(sub_data)
            else:
                invalid_count += 1
            pbar.set_description(f'preprocessing, {invalid_count} pics invalid')

        # multi_pool = Pool(4)
        # metrics = []
        # for idx in range(len(self.images)):
        #     metrics.append(multi_pool.apply_async(self.prework, args=(idx,)))
        # with tqdm.tqdm(total=len(metrics), desc='preprocessing', leave=False) as pbar:
        #     for result in metrics:
        #         sub_data = result.get()
        #         if sub_data is not None:
        #             self.data_pack.append(sub_data)
        #         else:
        #             invalid_count += 1
        #             # sleep(1)
        #         pbar.set_description(f'preprocessing, invalid: {invalid_count}')
        #         pbar.update(1)
        # multi_pool.close()
        # multi_pool.join()

        # pool = Pool(8)
        # metrics = []
        # pbar = tqdm.tqdm(total=len(self.images))
        # for idx in range(len(self.images)):
        #     metrics.append(pool.apply_async(self.prework, args=(idx,pbar)))
        # pool.close()
        # pool.join()
        # self.data_pack = []
        # for result in tqdm.tqdm(metrics):
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
        Points_pos = list()
        Points_lab = list()
        Boxes = list()
        mask = np.zeros_like(labels, bool)
        p_default = np.random.rand()

        ##
        total_unique_labels = np.unique(labels).tolist()
        point_style = random.choice(['+', '-', '+-'])
        if p_default <= 1:
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

        labels = self.masks[idx]
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


class Dataset_2D_semantic_Train(Dataset):
    def __init__(
            self,
            data_dir='./data/ninanjie/train',  # 数据的路径
            batch_num='second_6',
            raw_dir=None,
            label_dir=None,
            split='train',  # 划分方式
            crop_size=None,  # 切割尺寸
            padding_size=8,
            require_lsd=False,
            require_xz_yz=False,
            crop_xyz=None, # 将体块沿着xyz三轴分别均匀分成多少份
            chunk_position=None, # 体块坐标
    ):

        if crop_xyz is None:
            crop_xyz = [4, 4, 1]
        if chunk_position is None:
            chunk_position = [0, 0, 0]
        self.split = split
        self.crop_size = crop_size
        self.padding_size = padding_size
        self.split = split
        self.require_lsd = require_lsd

        self.images = list()
        self.masks = list()

        # ##Debug
        # data_list = ['trvol-250-1.zarr']
        # raw, labels = _load_images(data_dir, data_list)
        # print(f'unique: raw = {np.unique(raw)}, labels = {np.unique(labels)}')

        # raw_crop = []
        # labels_crop = []
        # lines, rows = (1, 1)
        # cropped_size = (raw.shape[1] // lines, raw.shape[2] // rows)
        # for line in range(lines):
        #     for row in range(rows):
        #         for raw_, labels_ in zip(raw, labels):
        #             raw_crop.append(raw_[line * cropped_size[0]:(line + 1) * cropped_size[0], row * cropped_size[1]:(row + 1) * cropped_size[1]])
        #             labels_crop.append(labels_[line * cropped_size[0]:(line + 1) * cropped_size[0], row * cropped_size[1]:(row + 1) * cropped_size[1]])
        # raw = np.array(raw_crop)
        # labels = label(np.array(labels_crop)).astype(np.uint16)

        # print('raw shape={}, label shape = {}'.format(raw.shape, labels.shape))

        raw, labels = [], []
        raw_dir = os.path.join(data_dir, batch_num, raw_dir if raw_dir else 'raw')
        labels_dir = os.path.join(data_dir, batch_num, label_dir if label_dir else 'label')

        z_start, z_end = chunk_position[2] * len(os.listdir(raw_dir)) // crop_xyz[2],\
                    (chunk_position[2] + 1) * len(os.listdir(raw_dir)) // crop_xyz[2]

        multi_pool = Pool(os.cpu_count() >> 2)
        results = []
        for image in os.listdir(raw_dir)[z_start:z_end]:
            results.append(multi_pool.apply_async(_add_image, args=(raw_dir, labels_dir, image, crop_xyz[:2], chunk_position[:2])))
        pbar = tqdm.tqdm(total=len(results), desc='load images', leave=False)
        for result in results:
            raw_img, label_img = result.get()
            if raw_img and label_img:
                raw.append(raw_img)
                labels.append(label_img)
            pbar.update(1)
        pbar.close()
        multi_pool.close()
        multi_pool.join()

        val_size = min(max(2, len(raw) // 5), 20) # 20% samples as val_set. 20 samples max if total samples over 100.
        raw = np.array(raw)
        labels = np.array(labels)

        print(f'image loaded, now shape: {raw.shape}')

        if self.crop_size is None:
            self.crop_size = min(raw.shape[1] // crop_xyz[0], raw.shape[2] // crop_xyz[1])
            self.crop_size -= self.crop_size % 4

        assert split in ['train', 'val'], "invalid split mode"
        if split == 'train':
            self.images.extend(raw[val_size:])
            self.masks.extend(labels[val_size:])
        elif split == 'val':
            self.images.extend(raw[:val_size])
            self.masks.extend(labels[:val_size])

        if require_xz_yz:
            assert split in ['train', 'val']
            if split == 'train':
                img_start, img_end_x, img_end_y = val_size, raw[0].shape[0], raw[0].shape[1]
            elif split == 'val':
                img_start, img_end_x, img_end_y = 0, val_size, val_size

            for n in range(img_start, img_end_x):
                self.images.append(raw[:, n, :])
                self.masks.append(labels[:, n, :])
            for n in range(img_start, img_end_y):
                self.images.append(raw[:, :, n])
                self.masks.append(labels[:, :, n])
            print('require_xz_yz done.')

        self.data_pack = []
        invalid_count = 0

        pbar = tqdm.tqdm(range(len(self.images)), leave=False)
        for idx in pbar:
            sub_data = self.prework(idx)
            if sub_data is not None:
                self.data_pack.append(sub_data)
            else:
                invalid_count += 1
            pbar.set_description(f'preprocessing, {invalid_count} pics invalid')

        # multi_pool = Pool(4)
        # metrics = []
        # for idx in range(len(self.images)):
        #     metrics.append(multi_pool.apply_async(self.prework, args=(idx,)))
        # with tqdm.tqdm(total=len(metrics), desc='preprocessing', leave=False) as pbar:
        #     for result in metrics:
        #         sub_data = result.get()
        #         if sub_data is not None:
        #             self.data_pack.append(sub_data)
        #         else:
        #             invalid_count += 1
        #             # sleep(1)
        #         pbar.set_description(f'preprocessing, invalid: {invalid_count}')
        #         pbar.update(1)
        # multi_pool.close()
        # multi_pool.join()

        # pool = Pool(8)
        # metrics = []
        # pbar = tqdm.tqdm(total=len(self.images))
        # for idx in range(len(self.images)):
        #     metrics.append(pool.apply_async(self.prework, args=(idx,pbar)))
        # pool.close()
        # pool.join()
        # self.data_pack = []
        # for result in tqdm.tqdm(metrics):
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
        Points_pos = list()
        Points_lab = list()
        Boxes = list()
        mask = np.zeros_like(labels, bool)
        p_default = np.random.rand()

        ##
        total_unique_labels = np.unique(labels).tolist()
        point_style = random.choice(['+', '-', '+-'])
        if p_default <= 1:
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
        raw = self.normalize(raw)  # 所有像素值归一化

        padding = self.get_padding(self.crop_size, self.padding_size)  # padding_size的整数倍，>= crop_size
        labels = self.masks[idx]  if self.split != 'test' else None

        raw, labels = self.augment_data(raw, labels, padding)

        raw = np.expand_dims(raw, axis=0)  # 1* 图像维度

        # if train/val, generate our gt labels
        ## 非测试数据的话，进行一轮边界腐蚀
        labels = self.erode(
            labels,
            iterations=1,  # 腐蚀的迭代次数
            border_value=1)  # 边界值

        affinity = self.getAffinity(labels)

        labels_one_hot = torch.from_numpy(labels).long()
        labels_one_hot = F.one_hot(labels_one_hot, num_classes=4)
        labels_one_hot = np.asarray(labels_one_hot.permute(2, 0, 1))

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
            return raw, labels_one_hot, point_map, mask, affinity, lsds
        else:
            # return raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,affinity
            return raw, labels_one_hot, point_map, mask, affinity

    def __getitem__(self, idx):
        return self.data_pack[idx]


class Dataset_2D_ninanjie_Test(Dataset):
    def __init__(
            self,
            data_dir='./data/ninanjie',  # 数据的路径
            batch_num='first',
            raw_dir='raw',
            label_dir='label',
            crop_size=None,  # 切割尺寸
            padding_size=8,
            crop_xyz=None, # 将体块沿着xyz三轴分别均匀分成多少份
            chunk_position=None, # 体块坐标
            class_num=None
    ):

        if crop_xyz is None:
            crop_xyz = [3, 3, 2]
        if chunk_position is None:
            chunk_position = [0, 0, 0]
        self.crop_size = crop_size
        self.padding_size = padding_size
        self.class_num = class_num
        self.images = list()
        self.labels = list()

        # ##Debug
        # data_list = ['trvol-250-1.zarr']
        # raw, labels = _load_images(data_dir, data_list)
        # print(f'unique: raw = {np.unique(raw)}, labels = {np.unique(labels)}')

        # raw_crop = []
        # labels_crop = []
        # lines, rows = (1, 1)
        # cropped_size = (raw.shape[1] // lines, raw.shape[2] // rows)
        # for line in range(lines):
        #     for row in range(rows):
        #         for raw_, labels_ in zip(raw, labels):
        #             raw_crop.append(raw_[line * cropped_size[0]:(line + 1) * cropped_size[0], row * cropped_size[1]:(row + 1) * cropped_size[1]])
        #             labels_crop.append(labels_[line * cropped_size[0]:(line + 1) * cropped_size[0], row * cropped_size[1]:(row + 1) * cropped_size[1]])
        # raw = np.array(raw_crop)
        # labels = label(np.array(labels_crop)).astype(np.uint16)

        # print('raw shape={}, label shape = {}'.format(raw.shape, labels.shape))

        raw, gt_labels = [], []
        raw_dir = os.path.join(data_dir, batch_num, raw_dir)
        labels_dir = os.path.join(data_dir, batch_num, label_dir)

        z_start, z_end = chunk_position[2] * len(os.listdir(raw_dir)) // crop_xyz[2],\
                    (chunk_position[2] + 1) * len(os.listdir(raw_dir)) // crop_xyz[2]

        raw_shape = None
        for image in tqdm.tqdm(sorted(os.listdir(raw_dir))[z_start:z_end], leave=False):
            cropped_raw, cropped_label = _add_image(raw_dir, labels_dir, image, crop_xyz[:2], chunk_position[:2])
            if cropped_raw is None:
                continue
            assert raw_shape is None or raw_shape == cropped_raw.size
            raw_shape = cropped_raw.size
            raw.append(cropped_raw)
            gt_labels.append(cropped_label)

        raw = np.array(raw)
        gt_labels = np.array(gt_labels)
        print(f'image loaded, now shape: {raw.shape}')

        if self.crop_size is None:
            self.crop_size = min(raw.shape[1] // crop_xyz[0], raw.shape[2] // crop_xyz[1])
            self.crop_size = self.crop_size - self.crop_size % 4

        self.images.extend(raw)
        self.labels.extend(gt_labels)

        self.data_pack = []
        invalid_count = 0
        pbar = tqdm.tqdm(range(len(self.images)), leave=False)
        for idx in pbar:
            sub_data = self.prework(idx)
            if sub_data is not None:
                self.data_pack.append(sub_data)
            else:
                invalid_count += 1
            pbar.set_description(f'preprocessing, invalid: {invalid_count}')

        # multi_pool = Pool(4)
        # metrics = []
        # for idx in range(len(self.images)):
        #     metrics.append(multi_pool.apply_async(self.prework, args=(idx,)))
        # pbar = tqdm.tqdm(total=len(metrics), desc='preprocessing', leave=False)
        # for result in metrics:
        #     sub_data = result.get()
        #     if sub_data is not None:
        #         self.data_pack.append(sub_data)
        #     else:
        #         invalid_count += 1
        #         # sleep(1)
        #     pbar.set_description(f'preprocessing, invalid: {invalid_count}')
        #     pbar.update(1)
        # pbar.close()
        # multi_pool.close()
        # multi_pool.join()

        # pool = Pool(8)
        # metrics = []
        # pbar = tqdm.tqdm(total=len(self.images))
        # for idx in range(len(self.images)):
        #     metrics.append(pool.apply_async(self.prework, args=(idx,pbar)))
        # pool.close()
        # pool.join()
        # self.data_pack = []
        # for result in tqdm.tqdm(metrics):
        #     self.data_pack.append(result.get())
        # pbar.close()


    def __len__(self):
        return len(self.data_pack)

    # takes care of padding
    def get_padding(self, crop_size, padding_size):
        if crop_size is None:
            return None
        # quotient
        q = int(crop_size / padding_size)

        if crop_size % padding_size != 0:
            padding = (padding_size * (q + 1))
        else:
            padding = crop_size

        return padding

        # sample augmentations (see https://albumentations.ai/docs/examples/example_kaggle_salt)

    def augment_data(self, raw, mask, padding):
        if padding:
            y_length, x_length = raw.shape
            x_min, y_min = (x_length - padding) >> 1, (y_length - padding) >> 1
            x_max, y_max = x_min + padding, y_min + padding

            transform = A.Compose([
                A.Crop(
                    x_min=x_min, x_max=x_max,
                    y_min=y_min, y_max=y_max
                ) if padding else None,
                A.PadIfNeeded(
                    min_height=padding,
                    min_width=padding,
                    p=1,
                    border_mode=0)
            ])  ## 数据增强

            transformed = transform(image=raw, mask=mask)
            raw = transformed['image']
            mask = transformed['mask']

        return raw, mask  ## 图像和标签的维度都是： padding * padding

    # normalize raw data between 0 and 1
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)

    def prework(self, idx, pbar=None):
        raw = self.images[idx]  # 获得第idx张图像
        raw = self.normalize(raw)  # 所有像素值归一化

        padding = self.get_padding(self.crop_size, self.padding_size)  # padding_size的整数倍，>= crop_size
        label_ = self.labels[idx]
        if self.class_num:
            label_ = (label_ == self.class_num) + 0
            label_ = label(label_)

        raw, label_ = self.augment_data(raw, label_, padding)
        raw = np.expand_dims(raw, axis=0)  # 1* 图像维度

        point_map = torch.ones(raw.shape, dtype=torch.float32).squeeze()

        if pbar is not None:
            pbar.update()

        return raw, point_map, label_


    def __getitem__(self, idx):
        return self.data_pack[idx]


class Dataset_2D_ninanjie_Origin(Dataset):
    def __init__(
            self,
            data_dir='./data/ninanjie',  # 数据的路径
            raw_dir=None,
            label_dir=None,
            batch_num='first',
            crop_xyz=None, # 将体块沿着xyz三轴分别均匀分成多少份
            chunk_position=None, # 体块坐标
    ):

        if crop_xyz is None:
            crop_xyz = [3, 3, 2]
        if chunk_position is None:
            chunk_position = [0, 0, 0]

        self.images = list()
        self.labels = list()
        self.file_names = list()

        # ##Debug
        # data_list = ['trvol-250-1.zarr']
        # raw, labels = _load_images(data_dir, data_list)
        # print(f'unique: raw = {np.unique(raw)}, labels = {np.unique(labels)}')

        # raw_crop = []
        # labels_crop = []
        # lines, rows = (1, 1)
        # cropped_size = (raw.shape[1] // lines, raw.shape[2] // rows)
        # for line in range(lines):
        #     for row in range(rows):
        #         for raw_, labels_ in zip(raw, labels):
        #             raw_crop.append(raw_[line * cropped_size[0]:(line + 1) * cropped_size[0], row * cropped_size[1]:(row + 1) * cropped_size[1]])
        #             labels_crop.append(labels_[line * cropped_size[0]:(line + 1) * cropped_size[0], row * cropped_size[1]:(row + 1) * cropped_size[1]])
        # raw = np.array(raw_crop)
        # labels = label(np.array(labels_crop)).astype(np.uint16)

        # print('raw shape={}, label shape = {}'.format(raw.shape, labels.shape))

        raw, gt_labels, file_names = [], [], []
        raw_dir = os.path.join(data_dir, batch_num, raw_dir if raw_dir else 'raw')
        labels_dir = os.path.join(data_dir, batch_num, label_dir if label_dir else 'label')

        z_start, z_end = chunk_position[2] * len(os.listdir(raw_dir)) // crop_xyz[2],\
                    (chunk_position[2] + 1) * len(os.listdir(raw_dir)) // crop_xyz[2]

        for image in tqdm.tqdm(sorted(os.listdir(raw_dir))[z_start:z_end], desc='load origin dataset', leave=False):
            cropped_raw, cropped_label = _add_image(raw_dir, labels_dir, image, crop_xyz[:2], chunk_position[:2])
            raw.append(cropped_raw)
            gt_labels.append(cropped_label)
            file_names.append(image)

        raw = np.array(raw)
        gt_labels = np.array(gt_labels)
        print(f'image loaded, now shape: {raw.shape}')

        self.images.extend(raw)
        self.labels.extend(gt_labels)
        self.file_names.extend(file_names)

        self.data_pack = []
        invalid_count = 0
        with tqdm.tqdm(total=len(self.images), leave=False) as pbar:
            for idx, _ in enumerate(self.images):
                sub_data = self.prework(idx)
                if sub_data is not None:
                    self.data_pack.append(sub_data + (self.file_names[idx],))
                else:
                    invalid_count += 1
                pbar.set_description(f'preprocessing, {invalid_count} pic(s) invalid')
                pbar.update(1)

        # multi_pool = Pool(4)
        # metrics = []
        # for idx in range(len(self.images)):
        #     metrics.append(multi_pool.apply_async(self.prework, args=(idx,)))
        # pbar = tqdm.tqdm(total=len(metrics), desc='preprocessing', leave=False)
        # for result in metrics:
        #     sub_data = result.get()
        #     if sub_data is not None:
        #         self.data_pack.append(sub_data)
        #     else:
        #         invalid_count += 1
        #         # sleep(1)
        #     pbar.set_description(f'preprocessing, invalid: {invalid_count}')
        #     pbar.update(1)
        # pbar.close()
        # multi_pool.close()
        # multi_pool.join()

        # pool = Pool(8)
        # metrics = []
        # pbar = tqdm.tqdm(total=len(self.images))
        # for idx in range(len(self.images)):
        #     metrics.append(pool.apply_async(self.prework, args=(idx,pbar)))
        # pool.close()
        # pool.join()
        # self.data_pack = []
        # for result in tqdm.tqdm(metrics):
        #     self.data_pack.append(result.get())
        # pbar.close()


    def __len__(self):
        return len(self.data_pack)

    # normalize raw data between 0 and 1
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)).astype(np.float32)

    def prework(self, idx, pbar=None):
        raw = self.images[idx]  # 获得第idx张图像
        raw = self.normalize(raw)  # 所有像素值归一化
        label = self.labels[idx]

        raw = np.expand_dims(raw, axis=0)  # 1* 图像维度

        point_map = torch.ones(raw.shape, dtype=torch.float32).squeeze()

        if pbar is not None:
            pbar.update()

        return raw, point_map, label


    def __getitem__(self, idx):
        return self.data_pack[idx]



def collate_fn_2D_fib25_Train(batch):
    raw = np.array([item[0] for item in batch]).astype(np.float32)

    labels = np.array([item[1] for item in batch]).astype(np.uint8)

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


def collate_fn_2D_fib25_Test(batch):
    raw = np.array([item[0] for item in batch]).astype(np.float32)
    point_map = np.array([item[1] for item in batch]).astype(np.float32)
    gt_label = np.array([item[2] for item in batch]).astype(np.int32)
    return raw, point_map, gt_label


def collate_fn_2D_ninanjie_Origin(batch):
    raw = np.array([item[0] for item in batch]).astype(np.float32)
    point_map = np.array([item[1] for item in batch]).astype(np.float32)
    gt_label = np.array([item[2] for item in batch]).astype(np.int32)
    file_names = [item[3] for item in batch]
    return raw, point_map, gt_label, file_names


class Dataset_3D_ninanjie_Train(Dataset):
    def __init__(
            self,
            data_dir='./data/ninanjie',  # 数据的路径
            batch_num='first',
            raw_dir=None,
            label_dir=None,
            split='train',  # 划分方式
            crop_size=None,  # 切割尺寸
            num_slices=8,
            padding_size=8,
            require_lsd=False,
            crop_xyz=None,  # 将体块沿着xyz三轴分别均匀分成多少份
            chunk_position=None,  # 体块坐标
            **kwargs):

        if crop_xyz is None:
            crop_xyz = [3, 3, 2]
        if chunk_position is None:
            chunk_position = [0, 0, 0]
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

        raw, labels = [], []
        raw_dir = os.path.join(data_dir, batch_num, raw_dir if raw_dir else 'raw')
        labels_dir = os.path.join(data_dir, batch_num, label_dir if label_dir else 'label')

        z_start, z_end = chunk_position[2] * len(os.listdir(raw_dir)) // crop_xyz[2], \
                         (chunk_position[2] + 1) * len(os.listdir(raw_dir)) // crop_xyz[2]

        # multi_pool = Pool(os.cpu_count() >> 1)
        # results = []
        # for image in os.listdir(raw_dir)[z_start:z_end]:
        #     results.append(
        #         multi_pool.apply_async(_add_image, args=(raw_dir, labels_dir, image, crop_xyz[:2], chunk_position[:2])))
        # for result in results:
        #     raw_img, label_img = result.get()
        #     if raw_img and label_img:
        #         raw.append(raw_img)
        #         labels.append(label_img)
        # multi_pool.close()
        # multi_pool.join()

        for image in os.listdir(raw_dir)[z_start:z_end]:
            raw_img, label_img = _add_image(raw_dir, labels_dir, image, crop_xyz[:2], chunk_position[:2])
            if raw_img and label_img:
                raw.append(raw_img)
                labels.append(label_img)

        val_size = min(max(2, len(raw) // 5), 20)  # 20% samples as val_set. 20 samples max if total samples over 100.
        raw = np.array(raw)
        labels = np.array(labels)

        print(f'image loaded, now shape: {raw.shape}')

        if self.crop_size is None:
            self.crop_size = min(raw.shape[1] // crop_xyz[0], raw.shape[2] // crop_xyz[1])

        assert split in ['train', 'val'], "invalid split mode"
        if split == 'train':
            self.images.extend(raw[val_size:])
            self.masks.extend(labels[val_size:])
        elif split == 'val':
            self.images.extend(raw[:val_size])
            self.masks.extend(labels[:val_size])

        self.images = np.asarray(self.images)
        self.masks = np.asarray(self.masks)

        ##load all 3D patches
        # for idx_slice in range(self.images[0].shape[0] - self.num_slices + 1):
        #     if not np.any(self.masks[idx_slice]):
        #         continue
        #     self.idxs.append(idx_slice)
        self.idxs = [idx_slice
                     for idx_slice in range(self.images.shape[0] - self.num_slices + 1)
                     if np.any(self.masks[idx_slice])]
        self.idxs = self.idxs[::-1]

        self.data_pack = []
        invalid_count = 0
        pbar = tqdm.tqdm(range(len(self.idxs)), leave=False)
        for idx in pbar:
            sub_data = self.prework(idx)
            if sub_data is not None:
                self.data_pack.append(sub_data)
            else:
                invalid_count += 1
            pbar.set_description(f'preprocessing, {invalid_count} pics invalid')

        # self.prework_for_all()

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
        mask_3D = np.zeros_like(labels, bool)

        ##
        total_unique_labels = np.unique(labels[:, :, 0]).tolist()

        ###选择保留哪些神经元, 得到mask
        labels_contain = total_unique_labels

        for label in labels_contain:
            mask_label = (labels == label)

            ##Mask
            mask_3D = np.logical_or(mask_3D, mask_label)

        return mask_3D, None, None

    def generate_gaussian_matrix(self, Points_pos, Points_lab, H, W, theta=10):
        total_matrix = np.ones((H, W))
        return total_matrix

    def prework(self, idx):
        idx_slice = self.idxs[idx]

        raw = self.images[idx_slice:(idx_slice + self.num_slices)]
        labels = self.masks[idx_slice:(idx_slice + self.num_slices)]
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
        # point_map = np.ones((self.crop_size, self.crop_size))

        if self.require_lsd:
            return raw, labels, mask_3D, affinity, point_map, lsds
        else:
            return raw, labels, mask_3D, affinity, point_map

    def prework_for_all(self):
        raw = self.images
        labels = self.masks
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
        # point_map = np.ones((self.crop_size, self.crop_size))

        if self.require_lsd:
            self.data_pack = (raw, labels, mask_3D, affinity, point_map, lsds)
        else:
            self.data_pack = (raw, labels, mask_3D, affinity, point_map)

    def __getitem__(self, idx):
        return self.data_pack[idx]
        # return [mat[..., idx: idx + self.num_slices] if mat is not None else None
        #         for mat in self.data_pack
        #         ]

    def get_item(self, idx):
        return (mat[..., idx : idx+self.num_slices] if mat else None
                for mat in self.data_pack
        )


class Dataset_3D_ninanjie_Train_GPU(Dataset):
    def __init__(
            self,
            data_dir='./data/ninanjie',
            batch_num='first',
            raw_dir=None,
            label_dir=None,
            split='train',
            crop_size=None,
            num_slices=8,
            padding_size=8,
            require_lsd=False,
            crop_xyz=None,
            chunk_position=None,
            **kwargs):

        if crop_xyz is None:
            crop_xyz = [3, 3, 2]
        if chunk_position is None:
            chunk_position = [0, 0, 0]
        self.split = split
        self.crop_size = crop_size
        self.num_slices = num_slices
        self.padding_size = padding_size
        self.require_lsd = require_lsd
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载原始数据
        self.images = []
        self.masks = []
        self.idxs = []

        raw_dir = os.path.join(data_dir, batch_num, raw_dir if raw_dir else 'raw')
        labels_dir = os.path.join(data_dir, batch_num, label_dir if label_dir else 'label')

        # 计算z轴切割范围
        z_files = sorted(os.listdir(raw_dir))
        z_start = chunk_position[2] * len(z_files) // crop_xyz[2]
        z_end = (chunk_position[2] + 1) * len(z_files) // crop_xyz[2]
        z_files = z_files[z_start:z_end]

        # 多进程加载图像
        with Pool(os.cpu_count() >> 1) as multi_pool:
            results = []
            for image in z_files:
                results.append(
                    multi_pool.apply_async(_add_image,
                                           args=(raw_dir, labels_dir, image, crop_xyz[:2], chunk_position[:2])))

            for result in results:
                raw_img, label_img = result.get()
                if raw_img is not None and label_img is not None:
                    self.images.append(raw_img)
                    self.masks.append(label_img)

        # 转换为numpy数组并分割训练/验证集
        self.images = np.asarray(self.images)  # 形状: (N, H, W)
        self.masks = np.asarray(self.masks, dtype=np.int32)  # 形状: (N, H, W)
        print(f'原始数据形状: {self.images.shape}, 标签形状: {self.masks.shape}')

        # 分割训练/验证集
        val_size = min(max(2, len(self.images) // 5), 20)
        if split == 'train':
            self.images = self.images[val_size:]
            self.masks = self.masks[val_size:]
        else:  # val
            self.images = self.images[:val_size]
            self.masks = self.masks[:val_size]

        # 预处理并转移到显存
        self._preprocess_and_move_to_gpu()

        # 生成有效的切片索引（只保留有标注的切片）
        self.idxs = [
            idx for idx in range(self.images.shape[0] - self.num_slices + 1)
            if np.any(self.masks[idx:idx + self.num_slices])
        ]
        if not self.idxs:
            raise ValueError("没有有效的训练样本，请检查数据或调整num_slices")

    def _preprocess_and_move_to_gpu(self):
        """预处理数据并将其转移到GPU显存"""
        pbar = tqdm.tqdm(total=7, leave=False)
        # 转置为 (H, W, N) 并归一化
        raw = self.images.transpose(1, 2, 0)  # (H, W, N)
        raw = self.normalize(raw)
        pbar.set_description("raw normalized")
        pbar.update(1)

        # 处理标签
        labels = self.masks.transpose(1, 2, 0)  # (H, W, N)
        labels = self.erode(labels, iterations=1, border_value=1)
        pbar.set_description("label eroded")
        pbar.update(1)

        # 数据增强
        padding = self.get_padding(self.crop_size, self.padding_size)
        raw, labels = self.augment_data(raw, labels, padding)
        pbar.set_description("raw & label augmented")
        pbar.update(1)

        # 预计算亲和力矩阵 (如果需要)
        self.affinity = self._compute_affinity(labels)
        self.affinity = torch.tensor(self.affinity, dtype=torch.float32, device=self.device)
        pbar.set_description("affinity generated")
        pbar.update(1)

        # 预计算mask和点图
        self.mask_3D, _, _ = self.get_Mask(labels)
        self.mask_3D = torch.tensor(self.mask_3D, dtype=torch.uint8, device=self.device)

        self.point_map = torch.tensor(
            self.generate_gaussian_matrix(None, None, self.crop_size, self.crop_size, theta=30),
            dtype=torch.float32, device=self.device
        )
        pbar.set_description("point_map generated")
        pbar.update(1)

        # 处理LSD特征 (如果需要)
        self.lsds = None
        if self.require_lsd:
            lsds = local_shape_descriptor.get_local_shape_descriptors(
                segmentation=labels,
                sigma=(5,) * 3,
                voxel_size=(1,) * 3
            )
            self.lsds = torch.tensor(lsds, dtype=torch.float32, device=self.device)
        pbar.set_description("lsd created")
        pbar.update(1)

        # 转换为torch张量并转移到GPU
        raw = torch.tensor(raw, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.int32, device=self.device)
        self.raw, self.labels = raw.unsqueeze(0), labels.unsqueeze(0)
        pbar.set_description("raw & label to cuda")
        pbar.update(1)
        pbar.close()

    def _compute_affinity(self, labels):
        """预先计算亲和力矩阵"""
        label_shift = np.pad(labels, ((0, 1), (0, 1), (0, 1)), 'edge')

        affinity_x = np.expand_dims(((labels - label_shift[1:, :-1, :-1]) != 0) + 0, axis=0)
        affinity_y = np.expand_dims(((labels - label_shift[:-1, 1:, :-1]) != 0) + 0, axis=0)
        affinity_z = np.expand_dims(((labels - label_shift[:-1, :-1, 1:]) != 0) + 0, axis=0)

        background = (labels == 0)
        affinity_x[0][background] = 1
        affinity_y[0][background] = 1
        affinity_z[0][background] = 1

        return np.concatenate([affinity_x, affinity_y, affinity_z], axis=0).astype('float32')

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        """直接从显存切片获取数据，避免重复存储"""
        start = self.idxs[idx]
        end = start + self.num_slices

        # 切片获取8层数据 (H, W, [start:end])
        raw_slice = self.raw[..., start:end]

        # 标签和其他数据同样切片
        labels_slice = self.labels[..., start:end]
        mask_slice = self.mask_3D[..., start:end]
        affinity_slice = self.affinity[..., start:end]
        point_map_slice = self.point_map

        # 组装返回数据
        result = [
            raw_slice,
            labels_slice,
            mask_slice,
            affinity_slice,
            point_map_slice
        ]

        # 如果需要LSD特征
        if self.require_lsd:
            result.append(self.lsds[..., start:end])

        return result

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

    def augment_data_gpu(self, raw, mask, padding):
        """GPU上的数据增强（完全匹配albumentations原功能）"""
        # 原始数据形状: raw为(1, H, W, D), mask为(H, W, D)
        # 1. 随机裁剪到crop_size（与A.RandomCrop一致）
        raw, mask = raw.unsqueeze(0), mask.unsqueeze(0)
        h, w = raw.shape[1], raw.shape[2]
        if h > self.crop_size and w > self.crop_size:
            # 随机生成裁剪起点
            top = torch.randint(0, h - self.crop_size, (1,), device=self.device).item()
            left = torch.randint(0, w - self.crop_size, (1,), device=self.device).item()
            # 执行裁剪
            raw = raw[:, top:top + self.crop_size, left:left + self.crop_size, :]
            mask = mask[top:top + self.crop_size, left:left + self.crop_size, :]

        # 2. 填充到padding大小（与A.PadIfNeeded一致）
        current_h, current_w = raw.shape[1], raw.shape[2]
        pad_h = max(0, padding - current_h)
        pad_w = max(0, padding - current_w)

        if pad_h > 0 or pad_w > 0:
            # 计算上下左右填充量（与albumentations默认相同，均匀填充）
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # 对raw和mask进行填充（border_mode=0对应常数填充，默认为0）
            raw = torch.nn.functional.pad(
                raw,
                (0, 0, pad_left, pad_right, pad_top, pad_bottom, 0, 0),  # 注意PyTorch的pad顺序是反向的
                mode='constant',
                value=0.0
            )
            mask = torch.nn.functional.pad(
                mask,
                (0, 0, pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=0
            )

        # 3. 随机水平翻转（与A.HorizontalFlip一致）
        if torch.rand(1, device=self.device) < 0.3:
            raw = torch.flip(raw, dims=[2])  # 水平翻转W维度
            mask = torch.flip(mask, dims=[2])

        # 4. 随机垂直翻转（与A.VerticalFlip一致）
        if torch.rand(1, device=self.device) < 0.3:
            raw = torch.flip(raw, dims=[1])  # 垂直翻转H维度
            mask = torch.flip(mask, dims=[1])

        # 5. 随机旋转90度（与A.RandomRotate90一致）
        if torch.rand(1, device=self.device) < 0.3:
            # 随机选择旋转次数（0-3次，每次90度）
            k = torch.randint(1, 4, (1,), device=self.device).item()
            raw = torch.rot90(raw, k=k, dims=[1, 2])  # 在H和W维度旋转
            mask = torch.rot90(mask, k=k, dims=[0, 1])

        # 6. 随机转置（与A.Transpose一致）
        if torch.rand(1, device=self.device) < 0.3:
            raw = torch.transpose(raw, 1, 2)  # 交换H和W维度
            mask = torch.transpose(mask, 0, 1)

        # 7. 随机亮度对比度调整（与A.RandomBrightnessContrast一致）
        if torch.rand(1, device=self.device) < 0.3:
            # 亮度调整范围: [-0.2, 0.2]，对比度调整范围: [-0.2, 0.2]（与albumentations默认一致）
            brightness_factor = 1.0 + torch.randn(1, device=self.device) * 0.1  # 更集中的分布
            brightness_factor = torch.clamp(brightness_factor, 0.8, 1.2)

            contrast_factor = 1.0 + torch.randn(1, device=self.device) * 0.1
            contrast_factor = torch.clamp(contrast_factor, 0.8, 1.2)

            # 先调整亮度
            raw = raw * brightness_factor
            # 再调整对比度（使用均值中心化）
            mean = raw.mean()
            raw = (raw - mean) * contrast_factor + mean
            # 确保值仍在[0,1]范围内（归一化后）
            raw = torch.clamp(raw, 0.0, 1.0)

        return raw, mask

    # 以下是其他辅助方法（保持不变）
    def erode(self, labels, iterations, border_value):
        foreground = np.zeros_like(labels, dtype=bool)
        for label in np.unique(labels):
            if label == 0:
                continue
            label_mask = labels == label
            eroded_mask = binary_erosion(label_mask, iterations=iterations, border_value=border_value)
            foreground = np.logical_or(eroded_mask, foreground)
        background = np.logical_not(foreground)
        labels[background] = 0
        return labels

    def get_padding(self, crop_size, padding_size):
        q = int(crop_size / padding_size)
        return padding_size * (q + 1) if crop_size % padding_size != 0 else crop_size

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8).astype(np.float32)

    def get_Mask(self, labels):
        mask_3D = np.zeros_like(labels, bool)
        total_unique_labels = np.unique(labels[:, :, 0]).tolist()
        for label in total_unique_labels:
            if label == 0:
                continue
            mask_label = (labels == label)
            mask_3D = np.logical_or(mask_3D, mask_label)
        return mask_3D, None, None

    def generate_gaussian_matrix(self, Points_pos, Points_lab, H, W, theta=10):
        return np.ones((H, W), dtype=np.float32)


class Dataset_3D_semantic_Train(Dataset):
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
    # raw = np.array([item[0] for item in batch]).astype(np.float32)  # 注意normalize了，这里要用float
    #
    # labels = np.array([item[1] for item in batch]).astype(np.int32)
    #
    # mask_3D = np.array([item[2] for item in batch]).astype(np.uint8)
    #
    # affinity = np.array([item[3] for item in batch]).astype(np.float32)
    #
    # point_map = np.array([item[4] for item in batch]).astype(np.float32)
    #
    # # lsds = np.array([item[5] for item in batch]).astype(np.float32)
    # if len(batch[0]) == 6:
    #     lsds = np.array([item[5] for item in batch]).astype(np.float32)
    #     return raw, labels, mask_3D, affinity, point_map, lsds
    # else:
    #     return raw, labels, mask_3D, affinity, point_map
    """直接在GPU上拼接数据，避免内存传输"""
    # 从第一个元素获取数据结构
    num_elements = len(batch[0])

    # 拼接每个元素
    collated = []
    for i in range(num_elements):
        # 直接在GPU上拼接张量
        collated.append(torch.stack([item[i] for item in batch], dim=0))

    return collated

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
