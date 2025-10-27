import os
import random
from multiprocessing.pool import Pool

import albumentations as A  ##一种数据增强库
import joblib
import numpy as np
import torch
import tqdm
from PIL import Image
from scipy.ndimage import binary_erosion
# import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# import h5py
from torch.utils.data import Dataset

ninanjie_data = '/home/liuhongyu2024/Documents/UniSPAC-edited/data/ninanjie'
ninanjie_save = '/home/liuhongyu2024/Documents/UniSPAC-edited/data/ninanjie-save'

def load_train_dataset(dataset_name: str, raw_dir=None, label_dir=None, from_temp=True, require_lsd=True, require_xz_yz=True,
                 crop_size=None, crop_xyz=None, chunk_position=None):
    temp_name = (f'{dataset_name}_{from_temp}_{require_lsd}_{require_xz_yz}_{crop_size}_'
                 f'{"".join((str(i) for i in crop_xyz))}_{"".join((str(i) for i in chunk_position))}_full')

    DATASET = Dataset_2D_ninanjie_Train
    dataset_name = dataset_name.replace('_3d', '')
    dataset_name = dataset_name.replace('_cpu', '')
    if os.path.exists(os.path.join(ninanjie_save, temp_name)) and from_temp:
        print(f'load {temp_name} from disk')
        train_dataset, val_dataset = joblib.load(os.path.join(ninanjie_save, temp_name))
        # train_dataset, val_dataset = [], []
    else:
        print(f'prepare {temp_name} from disk')
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


def _add_image(raw_dir_, labels_dir_, image_, crop_xy_, chunk_pos_, cellmask_dir_=None, probs_dir_=None):
    try:
        raw_img = Image.open(os.path.join(raw_dir_, image_))
        # label_img_ = str(int(image_.split('.')[0]) - 1).zfill(4) + '.tif'
        label_img = Image.open(os.path.join(labels_dir_, image_)) if labels_dir_ else None
        cellmask_img = Image.open(os.path.join(cellmask_dir_, image_)) if cellmask_dir_ else None
        probs_img = Image.open(os.path.join(probs_dir_, image_)) if probs_dir_ else None
        x_, y_ = raw_img.size
    except FileNotFoundError as e:
        size = 2 + (1 if cellmask_dir_ else 0) + (1 if probs_dir_ else 0)
        return [None] * size
    left, right = chunk_pos_[0] * x_ // crop_xy_[0], (chunk_pos_[0]+1) * x_ // crop_xy_[0]
    upper, lower = chunk_pos_[1] * y_ // crop_xy_[1], (chunk_pos_[1]+1) * y_ // crop_xy_[1]

    cropped_raw = raw_img.crop((left, upper, right, lower))
    cropped_label = label_img.crop((left, upper, right, lower)) if labels_dir_ else None
    cropped_cellmask = cellmask_img.crop((left, upper, right, lower)) if cellmask_dir_ else None
    cropped_probs = probs_img.crop((left, upper, right, lower)) if probs_dir_ else None

    res = [cropped_raw, cropped_label]
    if cellmask_dir_:
        res.append(cropped_cellmask)
    if probs_dir_:
        res.append(cropped_probs)
    return res


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
        self.probs = list()

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
        prob_dir = os.path.join(data_dir, batch_num, label_dir if label_dir else 'train_probability_tz')

        z_start, z_end = chunk_position[2] * len(os.listdir(raw_dir)) // crop_xyz[2],\
                    (chunk_position[2] + 1) * len(os.listdir(raw_dir)) // crop_xyz[2]

        multi_pool = Pool(os.cpu_count() >> 1)
        results = []
        for image in os.listdir(raw_dir)[z_start:z_end]:
            results.append(multi_pool.apply_async(_add_image, args=(raw_dir, labels_dir, image, crop_xyz[:2], chunk_position[:2],
                                                                    None, prob_dir)))
        pbar = tqdm.tqdm(total=len(results), desc='load images', leave=False)
        for result in results:
            raw_img, label_img, prob_img = result.get()
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
        invalid_count, prompt_count = 0, 0

        pbar = tqdm.tqdm(range(len(self.images)), leave=False)
        for idx in pbar:
            _result = self.prework(idx)
            is_prompt = False
            if _result is not None:
                sub_data, is_prompt = _result
                self.data_pack.append(sub_data)
            else:
                invalid_count += 1
            prompt_count += 1 if is_prompt else 0
            pbar.set_description(f'preprocessing, {invalid_count} pics invalid, {prompt_count} pics with prompt')

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

        return raw, mask ## 图像和标签的维度都是： padding * padding

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
        if p_default < 1:
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

        background = (labels == 0)
        affinity_x[:, background] = 1
        affinity_y[:, background] = 1

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
            iterations=2,  # 腐蚀的迭代次数
            border_value=4)  # 边界值

        affinity = self.getAffinity(labels)

        # if len(np.unique(labels)) == 1:
        #     if pbar is not None:
        #         pbar.update()
        #     return None

        Points_pos, Points_lab, Boxes, mask = self.get_prompt(labels)
        point_map = self.generate_gaussian_matrix(Points_pos, Points_lab, self.crop_size, self.crop_size, theta=30)
        if pbar is not None:
            pbar.update()

        return [raw, labels, point_map, mask, affinity, None], Points_pos is not None


    def __getitem__(self, idx):
        return self.data_pack[idx]

class Dataset_2D_ninanjie_Origin(Dataset):
    def __init__(
            self,
            data_dir='./data/ninanjie',  # 数据的路径
            raw_dir=None,
            label_dir=None,
            crop_xyz=None, # 将体块沿着xyz三轴分别均匀分成多少份
            chunk_position=None, # 体块坐标
    ):

        if crop_xyz is None:
            crop_xyz = [1, 1, 1]
        if chunk_position is None:
            chunk_position = [0, 0, 0]

        self.images = list()
        self.labels = list()
        self.probs = list()
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

        raw, gt_labels, probs, file_names = [], [], [], []
        raw_dir = os.path.join(data_dir, raw_dir if raw_dir else 'raw')
        labels_dir = os.path.join(data_dir, label_dir if label_dir else 'label')
        prob_dir = os.path.join(data_dir, 'prob')

        z_start, z_end = chunk_position[2] * len(os.listdir(raw_dir)) // crop_xyz[2],\
                    (chunk_position[2] + 1) * len(os.listdir(raw_dir)) // crop_xyz[2]

        for image in tqdm.tqdm(sorted(os.listdir(raw_dir))[z_start:z_end], desc='load origin dataset', leave=False):
            cropped_raw, cropped_label, prob = _add_image(raw_dir, labels_dir, image, crop_xyz[:2], chunk_position[:2],
                                                    probs_dir_=prob_dir)
            raw.append(cropped_raw)
            gt_labels.append(cropped_label)
            probs.append(prob)
            file_names.append(image)

        raw = np.array(raw)
        gt_labels = np.array(gt_labels)
        probs = np.array(probs)
        print(f'image loaded, now shape: {raw.shape}')

        self.images.extend(raw)
        self.labels.extend(gt_labels)
        self.probs.extend(probs)
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
        prob = self.probs[idx]

        crop_size = 4032
        H, W = raw.shape
        start_x, start_y = (H - crop_size) >> 1, (H + crop_size) >> 1
        start_a, start_b = (W - crop_size) >> 1, (W + crop_size) >> 1
        raw = raw[start_x:start_y, start_a:start_b]
        label = np.asarray(label[start_x:start_y, start_a:start_b])
        prob = np.asarray(prob[start_x:start_y, start_a:start_b])

        # raw = np.expand_dims(raw, axis=0)  # 1* 图像维度

        point_map = torch.ones(raw.shape, dtype=torch.float32).squeeze()

        if pbar is not None:
            pbar.update()

        return raw, point_map, label, prob


    def __getitem__(self, idx):
        return self.data_pack[idx]

def collate_fn_2D_ninanjie_Origin(batch):
    raw = np.array([item[0] for item in batch]).astype(np.float32)
    point_map = np.array([item[1] for item in batch]).astype(np.float32)
    gt_label = np.array([item[2] for item in batch]).astype(np.int32)
    file_names = [item[3] for item in batch]
    return raw, point_map, gt_label, file_names
