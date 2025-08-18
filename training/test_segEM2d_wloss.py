import logging
import multiprocessing
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.constants import precision
from skimage.measure import label
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from training.utils.dataloader_fib25_better import Dataset_2D_fib25_Origin
from utils.dataloader_fib25_better import Dataset_2D_fib25_Train, Dataset_2D_fib25_Test, collate_fn_2D_fib25_Test
from utils.dataloader_hemi_better import Dataset_2D_hemi_Train, collate_fn_2D_hemi_Train

## CUDA_VISIBLE_DEVICES=0 python main_segEM_2d_train_zebrafinch.py &

HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

WEIGHT_LOSS3 = 1


def set_seed(seed=1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True


####ACRLSD模型
class ACRLSD(torch.nn.Module):
    def __init__(
            self,
    ):
        super(ACRLSD, self).__init__()

        # create our network, 1 input channels in the raw data
        self.model_lsds = UNet2d(
            in_channels=1,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.lsd_predict = torch.nn.Conv2d(in_channels=12, out_channels=6, kernel_size=1).to(device)  # 最终输出层的卷积操作

        # create our network, 6 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet2d(
            in_channels=7,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.affinity_predict = torch.nn.Conv2d(in_channels=12, out_channels=2, kernel_size=1).to(device)  # 最终输出层的卷积操作

    def forward(self, x):
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x, y_lsds.detach()], dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds, y_affinity


####ACRLSD模型
class segEM2d(torch.nn.Module):
    def __init__(
            self,
    ):
        super(segEM2d, self).__init__()

        ##For affinity prediction
        self.model_affinity = ACRLSD().to(device)
        # model_path = './output/checkpoints/ACRLSD_2D(hemi+fib25+cremi)_Best_in_val.model' 
        model_path = os.path.join(HOME_PATH, 'checkpoints/ACRLSD_2D(hemi+fib25)_Best_in_val.model')
        weights = torch.load(model_path, map_location=device)
        self.model_affinity.load_state_dict(weights)
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 2 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet2d(
            in_channels=3,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.mask_predict = torch.nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1).to(device)  # 最终输出层的卷积操作

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_raw, x_prompt):
        y_lsds, y_affinity = self.model_affinity(x_raw)

        y_concat = torch.cat([x_prompt.unsqueeze(1), y_affinity.detach()], dim=1)

        y_mask = self.mask_predict(self.model_mask(y_concat))
        y_mask = self.sigmoid(y_mask)

        return y_mask, y_lsds, y_affinity


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


def model_step(model, input_image, input_prompt, activation):
    y_mask, _, _ = model(input_image, input_prompt)
    # y_pred = activation(y_pred)

    return y_mask


def visualize_and_save_mask(raw, segmentation, idx=0, mode='origin',
                            background_color=(0, 0, 0), seed=98):
    # 获取所有实例标签（排除背景0）
    output_path = os.path.join(HOME_PATH, 'data/fib25/test', f"{idx}_{mode}.png")

    # 获取所有实例标签（排除背景0）
    instances = np.unique(segmentation)
    instances = instances[instances != 0]
    num_instances = len(instances)

    # 设置随机种子，确保颜色分配一致
    np.random.seed(seed)

    # 为每个实例生成随机颜色（RGB格式，值在0-1之间）
    # 避免过暗颜色，提高可视性
    colors = np.random.rand(num_instances, 3)
    colors = np.clip(colors, 0.2, 0.9)  # 限制颜色亮度范围

    # 创建颜色映射：索引0对应背景，其余对应各个实例
    # 颜色映射的索引需要与segmentation中的标签值对应
    max_label = int(np.max(segmentation)) if num_instances > 0 else 0
    cmap_colors = [background_color] * (max_label + 1)  # 初始化所有标签颜色

    # 为每个实例标签分配颜色
    for i, label in enumerate(instances):
        cmap_colors[label] = tuple(colors[i])

    # 创建自定义颜色映射
    cmap = ListedColormap(cmap_colors)

    # 绘制并保存图像
    plt.figure(figsize=(10, 10))
    # 显示原始图像
    plt.imshow(raw, cmap='gray' if raw.ndim == 2 else None)
    # 显示分割结果（仅显示非背景区域）
    masked_segmentation = np.ma.masked_where(segmentation == 0, segmentation)
    plt.imshow(masked_segmentation, cmap=cmap, alpha=0.5)  # alpha控制透明度
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout(pad=0)  # 去除边距

    # 保存图像，不包含白边
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    return 1


def visualize_dataset(raw, labels):
    pool = multiprocessing.Pool(os.cpu_count() >> 2)
    processes = []
    for single_img, single_mask, idx in zip(raw, labels, range(len(raw))):
        single_img = single_img.squeeze()
        single_mask = single_mask.squeeze()
        processes.append(pool.apply_async(visualize_and_save_mask,
                                          args=(single_img, single_mask, idx + count * batch_size, 'origin')))
        segmentation = label(single_mask[:, :])
        # segmentation_color = create_lut(segmentation)
        processes.append(pool.apply_async(visualize_and_save_mask,
                                          args=(single_img, segmentation, idx + count * batch_size, 'segment')))
    with tqdm(total=len(processes), leave=False) as pbar:
        for process in processes:
            res = process.get()
            pbar.update(res)
    pool.close()
    pool.join()


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 1000
    learning_rate = 1e-4
    batch_size = 256

    set_seed()

    ###创建模型
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,5'  # 设置所有可以使用的显卡，共计四块
    device_ids = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]  # 选中显卡
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # load trained model weights
    model_path = os.path.join(HOME_PATH, 'checkpoints/segEM2d(hemi+fib25)wloss-1_Best_in_val.model')
    model = segEM2d().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device)

    ##装载数据
    fib25_data = os.path.join(HOME_PATH, 'data/fib25')
    if os.path.exists(os.path.join(fib25_data, 'fib25_test.joblib')):
        print("Load data from disk...")
        test_dataset = joblib.load(os.path.join(fib25_data, 'fib25_test.joblib'))
    else:
        test_dataset = Dataset_2D_fib25_Test(data_dir=os.path.join(fib25_data, 'training'), crop_size=240, padding_size=8)
        joblib.dump(test_dataset, os.path.join(fib25_data, 'fib25_test.joblib'))


    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=48, pin_memory=True,
                              drop_last=False, collate_fn=collate_fn_2D_fib25_Test)

    def load_data_to_device(loader):
        tmp_loader = iter(loader)
        res = []
        # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
        for raw, point_map in tqdm(tmp_loader, leave=True):
            ##Get Tensor
            raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
            point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)  # (batch, height, width)
            res.append([raw, point_map])
        return res

    test_loader = load_data_to_device(test_loader)

    ##开始训练
    # set activation
    activation = torch.nn.Sigmoid().to(device)

    # training loop
    model.eval()
    ##Fix validation set
    seed = 98
    np.random.seed(seed)
    random.seed(seed)
    count, total = 0, len(test_loader)
    # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
    for raw, point_map in tqdm(test_loader, desc='saving'):
        with torch.no_grad():
            results: torch.Tensor = model_step(model, raw, point_map, activation).detach().cpu()
            results: np.ndarray = (np.asarray(results) > 0.5) + 0
            raw = np.asarray(raw.detach().cpu())
        # for single_img, single_prob, idx in zip(raw, metrics, range(len(raw))):
        #     single_img = single_img.squeeze()
        #     single_prob = single_prob.squeeze()
        #     visualize_and_save_mask(single_img, single_prob, idx + count * batch_size)
        pool = multiprocessing.Pool(os.cpu_count() >> 2)
        processes = []
        for single_img, single_mask, idx in zip(raw, results, range(len(raw))):
            single_img = single_img.squeeze()
            single_mask = single_mask.squeeze()
            processes.append(pool.apply_async(visualize_and_save_mask,
                                              args=(single_img, single_mask, idx + count * batch_size, 'origin')))
            segmentation = label(single_mask[:, :])
            # segmentation_color = create_lut(segmentation)
            processes.append(pool.apply_async(visualize_and_save_mask,
                                              args=(single_img, segmentation, idx + count * batch_size, 'segment')))
        with tqdm(total=len(processes), leave=False) as pbar:
            for process in processes:
                res = process.get()
                pbar.update(res)
        pool.close()
        pool.join()
        count += 1

