import logging
import os
import random
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, precision, recall, f1_score
from tqdm.auto import tqdm

# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from utils.dataloader_ninanjie import Dataset_2D_ninanjie_Train, load_dataset, collate_fn_2D_fib25_Train
from utils.dataloader_hemi_better import Dataset_2D_hemi_Train, collate_fn_2D_hemi_Train

## CUDA_VISIBLE_DEVICES=0 python main_segEM_2d_train_zebrafinch.py &

WEIGHT_LOSS2 = 2
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

        self.lsd_predict = torch.nn.Conv2d(in_channels=12, out_channels=6, kernel_size=1)  # 最终输出层的卷积操作

        # create our network, 6 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet2d(
            in_channels=7,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.affinity_predict = torch.nn.Conv2d(in_channels=12, out_channels=2, kernel_size=1)  # 最终输出层的卷积操作

    def forward(self, x):
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x, y_lsds.detach()], dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds, y_affinity


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') # 去掉 'module.' 前缀
        new_state_dict[name] = v
    return new_state_dict


####ACRLSD模型
class segEM2d(torch.nn.Module):
    def __init__(
            self,
    ):
        super(segEM2d, self).__init__()

        ##For affinity prediction
        self.model_affinity = ACRLSD()
        # model_path = './output/checkpoints/ACRLSD_2D(hemi+fib25+cremi)_Best_in_val.model'
        model_path = ('/home/liuhongyu2024/sshfs_share/liuhongyu2024/project/unispac/UniSPAC-edited/training/'
                      'output/checkpoints/ACRLSD_2D(ninanjie)_origin_Best_in_val.model')
        weights = torch.load(model_path, map_location=torch.device('cuda'))
        self.model_affinity.load_state_dict(remove_module(weights))
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 2 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet2d(
            in_channels=3,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.mask_predict = torch.nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1)  # 最终输出层的卷积操作

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
    # forward
    # lsd_logits,affinity_logits = model(raw)
    y_mask, y_lsds, y_affinity = model(input_image, input_prompt)




if __name__ == '__main__':
    ##设置超参数
    batch_size = 32
    Save_Name = 'segEM2d(ninanjie)-w2-{}-w3-{}'.format(WEIGHT_LOSS2, WEIGHT_LOSS3)

    set_seed()
    device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = segEM2d()

    ###创建模型
    #多卡训练
    # 一机多卡设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '2' #设置所有可以使用的显卡，共计四块
    device_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]  # 选中显卡
    # set device
    model = nn.DataParallel(model.cuda(), device_ids=device_ids, output_device=device_ids[0])#并行使用两块

    test_dataset_1 = load_dataset('ninanjie_val.joblib', 'val', require_xz_yz=False, from_temp=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=48, pin_memory=True,
    #                           drop_last=True, collate_fn=collate_fn_2D_fib25_Train)
    test_loader = DataLoader(test_dataset_1, batch_size=(batch_size // 2) + 1, shuffle=False, num_workers=48, pin_memory=True,
                            collate_fn=collate_fn_2D_fib25_Train)

    def load_data_to_device(loader):
        tmp_loader = iter(loader)
        res = []
        # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
        for raw, labels, point_map, mask, gt_affinity, _ in tqdm(tmp_loader, leave=True):
            ##Get Tensor
            raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
            point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)  # (batch, height, width)
            mask = torch.as_tensor(mask, dtype=torch.float, device=device)  # (batch, 1, height, width)
            gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                                          device=device)  # (batch, 2, height, width)
            res.append([raw, labels, point_map, mask, gt_affinity])
        return res

    test_loader = load_data_to_device(test_loader)

    # set activation
    activation = torch.nn.Sigmoid()

    model.eval()
    ##Fix validation set
    seed = 98
    np.random.seed(seed)
    random.seed(seed)
    acc_loss = []
    count, total = 0, len(test_loader)
    # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
    with torch.no_grad():
        for raw, labels, point_map, mask, gt_affinity in tqdm(test_loader, leave=True):
            loss_value, _, results = model_step(model, optimizer, raw, point_map, mask, gt_affinity, activation,
                                                train_step=False)
