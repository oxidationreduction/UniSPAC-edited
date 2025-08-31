import logging
import multiprocessing.pool
import os
import random
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
# from torchmetrics.functional import accuracy, precision, recall, f1_score
from tqdm.auto import tqdm
from skimage.measure import label

# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from training.test_segEM2d_wloss import HOME_PATH
from training.utils.aftercare import calculate_multi_class_voi, visualize_and_save_mask
from training.utils.dataloader_ninanjie import load_test_dataset, collate_fn_2D_fib25_Test
from matplotlib.colors import ListedColormap

## CUDA_VISIBLE_DEVICES=0 python main_segEM_2d_train_zebrafinch.py &

HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


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


class segEM2d(torch.nn.Module):
    def __init__(
            self,
    ):
        super(segEM2d, self).__init__()

        ##For affinity prediction
        self.model_affinity = ACRLSD()
        # model_path = './output/checkpoints/ACRLSD_2D(hemi+fib25+cremi)_Best_in_val.model'
        model_path = ('/home/liuhongyu2024/Documents/UniSPAC-edited/'
                      'training/output/checkpoints/ACRLSD_2D(ninanjie)_semantic_Best_in_val.model')
        weights = torch.load(model_path, map_location=torch.device('cuda'))
        self.model_affinity.load_state_dict(remove_module(weights))
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 2 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet2d(
            in_channels=4,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.class_predict = torch.nn.Conv2d(in_channels=12, out_channels=4, kernel_size=1)  # 最终输出层的卷积操作

        self.sigmoid = torch.nn.Sigmoid()
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x_raw, x_prompt):
        y_lsds, y_affinity = self.model_affinity(x_raw)

        y_concat = torch.cat([x_raw, y_affinity.detach(), x_prompt.unsqueeze(1)], dim=1)

        y_logits = self.class_predict(self.model_mask(y_concat))

        return y_logits, y_lsds, y_affinity


def get_acc_prec_recall_f1(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total != 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    return {
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "acc": accuracy,
                "prec": precision,
                "recall": recall,
                "f1": f1
            }


if __name__ == '__main__':
    ##设置超参数
    batch_size = 1

    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 设置所有可以使用的显卡，共计四块
    device_ids = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]  # 选中显卡
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # load trained model weights

    model_dir = os.path.join(HOME_PATH, 'training/output/segEM2d(ninanjie)_semantic/1.0-2.0-1.0/1-16-10-50')
    output_dir = os.path.join(model_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'segEM2d(ninanjie)_semantic-576.model')
    model = segEM2d().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device)
    ##装载数据
    dataset_names = ['second_6', 'fourth_1']

    ##装载数据
    crop_xyz = [1, 1, 1]
    test_dataset = []
    for dataset_name in dataset_names:
        for i in [crop_xyz[0] - 1]:
            for j in range(crop_xyz[1]):
                for k in range(crop_xyz[2]):
                    _test = load_test_dataset(dataset_name, raw_dir='raw', label_dir='export',
                                              from_temp=True, crop_xyz=crop_xyz, chunk_position=[i, j, k], crop_size=2400)
                    test_dataset.append(_test)
    test_dataset = ConcatDataset(test_dataset)


    def load_data_to_device(loader):
        tmp_loader = iter(loader)
        res = []
        # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
        for raw, point_map, gt_labels in tqdm(tmp_loader, desc="assigning to cuda", leave=True):
            ##Get Tensor
            raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
            point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)  # (batch, height, width)
            gt_labels = torch.as_tensor(gt_labels, dtype=torch.float, device=device)  # (batch, height, width)
            res.append([raw, point_map, gt_labels])
        return res

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=48, pin_memory=True,
                              drop_last=False, collate_fn=collate_fn_2D_fib25_Test)
    test_loader = load_data_to_device(test_loader)

    # set activation
    activation = torch.nn.Sigmoid()

    model.eval()
    seed = 98
    np.random.seed(seed)
    random.seed(seed)
    count, total = 0, len(test_loader)
    # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
    pool = multiprocessing.pool.Pool(28)
    processes = []

    with (torch.no_grad()):
        base_idx = 0
        voi_sum = 0
        for raw, point_map, gt_seg in tqdm(test_loader, leave=True, desc='testing'):
            y_logits, y_lsds, y_affinity = model(raw, point_map)
            y_pred = activation(y_logits)
            y_probs = torch.argmax(y_pred, dim=1)
            voi_sum += calculate_multi_class_voi(y_probs, gt_seg)

            y_probs = y_probs.detach().cpu()
            y_probs = np.asarray(y_probs)

            raw = np.asarray(raw.detach().cpu())
            gt_seg = gt_seg.detach().cpu()
            gt_seg = np.asarray(gt_seg)

            for idx, (single_img, single_prob) in enumerate(zip(raw, y_probs)):
                single_img = single_img.squeeze()
                single_prob = single_prob.squeeze()
                # visualize_and_save_mask(single_img, single_prob, idx + base_idx * batch_size, 'semantic')
                processes.append(
                    pool.apply_async(visualize_and_save_mask,
                                     args=(single_img, single_prob, output_dir,
                                           idx + base_idx * batch_size, 'semantic-seg_only')
                                     )
                )
                processes.append(
                    pool.apply_async(visualize_and_save_mask,
                                     args=(single_img, single_prob, output_dir,
                                           idx + base_idx * batch_size, 'semantic')
                                     )
                )
            base_idx += 1

        voi_sum /= (len(test_loader) + 0.)
        print(f'VOI: {voi_sum:.6f}')

    with tqdm(total=len(processes), leave=False, desc='saving') as pbar:
        for process in processes:
            res = process.get()
            pbar.update(res)
    pool.close()
    pool.join()
