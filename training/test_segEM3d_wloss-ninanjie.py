import logging
import multiprocessing.pool
import os
import random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.metrics import variation_of_information
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
# from torchmetrics.functional import accuracy, precision, recall, f1_score
from tqdm.auto import tqdm
from skimage.measure import label

# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from training.models.unet3d import UNet3d
from training.test_segEM2d_wloss import HOME_PATH
from training.utils.dataloader_ninanjie import load_test_dataset, collate_fn_2D_fib25_Test
from matplotlib.colors import ListedColormap

## CUDA_VISIBLE_DEVICES=0 python main_segEM_2d_train_zebrafinch.py &

HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

WEIGHT_LOSS2 = 15
WEIGHT_LOSS3 = 1


def set_seed(seed=19260817):
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
            constant_upsample=True)

        self.lsd_predict = torch.nn.Conv2d(in_channels=12, out_channels=6, kernel_size=1)  # 最终输出层的卷积操作

        # create our network, 6 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet2d(
            in_channels=7,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.affinity_predict = torch.nn.Conv2d(in_channels=12, out_channels=2, kernel_size=1)  # 最终输出层的卷积操作

    def forward(self, x):
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x, y_lsds], dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds, y_affinity


####ACRLSD模型
class segEM_2d(torch.nn.Module):
    def __init__(
            self,
    ):
        super(segEM_2d, self).__init__()

        ##For affinity prediction
        self.model_affinity = ACRLSD()
        # model_path = './output/checkpoints/ACRLSD_2D(hemi+fib25+cremi)_Best_in_val.model'
        model_path = './output/checkpoints/ACRLSD_2D(ninanjie)_half_crop_Best_in_val.model'
        weights = torch.load(model_path, map_location=torch.device('cuda:0'))
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

        y_concat = torch.cat([x_prompt.unsqueeze(1), y_affinity], dim=1)

        y_mask = self.mask_predict(self.model_mask(y_concat))
        y_mask = self.sigmoid(y_mask)

        return y_mask, y_lsds, y_affinity


####ACRLSD_3d模型

####ACRLSD模型
class ACRLSD_3d(torch.nn.Module):
    def __init__(
            self,
    ):
        super(ACRLSD_3d, self).__init__()
        # d_factors = [[2,2,2],[2,2,2],[2,2,2]]  #降采样的因子
        # in_channels=1 #输入的图像通道数
        # num_fmaps=12
        # fmap_inc_factor=5

        # create our network, 1 input channels in the raw data
        self.model_lsds = UNet3d(
            in_channels=1,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.lsd_predict = torch.nn.Conv3d(in_channels=12, out_channels=10, kernel_size=1)  # 最终输出层的卷积操作

        # create our network, 10 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet3d(
            in_channels=11,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.affinity_predict = torch.nn.Conv3d(in_channels=12, out_channels=3, kernel_size=1)  # 最终输出层的卷积操作

    def forward(self, x):
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x, y_lsds.detach()], dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds, y_affinity


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # 去掉 'module.' 前缀
        new_state_dict[name] = v
    return new_state_dict


class segEM_3d(torch.nn.Module):
    def __init__(
            self,
    ):
        super(segEM_3d, self).__init__()

        ##2D slice mask prediction
        self.model_mask_2d = segEM_2d()
        # model_path = './output/checkpoints/segEM2d(hemi+fib25+cremi)_Best_in_val.model'
        # model_path = './output/checkpoints/segEM2d(hemi+fib25)faster_wloss3({})_Best_in_val.model'.format(WEIGHT_LOSS_AFFINITY)
        model_path = './output/checkpoints/segEM2d(ninanjie)-w2-2-w3-1_Best_in_val.model'
        weights = torch.load(model_path, map_location=torch.device('cuda:0'))
        self.model_mask_2d.load_state_dict(remove_module(weights))
        for param in self.model_mask_2d.parameters():
            param.requires_grad = False

        ##For affinity prediction
        self.model_affinity = ACRLSD_3d()
        # model_path = './output/checkpoints/ACRLSD_3D(hemi+fib25+cremi)_Best_in_val.model'
        model_path = './output/log/ACRLSD_3D(ninanjie)_384/ACRLSD_3D(ninanjie)_384_Best_in_val.model'
        weights = torch.load(model_path, map_location=torch.device('cuda:0'))
        self.model_affinity.load_state_dict(remove_module(weights))
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 3 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet3d(
            in_channels=4,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True).cuda()

        self.mask_predict = torch.nn.Conv3d(in_channels=12, out_channels=1, kernel_size=1)  # 最终输出层的卷积操作

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_raw, x_prompt):
        '''
        x_raw: shape = (Batch * channel * dim_x * dim_y * dim_z)
        x_prompt: shape = (Batch * channel * dim_x * dim_y * dim_z)
        '''
        ##Get mask for slice0
        y_mask2d_slice0, _, _ = self.model_mask_2d(x_raw[:, :, :, :, 0], x_prompt)

        ##Get affinity for raw
        y_lsds, y_affinity = self.model_affinity(x_raw)

        # replace raw slice0
        x_raw_new = deepcopy(x_raw)
        x_raw_new[:, 0, :, :, 0] = (y_mask2d_slice0.detach().squeeze() > 0.5) + 0
        y_concat = torch.cat([x_raw_new, y_affinity.detach()], dim=1)

        y_mask3d = self.mask_predict(self.model_mask(y_concat))
        y_mask3d = self.sigmoid(y_mask3d)

        return y_mask3d, y_mask2d_slice0, y_affinity, y_lsds


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


def visualize_and_save_mask(raw, segmentation, idx=0, mode='origin',
                            background_color=(0, 0, 0), seed=98):
    import matplotlib.pyplot as plt
    # 获取所有实例标签（排除背景0）
    output_path = os.path.join(HOME_PATH, f'data/ninanjie/train/second_6/output_{WEIGHT_LOSS2}', f"{mode}_{str(idx).zfill(4)}.png")

    # 获取所有实例标签（排除背景0）
    if mode == 'seg_only':
        mask = segmentation > 0
        raw = raw * mask
    instances = np.unique(segmentation)
    instances = instances[instances != 0]
    num_instances = len(instances)

    # 设置随机种子，确保颜色分配一致
    np.random.seed(seed)

    # 为每个实例生成随机颜色（RGB格式，值在0-1之间）
    # 避免过暗颜色，提高可视性
    colors = np.random.rand(num_instances, 3)
    colors = np.clip(colors, 0.5, 0.9)  # 限制颜色亮度范围

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
    batch_size = 4

    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'  # 设置所有可以使用的显卡，共计四块
    device_ids = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]  # 选中显卡
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # load trained model weights
    model_path = os.path.join(HOME_PATH, f'training/output/checkpoints/segEM2d(ninanjie)_subcell_1-w2-{WEIGHT_LOSS2}-w3-1_512_Best_in_val.model')
    model = segEM2d().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device)

    ##装载数据
    class_num = 1
    dataset_names = ['second_6']
    # Save_Name = 'segEM2d(ninanjie)_subcell_{}-w2-{}-w3-{}_512'.format(class_num, WEIGHT_LOSS2, WEIGHT_LOSS3)

    ##装载数据
    crop_xyz = [1, 1, 1]
    test_dataset = []
    i = crop_xyz[0] - 1
    for dataset_name in dataset_names:
        for j in range(crop_xyz[1]):
            for k in range(crop_xyz[2]):
                test_tmp = load_test_dataset(dataset_name, raw_dir='raw', label_dir='export',
                                                       from_temp=True, semantic_class_num=class_num,
                                                       crop_xyz=crop_xyz, chunk_position=[i, j, k])
                test_dataset.append(test_tmp)

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
    acc_loss = []
    count, total = 0, len(test_loader)
    # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
    pool = multiprocessing.pool.Pool(28)
    processes = []

    metrics, voi_vals = [], []

    with (torch.no_grad()):
        base_idx = 0
        for raw, point_map, gt_seg in tqdm(test_loader, leave=True, desc='testing'):
            y_mask, y_lsds, y_affinity = model(raw, point_map)
            y_mask = y_mask.detach().cpu()
            binary_y_mask = (np.asarray(y_mask) > 0.5) + 0
            raw = np.asarray(raw.detach().cpu())
            gt_seg = gt_seg.detach().cpu()
            binary_gt_seg = (np.asarray(gt_seg) > 0) + 0

            voi_ = sum(variation_of_information(binary_y_mask.squeeze(), binary_gt_seg.squeeze()))

            y_flat = binary_y_mask.flatten()
            gt_flat = binary_gt_seg.flatten()

            TP = np.sum((y_flat == 1) & (gt_flat == 1))
            TN = np.sum((y_flat == 0) & (gt_flat == 0))
            FP = np.sum((y_flat == 1) & (gt_flat == 0))
            FN = np.sum((y_flat == 0) & (gt_flat == 1))

            metric_ = {'voi': voi_}
            metric_.update(get_acc_prec_recall_f1(TP, TN, FP, FN))
            metrics.append(metric_)

            for idx, (single_img, single_mask) in enumerate(zip(raw, binary_y_mask)):
                single_img = single_img.squeeze()
                single_mask = single_mask.squeeze()
                processes.append(pool.apply_async(visualize_and_save_mask,
                                                  args=(single_img, single_mask, idx + base_idx * batch_size, 'origin')))
                segmentation = label(single_mask[:, :])
                processes.append(pool.apply_async(visualize_and_save_mask,
                                                  args=(single_img, segmentation, idx + base_idx * batch_size, 'segment')))
                processes.append(pool.apply_async(visualize_and_save_mask,
                                                  args=(
                                                  single_img, segmentation, idx + base_idx * batch_size, 'seg_only')))
            base_idx += 1

        results_df = pd.DataFrame(metrics)
        # TP_sum, TN_sum, FP_sum, FN_sum = results_df["TP"].sum(), results_df["TN"].sum(), \
        #                                  results_df["FP"].sum(), results_df["FN"].sum()
        # results_df = pd.concat([results_df, pd.DataFrame([get_acc_prec_recall_f1(TP_sum, TN_sum, FP_sum, FN_sum)])],
        #                        ignore_index=True)

        output_path = os.path.join(HOME_PATH, "testing", f"second_6_{WEIGHT_LOSS2}.xlsx")
        results_df.to_excel(output_path, index=False, engine="openpyxl")

    with tqdm(total=len(processes), leave=False, desc='saving') as pbar:
        for process in processes:
            res = process.get()
            pbar.update(res)
    pool.close()
    pool.join()
