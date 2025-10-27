import itertools
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
from scipy.ndimage import binary_fill_holes
from skimage.metrics import variation_of_information
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
# from torchmetrics.functional import accuracy, precision, recall, f1_score
from tqdm.auto import tqdm
from skimage.measure import label

# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from training.test_segEM2d_wloss import HOME_PATH
from training.utils.dataloader_ninanjie import load_test_dataset, collate_fn_2D_fib25_Test
from matplotlib.colors import ListedColormap

## CUDA_VISIBLE_DEVICES=0 python main_segEM_2d_train_zebrafinch.py &

HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

WEIGHT_LOSS2 = 15
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
            num_fmaps=24,
            fmap_inc_factors=3,
            downsample_factors=[[2, 2] for _ in range(4)],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.lsd_predict = torch.nn.Conv2d(in_channels=24, out_channels=6, kernel_size=1)  # 最终输出层的卷积操作

        # create our network, 6 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet2d(
            in_channels=7,  # 输入的图像通道数
            num_fmaps=24,
            fmap_inc_factors=3,
            downsample_factors=[[2, 2] for _ in range(4)],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.affinity_predict = torch.nn.Conv2d(in_channels=24, out_channels=2, kernel_size=1)  # 最终输出层的卷积操作

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
        model_path = ('/home/liuhongyu2024/Documents/UniSPAC-edited/training/output/log/'
                      'ACRLSD_2D(ninanjie)_all/24_3_4/normal/Best_in_val.model')
        weights = torch.load(model_path, map_location=torch.device('cuda'))
        self.model_affinity.load_state_dict(remove_module(weights))
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 2 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet2d(
            in_channels=3,  # 输入的图像通道数
            num_fmaps=24,
            fmap_inc_factors=4,
            downsample_factors=[[2, 2] for _ in range(4)],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.mask_predict = torch.nn.Conv2d(in_channels=24, out_channels=1, kernel_size=1)  # 最终输出层的卷积操作

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_raw, x_prompt):
        y_lsds, y_affinity = self.model_affinity(x_raw)

        y_concat = torch.cat([x_prompt.unsqueeze(1), y_affinity.detach()], dim=1)

        y_mask = self.mask_predict(self.model_mask(y_concat))
        y_mask = self.sigmoid(y_mask)

        return y_mask, y_lsds, y_affinity


def visualize_and_save_mask(raw, segmentation, idx=0, mode='origin',
                            background_color=(0, 0, 0), seed=98):
    import matplotlib.pyplot as plt
    # 获取所有实例标签（排除背景0）
    output_path = os.path.join(HOME_PATH, f'data/ninanjie/train/{dataset_names[0]}/output_{WEIGHT_LOSS2}')
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path,
                               f"{mode}_{str(idx).zfill(4)}.png")

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


def save_mask(mask, idx=0):
    import tifffile
    import matplotlib.pyplot as plt
    import os

    # 定义输出路径
    output_path = os.path.join(HOME_PATH, f'data/ninanjie/train/{dataset_names[0]}/mask')
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path,
                               f"{str(idx).zfill(4)}.tif")

    if mask.dtype not in (np.uint8, np.uint16):
        mask = mask.astype(np.uint8)  # 转换为 PIL 支持的数据类型
    tifffile.imwrite(output_path, mask)
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


def aftercare(y_mask: torch.Tensor):
    """
    对生物组织图像分割结果进行后处理

    参数:
        y_mask: 输入的分割结果张量，只含0和1
                可以是(B, H, W)的单层分割结果
                也可以是(B, N, H, W)的多层连续分割结果

    返回:
        处理后的分割结果张量
    """
    # 确保输入是浮点型张量
    y_mask = y_mask + 0.
    device = y_mask.device
    batch_size = y_mask.size(0)

    # 根据输入形状确定是否为多层结构
    is_multi_layer = len(y_mask.shape) == 4  # (B, N, H, W)

    # 定义结构元素（3x3的十字形结构，适合生物组织的连接性）
    kernel = torch.tensor([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], device=device, dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3)  # 适配卷积操作的形状 [out_channels, in_channels, kH, kW]

    # 存储处理结果
    result = []

    # 对每个样本进行处理
    for b in range(batch_size):
        if is_multi_layer:
            # 多层结构: (B, N, H, W)
            layers = []
            for n in range(y_mask.size(1)):
                layer = y_mask[b, n].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                processed_layer = process_single_layer(layer, device=device, kernel=kernel)
                layers.append(processed_layer)
            # 堆叠处理后的多层
            result.append(torch.cat(layers, dim=1))
        else:
            # 单层结构: (B, H, W)
            layer = y_mask[b].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            processed_layer = process_single_layer(layer, device=device, kernel=kernel)
            result.append(processed_layer.squeeze(0))  # 移除通道维度

    # 堆叠所有样本
    result = torch.stack(result, dim=0)

    # 确保输出仍是0和1的二进制张量
    return (np.asarray(result) > 0.5) + 0


def process_single_layer(layer: torch.Tensor, device, kernel):
    """处理单个图层，执行填充空心和圆化操作"""
    layer = np.asarray(layer.cpu())

    # 0. 过滤小目标
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    #     layer, connectivity=8  # connectivity=4 表示4连通
    # )
    #
    # min_size = 50
    # filtered_array = np.zeros_like(labels, dtype=np.uint8)
    # for i in range(1, num_labels):  # 从1开始，0是背景
    #     area = stats[i, cv2.CC_STAT_AREA]  # 获取第i个分量的面积
    #     if area >= min_size:
    #         filtered_array[labels == i] = 255  # 保留大分量（设为255）

    # 1. 填充连通域中间的空心区域
    filled = binary_fill_holes(layer) + 0.
    filled = torch.as_tensor(filled, device=device, dtype=torch.float32)
    # 2. 先膨胀再腐蚀（闭运算），圆化目标边缘
    # 膨胀操作
    dilated = morphological_dilation(filled, kernel)
    # dilated = binary_dilation(filled, iterations=1, border_value=1)
    # 腐蚀操作
    rounded = morphological_erosion(dilated, kernel)
    # rounded = binary_erosion(dilated, iterations=1, border_value=1)

    return rounded


def morphological_dilation(x: torch.Tensor, kernel: torch.Tensor):
    """形态学膨胀操作"""
    padding = (kernel.size(2) // 2, kernel.size(3) // 2)  # SAME padding
    x_dilated = F.conv2d(x, kernel, padding=padding)
    return (x_dilated > 0.5).float()  # 二值化


def morphological_erosion(x: torch.Tensor, kernel: torch.Tensor):
    """形态学腐蚀操作"""
    padding = (kernel.size(2) // 2, kernel.size(3) // 2)  # SAME padding
    # 腐蚀是对反相图像的膨胀
    x_inverted = 1 - x
    x_eroded_inverted = F.conv2d(x_inverted, kernel, padding=padding)
    x_eroded = 1 - (x_eroded_inverted > 0.5).float()  # 反相回来并二值化
    return x_eroded


def morphological_closing(x: torch.Tensor, kernel: torch.Tensor, iterations: int = 1):
    """形态学闭运算（先膨胀后腐蚀）"""
    result = x
    for _ in range(iterations):
        result = morphological_dilation(result, kernel)
        result = morphological_erosion(result, kernel)
    return result


if __name__ == '__main__':
    ##设置超参数
    batch_size = 1

    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 设置所有可以使用的显卡，共计四块
    device_ids = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]  # 选中显卡
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # load trained model weights
    model_path = os.path.join(HOME_PATH,
                              f'/home/liuhongyu2024/Documents/UniSPAC-edited/training/output/log/'
                              f'segEM2d(ninanjie)-prompt-3rd-1/weighted-w2-5-w3-1/24_3_4/Best_in_val.model')
    model = segEM2d().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device)

    ##装载数据
    class_num = None
    dataset_names = ['second_6']
    # Save_Name = 'segEM2d(ninanjie)_subcell_{}-w2-{}-w3-{}_512'.format(class_num, WEIGHT_LOSS2, WEIGHT_LOSS3)

    ##装载数据
    crop_xyz = [1, 1, 1]
    test_dataset = []

    for dataset_name in dataset_names:
        for i, j, k in itertools.product(
                range(crop_xyz[0]),
                range(crop_xyz[1]),
                range(crop_xyz[2])
        ):
            test_tmp = load_test_dataset(dataset_name, raw_dir='raw', label_dir='export',
                                         from_temp=False, semantic_class_num=class_num,
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
            # gt_labels = torch.as_tensor(gt_labels, dtype=torch.float, device=device)  # (batch, height, width)
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
        for raw, point_map, _ in tqdm(test_loader, leave=True, desc='testing'):
            y_mask, y_lsds, y_affinity = model(raw, point_map)
            y_mask = y_mask.detach().cpu()
            binary_y_mask = (y_mask > 0.5) + 0
            binary_y_mask = aftercare(binary_y_mask)

            raw = np.asarray(raw.detach().cpu())
            # gt_seg = gt_seg.detach().cpu()
            # binary_gt_seg = (np.asarray(gt_seg) > 0) + 0

            # voi_ = sum(variation_of_information(binary_y_mask.squeeze(), binary_gt_seg.squeeze()))
            #
            # y_flat = binary_y_mask.flatten()
            # gt_flat = binary_gt_seg.flatten()
            #
            # TP = np.sum((y_flat == 1) & (gt_flat == 1))
            # TN = np.sum((y_flat == 0) & (gt_flat == 0))
            # FP = np.sum((y_flat == 1) & (gt_flat == 0))
            # FN = np.sum((y_flat == 0) & (gt_flat == 1))
            #
            # metric_ = {'voi': voi_}
            # metric_.update(get_acc_prec_recall_f1(TP, TN, FP, FN))
            # metrics.append(metric_)

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
                processes.append(pool.apply_async(save_mask,
                                                  args=(single_mask, idx + base_idx * batch_size)))
            base_idx += 1

    with tqdm(total=len(processes), leave=False, desc='saving') as pbar:
        for process in processes:
            res = process.get()
            pbar.update(res)
    pool.close()
    pool.join()
