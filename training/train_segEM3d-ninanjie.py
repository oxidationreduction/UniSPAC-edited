import gc
import itertools
from collections import OrderedDict

import cv2
import numpy as np
import os
import random

import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import binary_fill_holes
from skimage import measure
from skimage.metrics import variation_of_information
from torch.nn import functional as F
from torch.utils.data import DataLoader,ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import logging
from copy import deepcopy
# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from models.unet3d import UNet3d
from training.utils.dataloader_ninanjie import collate_fn_3D_ninanjie_Train, load_train_dataset, get_acc_prec_recall_f1
from utils.dataloader_hemi_better import Dataset_3D_hemi_Train,collate_fn_3D_hemi_Train
from utils.dataloader_fib25_better import Dataset_3D_fib25_Train,collate_fn_3D_fib25_Train
# from utils.dataloader_cremi import Dataset_3D_cremi_Train,collate_fn_3D_cremi_Train

## CUDA_VISIBLE_DEVICES=0 python main_segEM_3d_train_zebrafinch.py &

WEIGHT_LOSS3 = 10


def set_seed(seed = 1998):
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
            in_channels=1, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2],[2,2],[2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.lsd_predict = torch.nn.Conv2d(in_channels=12,out_channels=6, kernel_size=1)  #最终输出层的卷积操作


        # create our network, 6 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet2d(
            in_channels=7, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2],[2,2],[2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.affinity_predict = torch.nn.Conv2d(in_channels=12,out_channels=2, kernel_size=1)  #最终输出层的卷积操作
    
    def forward(self, x):

        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x,y_lsds],dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds,y_affinity
    
    
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
            in_channels=3, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2],[2,2],[2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.mask_predict = torch.nn.Conv2d(in_channels=12,out_channels=1, kernel_size=1)  #最终输出层的卷积操作
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x_raw, x_prompt):

        y_lsds,y_affinity = self.model_affinity(x_raw)
        
        y_concat = torch.cat([x_prompt.unsqueeze(1),y_affinity],dim=1)

        y_mask = self.mask_predict(self.model_mask(y_concat))
        y_mask = self.sigmoid(y_mask)

        return y_mask,y_lsds,y_affinity



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
            in_channels=1, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2,2],[2,2,2],[2,2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.lsd_predict = torch.nn.Conv3d(in_channels=12,out_channels=10, kernel_size=1)  #最终输出层的卷积操作

        # create our network, 10 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet3d(
            in_channels=11, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2,2],[2,2,2],[2,2,2]], #降采样的因子
            padding='same',
            constant_upsample=True)
        
        self.affinity_predict = torch.nn.Conv3d(in_channels=12,out_channels=3, kernel_size=1)  #最终输出层的卷积操作
    
    def forward(self, x):
        
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x,y_lsds.detach()],dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds,y_affinity


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') # 去掉 'module.' 前缀
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
        model_path = './output/log/segEM2d(ninanjie)-prompt/w2-5-w3-1*/w2-5-w3-1_Best_in_val.model'
        weights = torch.load(model_path,map_location=torch.device('cuda:0'))
        self.model_mask_2d.load_state_dict(remove_module(weights))
        for param in self.model_mask_2d.parameters():
            param.requires_grad = False
        
        ##For affinity prediction
        self.model_affinity = ACRLSD_3d()
        # model_path = './output/checkpoints/ACRLSD_3D(hemi+fib25+cremi)_Best_in_val.model' 
        model_path = './output/log/ACRLSD_3D(ninanjie)_384/ACRLSD_3D(ninanjie)_384_Best_in_val.model'
        weights = torch.load(model_path,map_location=torch.device('cuda:0'))
        self.model_affinity.load_state_dict(remove_module(weights))
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 3 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet3d(
            in_channels=4, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2,2],[2,2,2],[2,2,2]], #降采样的因子
            padding='same',
            constant_upsample=True).cuda()
        
        self.mask_predict = torch.nn.Conv3d(in_channels=12,out_channels=1, kernel_size=1)  #最终输出层的卷积操作
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x_raw, x_prompt):
        '''
        x_raw: shape = (Batch * channel * dim_x * dim_y * dim_z)
        x_prompt: shape = (Batch * channel * dim_x * dim_y * dim_z)
        '''
        ##Get mask for slice0
        y_mask2d_slice0,_,_ = self.model_mask_2d(x_raw[:,:,:,:,0],x_prompt)
        
        ##Get affinity for raw
        y_lsds,y_affinity = self.model_affinity(x_raw)
        
        #replace raw slice0
        x_raw_new = deepcopy(x_raw)
        x_raw_new[:,0,:,:,0] = (y_mask2d_slice0.detach().squeeze()>0.5) + 0
        y_concat = torch.cat([x_raw_new,y_affinity.detach()],dim=1)

        y_mask3d = self.mask_predict(self.model_mask(y_concat))
        y_mask3d = self.sigmoid(y_mask3d)

        return y_mask3d,y_mask2d_slice0,y_affinity,y_lsds

    
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

def model_step(model, optimizer, input_image, input_prompt, gt_binary_mask, gt_affinity, activation, train_step=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()
        
    # forward
    # lsd_logits,affinity_logits = model(raw)
    y_mask, _, _, _ = model(input_image,input_prompt)

    gt_binary_mask = torch.as_tensor(gt_binary_mask, dtype=torch.float32, device=gt_binary_mask.device)
    loss1 = F.binary_cross_entropy(y_mask.squeeze(), gt_binary_mask.squeeze())
    Diceloss_fn = DiceLoss().to(device)
    loss2 = Diceloss_fn(1-y_mask.squeeze(), 1-gt_binary_mask.squeeze())
    
    loss3 = torch.sum(y_mask * gt_affinity)/torch.sum(gt_affinity)
    
    loss = loss1 + loss2 + loss3 * WEIGHT_LOSS3
    # loss = loss1 + loss2
    
    # backward if training mode
    if train_step:
        loss.backward()
        optimizer.step()
    else:
        y_mask = aftercare((y_mask.squeeze() > 0.5) + 0.)
        
    # lsd_output = activation(lsd_logits)
    # affinity_output = activation(affinity_logits) 
    # outputs = {
    #     'pred_lsds': lsd_output,
    #     'lsds_logits': lsd_logits,
    #     'pred_affinity': affinity_output,
    #     'affinity_logits': affinity_logits,
    # }
    
    return loss, y_mask, (loss1, loss2, loss3)


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
    return (result > 0.5) + 0.


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


def visualize_and_save_mask(raw: torch.Tensor, segmentation: torch.Tensor, idx=0, mode='origin',
                            background_color=(0, 0, 0), seed=98, writer=None):
    """
    可视化原始图像和分割掩码，并可选择保存为图片或写入TensorBoard

    参数:
        raw: 原始图像
        segmentation: 分割结果
        idx: 图像索引，用于命名
        mode: 可视化模式
        background_color: 背景颜色
        seed: 随机种子，确保颜色一致性
        writer: TensorBoard的SummaryWriter实例，如果提供则写入TensorBoard
        tag: TensorBoard中的标签
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    raw = np.asarray(raw.cpu()).squeeze().transpose(0, 3, 1, 2)[1, 1, ...].squeeze()
    segmentation = np.asarray(segmentation.cpu(), dtype=np.int32).squeeze().transpose(0, 3, 1, 2)[1, 1, ...].squeeze()
    segmentation = measure.label(segmentation)

    # 处理模式为'seg_only'的情况
    if mode == 'seg_only':
        mask = segmentation > 0
        raw = raw * mask

    # 获取所有实例标签（排除背景0）
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
    max_label = int(np.max(segmentation)) if num_instances > 0 else 0
    cmap_colors = [background_color] * (max_label + 1)  # 初始化所有标签颜色

    # 为每个实例标签分配颜色
    for i, label in enumerate(instances):
        cmap_colors[label] = tuple(colors[i])

    # 创建自定义颜色映射
    cmap = ListedColormap(cmap_colors)

    # 绘制图像
    plt.figure(figsize=(10, 10))
    # 显示原始图像
    plt.imshow(raw, cmap='gray' if raw.ndim == 2 else None)
    # 显示分割结果（仅显示非背景区域）
    masked_segmentation = np.ma.masked_where(segmentation == 0, segmentation)
    plt.imshow(masked_segmentation, cmap=cmap, alpha=0.5)  # alpha控制透明度
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout(pad=0)  # 去除边距

    # 如果提供了writer，则将图像写入TensorBoard
    if writer is not None:
        # 将matplotlib图像转换为numpy数组以便TensorBoard显示
        import io
        from PIL import Image

        # 将图像保存到内存缓冲区
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
        buf.seek(0)

        # 转换为PIL图像并再转换为numpy数组
        img = Image.open(buf)
        img_array = np.array(img)

        # 写入TensorBoard，注意需要调整通道顺序为[C, H, W]
        if img_array.ndim == 3:
            img_array = img_array.transpose(2, 0, 1)

        # 添加到TensorBoard，使用idx作为全局步长
        writer.add_image(mode, img_array, global_step=idx, dataformats='CHW')

    plt.close()


def trainer(log_dir):
    model = segEM_3d()
    model = model.to(device)

    # ###多卡训练
    # ###一机多卡设置
    model = nn.DataParallel(model, device_ids=device_ids)  # 并行使用

    Save_Name = f'{WEIGHT_LOSS1}-{WEIGHT_LOSS2}-{WEIGHT_LOSS3}'

    ##创建log日志
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    log_dir = f'{log_dir}/{Save_Name}'

    logfile = '{}/log.txt'.format(log_dir)
    csvfile = '{}/log.csv'.format(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info(f'''Starting training:
        training_epochs:  {training_epochs}
        Num_patch_train:  {len(train_dataset)}
        Num_patch_val:    {len(val_dataset)}
        Batch size:       {batch_size}
        Learning rate:    {learning_rate}
        Device:           {device.type}
        ''')

    ##开始训练
    # set optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # set activation
    activation = torch.nn.Sigmoid()

    # training loop
    model.train()
    epoch = 0
    Best_val_loss = 100000
    Best_epoch = 0
    early_stop_count = 64
    no_improve_count = 0
    with tqdm(total=training_epochs) as pbar:
        analysis = pd.DataFrame(columns=['bce', 'dice', 'affinity', 'acc', 'prec', 'recall', 'f1', 'voi'])
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            ###################Train###################
            train_desc = f"loss: {Best_val_loss:.4f} in {Best_epoch}, "
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            tmp_loader = iter(train_loader)
            count, total = 0, len(tmp_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
            for raw, labels, mask, affinity, point_map, gt_lsd in tmp_loader:
                ##Get Tensor
                # raw = torch.as_tensor(raw,dtype=torch.float, device= device)
                # point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)
                # mask = torch.as_tensor(mask, dtype=torch.float, device=device)
                # affinity = torch.as_tensor(affinity, dtype=torch.float, device=device) #(batch, 2, height, width)
                model_step(model, optimizer, raw, point_map, mask, affinity, activation, train_step=True)

                count += 1
                progress_desc = f"Train {count}/{total}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 98
            np.random.seed(seed)
            random.seed(seed)
            tmp_val_loader = iter(val_loader)
            acc_loss = []
            detailed_losses = np.array([0.] * 3)
            voi_val = 0.
            judgement_rates = np.array([0.] * 4)
            count, total = 0, len(tmp_val_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            for raw, labels, mask, affinity, point_map, gt_lsd in tmp_val_loader:
                # raw = torch.as_tensor(raw,dtype=torch.float, device= device) #(batch, 1, height, width, depth)
                # point_map = torch.as_tensor(point_map, dtype=torch.float, device=device) #(batch, height, width, depth)
                # mask = torch.as_tensor(mask, dtype=torch.float, device=device) #(batch, 2, height, width, depth)
                # affinity = torch.as_tensor(affinity, dtype=torch.float, device=device) #(batch, 2, height, width, depth)

                with torch.no_grad():
                    loss_value, pred, losses = model_step(model, optimizer, raw, point_map, mask, affinity, activation,
                                                          train_step=False)

                    binary_y_pred = np.asarray(pred.cpu(), dtype=np.uint8).flatten()
                    binary_gt_mask = np.asarray(mask.cpu(), dtype=np.uint8).flatten()

                    voi_val += np.sum(variation_of_information(binary_y_pred, binary_gt_mask))
                    judgement_rates += np.asarray(get_acc_prec_recall_f1(binary_y_pred, binary_gt_mask))
                    acc_loss.append(loss_value.cpu().detach().numpy())
                    detailed_losses += np.asarray([loss_.cpu().detach().numpy() for loss_ in losses])

                count += 1
                progress_desc = f'Val: {count}/{total}, '
                pbar.set_description(progress_desc + train_desc + val_desc)

            detailed_losses /= (total + 0.)
            bce_, dice_, affinity_ = detailed_losses[:]
            voi_val /= (total + 0.)
            judgement_rates /= (total + 0.)
            acc, prec, recall, f1 = judgement_rates[:]
            val_loss = torch.as_tensor([loss_value.item() for loss_value in acc_loss]).mean().item()
            val_desc = f'acc: {acc:.3f}, prec: {prec:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, VOI: {voi_val:.5f}'
            analysis.loc[len(analysis)] = [bce_, dice_, affinity_, acc, prec, recall, f1, voi_val]

            visualize_and_save_mask(raw, pred, mode='visual/seg_only', idx=epoch, writer=writer)
            visualize_and_save_mask(raw, pred, mode='visual/normal', idx=epoch, writer=writer)

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.state_dict(), '{}/Best_in_val.model'.format(log_dir))
                no_improve_count = 0
            else:
                no_improve_count += 1

            pbar.update(1)
            epoch += 1
            ##Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, Best_val_loss, Best_epoch))
            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('loss/bce', bce_, epoch)
            writer.add_scalar('loss/dice', dice_, epoch)
            writer.add_scalar('loss/affinity', affinity_, epoch)
            writer.add_scalar('metrics/accuracy', acc, epoch)
            writer.add_scalar('metrics/precision', prec, epoch)
            writer.add_scalar('metrics/recall', recall, epoch)
            writer.add_scalar('metrics/f1_score', f1, epoch)
            writer.add_scalar('VOI', voi_val, epoch)
            analysis.to_csv(csvfile)

            ##Early stop
            if no_improve_count == early_stop_count:
                logging.info("Early stop!")
                break

    torch.save(model.module.state_dict(), f'{log_dir}/final.model')
    del model, optimizer, activation
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 32
    # Save_Name = 'segEM_3d(hemi+fib25+cremi)'
    # Save_Name = 'segEM_3d(hemi+fib25)faster_wloss3({})'.format(WEIGHT_LOSS_AFFINITY)

    set_seed()

    ###创建模型
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,1,0,5'  # 设置所有可以使用的显卡，共计四块
    device_ids = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))] #选中显卡
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    dataset_names = ['best_truth_3d']
    crop_size = 384
    crop_xyz = [4, 3, 1]
    train_dataset, val_dataset = [], []

    # multiprocessing.set_start_method('spawn', force=True)
    # with multiprocessing.Pool(2) as pool:
    #     results = []
    #     for dataset_name in dataset_names:
    #         for i, j, k in itertools.product(
    #                 range(crop_xyz[0] - 1),
    #                 range(crop_xyz[1]),
    #                 range(crop_xyz[2])
    #         ):
    #             results.append(pool.apply_async(_load_datasets, args=(dataset_name, crop_size, crop_xyz, i, j, k)))
    #     for result_ in results:
    #         train_, val_ = result_.get()
    #         train_dataset.append(train_)
    #         val_dataset.append(val_)

    for dataset_name, i, j, k in itertools.product(
            dataset_names,
            range(crop_xyz[0] - 1),
            range(crop_xyz[1]),
            range(crop_xyz[2])
    ):
        train_, val_ = load_train_dataset(dataset_name, raw_dir='raw_2', label_dir='truth_label_2', from_temp=True,
                                          require_lsd=False, require_xz_yz=False,
                                          crop_size=crop_size, crop_xyz=crop_xyz, chunk_position=[i, j, k])
        train_dataset.append(train_)
        val_dataset.append(val_)

    train_dataset, val_dataset = ConcatDataset(train_dataset), ConcatDataset(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn_3D_ninanjie_Train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
                            collate_fn=collate_fn_3D_ninanjie_Train)

    log_dir = './output/log/segEM_3d(ninanjie)-prompt'
    os.makedirs(log_dir, exist_ok=True)
    for WEIGHT_LOSS1, WEIGHT_LOSS2, WEIGHT_LOSS3 in (
        [10, 1, 1], [1, 10, 1], [1, 1, 10]
    ):
        trainer()
