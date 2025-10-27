import gc
import itertools
import sys
from collections import OrderedDict

import cv2
import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.distributed as dist
from skimage.metrics import variation_of_information
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import logging
from copy import deepcopy
# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from models.unet3d import UNet3d
from training.utils.aftercare import aftercare, visualize_and_save_mask, visualize_and_save_affinity, normalize_affinity
from training.utils.dataloader_ninanjie import collate_fn_3D_ninanjie_Train, load_train_dataset, get_acc_prec_recall_f1
from utils.dataloader_hemi_better import Dataset_3D_hemi_Train, collate_fn_3D_hemi_Train
from utils.dataloader_fib25_better import Dataset_3D_fib25_Train, collate_fn_3D_fib25_Train

# from utils.dataloader_cremi import Dataset_3D_cremi_Train,collate_fn_3D_cremi_Train

## CUDA_VISIBLE_DEVICES=0 python main_segEM_3d_train_zebrafinch.py &

WEIGHT_LOSS1 = 1
WEIGHT_LOSS2 = 5
WEIGHT_LOSS3 = 2


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
            num_fmaps=24,
            fmap_inc_factors=3,
            downsample_times=4
    ):
        super(ACRLSD, self).__init__()

        # create our network, 1 input channels in the raw data
        self.model_lsds = UNet2d(
            in_channels=1,  # 输入的图像通道数
            num_fmaps=num_fmaps,
            fmap_inc_factors=fmap_inc_factors,
            downsample_factors=[[2, 2] for _ in range(downsample_times)],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.lsd_predict = torch.nn.Conv2d(in_channels=num_fmaps, out_channels=6, kernel_size=1)  # 最终输出层的卷积操作

        # create our network, 6 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet2d(
            in_channels=7,  # 输入的图像通道数
            num_fmaps=num_fmaps,
            fmap_inc_factors=fmap_inc_factors,
            downsample_factors=[[2, 2] for _ in range(downsample_times)],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.affinity_predict = torch.nn.Conv2d(in_channels=num_fmaps, out_channels=2, kernel_size=1)  # 最终输出层的卷积操作

    def forward(self, x):
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x, y_lsds.detach()], dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds, y_affinity



####ACRLSD模型
class segEM_2d(torch.nn.Module):
    def __init__(
            self,
            num_fmaps=24,
            fmap_inc_factors=3,
            downsample_times=4
    ):
        super(segEM_2d, self).__init__()

        ##For affinity prediction
        self.model_affinity = ACRLSD(num_fmaps=num_fmaps, fmap_inc_factors=fmap_inc_factors, downsample_times=downsample_times)
        # model_path = './output/checkpoints/ACRLSD_2D(hemi+fib25+cremi)_Best_in_val.model'
        model_path = ('/home/liuhongyu2024/Documents/UniSPAC-edited/training/output/log/ACRLSD_2D(ninanjie)_all/'
                      f'{num_fmaps}_{fmap_inc_factors}_{downsample_times}/background/Best_in_val.model')
        weights = torch.load(model_path, map_location=torch.device('cuda'))
        self.model_affinity.load_state_dict(remove_module(weights))
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 2 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet2d(
            in_channels=3,  # 输入的图像通道数
            num_fmaps=num_fmaps,
            fmap_inc_factors=4,
            downsample_factors=[[2, 2] for _ in range(downsample_times)],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.mask_predict = torch.nn.Conv2d(in_channels=num_fmaps, out_channels=1, kernel_size=1)  # 最终输出层的卷积操作

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_raw, x_prompt):
        y_lsds, y_affinity = self.model_affinity(x_raw)

        y_concat = torch.cat([x_prompt.unsqueeze(1), y_affinity.detach()], dim=1)

        y_mask = self.mask_predict(self.model_mask(y_concat))
        y_mask = self.sigmoid(y_mask)

        return y_mask, y_lsds, y_affinity


####ACRLSD_3d模型
class ACRLSD_3d(torch.nn.Module):
    def __init__(
            self,
            num_fmaps=21,
            fmap_inc_factor=5
    ):
        super(ACRLSD_3d, self).__init__()
        # d_factors = [[2,2,2],[2,2,2],[2,2,2]]  #降采样的因子
        # in_channels=1 #输入的图像通道数
        # num_fmaps=12
        # fmap_inc_factor=5

        # create our network, 1 input channels in the raw data
        self.model_lsds = UNet3d(
            in_channels=1,  # 输入的图像通道数
            num_fmaps=num_fmaps,
            fmap_inc_factors=fmap_inc_factor,
            downsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 1]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.lsd_predict = torch.nn.Conv3d(in_channels=num_fmaps, out_channels=10, kernel_size=1)  # 最终输出层的卷积操作

        # create our network, 10 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet3d(
            in_channels=11,  # 输入的图像通道数
            num_fmaps=num_fmaps,
            fmap_inc_factors=fmap_inc_factor,
            downsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 1]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.affinity_predict = torch.nn.Conv3d(in_channels=num_fmaps, out_channels=3, kernel_size=1)  # 最终输出层的卷积操作

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


class segEM_3d_single(torch.nn.Module):
    def __init__(
            self,
            num_fmaps=21,
            fmap_inc_factor=4
    ):
        super(segEM_3d_single, self).__init__()

        # create our network, 3 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet3d(
            in_channels=4,  # 输入的图像通道数
            num_fmaps=num_fmaps,
            fmap_inc_factors=fmap_inc_factor,
            downsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 1]],  # 降采样的因子
            padding='same',
            constant_upsample=True).cuda()

        self.mask_predict = torch.nn.Conv3d(in_channels=num_fmaps, out_channels=1, kernel_size=1)  # 最终输出层的卷积操作

    def forward(self, x_raw, y_affinity, y_mask2d_slice0):
        '''
        x_raw: shape = (Batch * channel * dim_x * dim_y * dim_z)
        '''
        # replace raw slice0
        x_raw_new = deepcopy(x_raw)
        x_raw_new[:, 0, :, :, 0] = (y_mask2d_slice0.detach().squeeze() > 0.5) + 0.
        y_concat = torch.cat([x_raw_new, y_affinity.detach()], dim=1)

        y_mask3d_logits = self.mask_predict(self.model_mask(y_concat))

        return y_mask3d_logits



class segEM_3d(torch.nn.Module):
    def __init__(
            self,
            num_fmaps=21,
            fmap_inc_factor=4
    ):
        super(segEM_3d, self).__init__()

        ##2D slice mask prediction
        self.model_mask_2d = segEM_2d()
        # model_path = './output/checkpoints/segEM2d(hemi+fib25+cremi)_Best_in_val.model'
        # model_path = './output/checkpoints/segEM2d(hemi+fib25)faster_wloss3({})_Best_in_val.model'.format(WEIGHT_LOSS_AFFINITY)
        model_path = './output/log/segEM2d(ninanjie)-prompt-3rd-1/weighted-w2-5-w3-1/24_3_4/Best_in_val.model'
        weights = torch.load(model_path, map_location=torch.device('cuda'))
        self.model_mask_2d.load_state_dict(remove_module(weights))
        for param in self.model_mask_2d.parameters():
            param.requires_grad = False

        ##For affinity prediction
        self.model_affinity = ACRLSD_3d(21, 5)
        # model_path = './output/checkpoints/ACRLSD_3D(hemi+fib25+cremi)_Best_in_val.model'
        model_path = './output/log/ACRLSD_3D(ninanjie)_all/21_5/Best_in_val.model'
        weights = torch.load(model_path, map_location=torch.device('cuda'))
        self.model_affinity.load_state_dict(remove_module(weights))
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 3 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = segEM_3d_single(num_fmaps=num_fmaps, fmap_inc_factor=fmap_inc_factor)

    def forward(self, x_raw, x_prompt):
        '''
        x_raw: shape = (Batch * channel * dim_x * dim_y * dim_z)
        x_prompt: shape = (Batch * channel * dim_x * dim_y * dim_z)
        '''
        ##Get mask for slice0
        y_mask2d_slice0, _, _ = self.model_mask_2d(x_raw[:, :, :, :, 0], x_prompt)

        ##Get affinity for raw
        y_lsds, y_affinity = self.model_affinity(x_raw)

        y_mask3d_logits = self.model_mask(x_raw, y_affinity, y_mask2d_slice0)

        return y_mask3d_logits, y_mask2d_slice0, y_affinity, y_lsds


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


class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, foreground_weight):
        # 仅关注前景的Dice损失（target=1的区域）
        pred_fg = pred * target  # 前景预测
        target_fg = target  # 前景标签

        # 加权计算交集和并集（前景权重更高）
        intersection = (pred_fg * target_fg * foreground_weight).sum()
        union = (pred_fg * foreground_weight).sum() + target_fg.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice  # 损失值越小越好


def model_step(model, optimizer, input_image, input_prompt, gt_binary_mask, gt_affinity, activation,
               scaler=None, train_step=True, weighted=False):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    # lsd_logits,affinity_logits = model(raw)
    with torch.cuda.amp.autocast():
        y_mask_logits, y_mask2d_slice0, y_affinity, _ = model(input_image, input_prompt)
        y_mask = activation(y_mask_logits)

        gt_binary_mask = torch.as_tensor(gt_binary_mask, dtype=torch.float32, device=gt_binary_mask.device)

        weight = 1.0
        if weighted:
            foreground = gt_binary_mask.sum()
            total = gt_binary_mask.numel()
            background = total - foreground
            weight = background / foreground

        weights = torch.where(torch.as_tensor(gt_binary_mask == 1),
                              torch.as_tensor(weight, device=device), torch.tensor(1.0, device=device))
        loss1 = F.binary_cross_entropy_with_logits(y_mask_logits.squeeze(), gt_binary_mask.squeeze(), weight=weights)
        Diceloss_fn = WeightedDiceLoss().to(device)
        loss2 = Diceloss_fn(1 - y_mask.squeeze(), 1 - gt_binary_mask.squeeze(), weights)

        loss3 = torch.sum(y_mask * gt_affinity) / torch.sum(gt_affinity)

        loss = loss1 * WEIGHT_LOSS1 + loss2 * WEIGHT_LOSS2 + loss3 * WEIGHT_LOSS3

    # backward if training mode
    if train_step:
        scaler.scale(loss).backward()  # 替代 loss_value.backward()
        scaler.step(optimizer)  # 替代 optimizer.step()
        scaler.update()
    else:
        y_mask = (y_mask.squeeze() > 0.5) + 0.
        y_mask2d_slice0 = (y_mask2d_slice0 > 0.5) + 0.
        y_affinity = activation(y_affinity)

    # lsd_output = activation(lsd_logits)
    # affinity_output = activation(affinity_logits)
    # outputs = {
    #     'pred_lsds': lsd_output,
    #     'lsds_logits': lsd_logits,
    #     'pred_affinity': affinity_output,
    #     'affinity_logits': affinity_logits,
    # }

    return loss, (y_mask, y_mask2d_slice0, y_affinity), (loss1, loss2, loss3)


def _load_datasets(dataset_name_, crop_size_, crop_xyz_, a, b, c):
    # 确保加载的数据集张量默认在CPU（核心：避免数据集提前占用GPU）
    dataset = load_train_dataset(
        dataset_name_,
        raw_dir='raw_2',
        label_dir='truth_label_2_seg_1',
        from_temp=True,
        require_xz_yz=False,
        require_lsd=True,
        crop_size=crop_size_,
        crop_xyz=crop_xyz_,
        chunk_position=[a, b, c]
    )
    # 额外校验：强制数据集内所有张量在CPU（防止load_train_dataset内部误移GPU）
    # for data in dataset:  # 若dataset是自定义类，需确保__getitem__返回CPU张量；此处为通用校验
    #     for k, v in data.items():  # 假设data是字典格式，若为元组需对应调整
    #         if isinstance(v, torch.Tensor) and v.is_cuda:
    #             v = v.cpu()
    return dataset


def load_data(local_rank):  # 1. 新增local_rank参数：区分主进程/从进程，用于分布式配置
    dataset_names = ['best_val_3_3d_cpu']
    crop_size = 384
    crop_xyz = [9, 9, 1]
    train_dataset, val_dataset = None, None  # 2. 初始化改为None，避免非主进程空列表占用内存

    # -------------------------- 主进程（local_rank=0）加载数据 --------------------------
    if local_rank == 0:
        train_list_temp, val_list_temp = [], []
        for dataset_name, i, j, k in itertools.product(
                dataset_names,
                range(crop_xyz[0]),
                range(crop_xyz[1]),
                range(crop_xyz[2])
        ):
            train_, val_ = _load_datasets(dataset_name, crop_size, crop_xyz, i, j, k)
            train_list_temp.append(train_)
            val_list_temp.append(val_)

        random.shuffle(train_list_temp)  # 仅主进程打乱训练集分片
        train_dataset = ConcatDataset(train_list_temp)  # 合并为完整训练集（CPU）
        val_dataset = ConcatDataset(val_list_temp)      # 合并为完整验证集（CPU）
        print(f"主进程加载完成：训练集样本数{len(train_dataset)}，验证集样本数{len(val_dataset)}")

    # -------------------------- 强制CPU广播：避免GPU内存占用 --------------------------
    # 包装为列表（broadcast_object_list要求输入是列表，且原地修改）
    train_broadcast = [train_dataset]  # 主进程：[ConcatDataset]；从进程：[None]
    val_broadcast = [val_dataset]

    # 关键：用CPU上下文管理器，强制序列化后的张量在CPU，不占GPU内存
    dist.broadcast_object_list(train_broadcast, src=0, device=torch.device('cpu'))  # 主进程→所有进程广播训练集
    dist.broadcast_object_list(val_broadcast, src=0, device=torch.device('cpu'))    # 主进程→所有进程广播验证集

    # 提取广播后的数据集（此时所有进程的train_broadcast[0]都是主进程的ConcatDataset）
    train_dataset = train_broadcast[0]
    val_dataset = val_broadcast[0]

    # -------------------------- 分布式采样器配置（修正分片逻辑） --------------------------
    # 训练集：开启shuffle（保证每个epoch数据打乱）；验证集：关闭shuffle（结果可复现）
    train_sampler = DistributedSampler(
        train_dataset,
        rank=local_rank,                # 当前进程的rank（必传，确保分片正确）
        num_replicas=dist.get_world_size(),  # 总进程数（必传，计算分片数量）
        shuffle=True,                   # 训练集必须打乱
        drop_last=False                 # 根据需求选择是否丢弃最后一个不完整batch
    )
    val_sampler = DistributedSampler(
        val_dataset,
        rank=local_rank,
        num_replicas=dist.get_world_size(),
        shuffle=False,                  # 验证集不打乱
        drop_last=False
    )

    # -------------------------- 数据加载器（优化传输效率） --------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,          # 用分布式采样器，禁止同时设shuffle=True
        num_workers=14,                 # 保持原配置，若CPU内存不足可适当减少
        pin_memory=True,                # 开启：加速CPU→GPU数据传输（仅当数据在CPU时有效）
        drop_last=False,
        collate_fn=collate_fn_3D_ninanjie_Train
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=14,
        pin_memory=True,                # 修正原False为True，提升传输效率（无GPU内存代价）
        collate_fn=collate_fn_3D_ninanjie_Train
    )

    return train_loader, val_loader


def trainer_distribute(num_fmaps, fmap_inc_factors, weighted=False):
    # 1. 所有进程共享：模型初始化与分布式包装（每个进程都需创建模型并绑定到对应GPU）
    model = segEM_3d(num_fmaps, fmap_inc_factors)
    model = DistributedDataParallel(
        model.cuda(),
        device_ids=[local_rank],  # 当前进程绑定的GPU
        output_device=local_rank,  # 输出设备改为当前进程的GPU（原0号卡聚合可能导致负载不均，建议改成本地）
        broadcast_buffers=False
    )
    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = load_data(local_rank)

    # -------------------------- 主进程专属：日志/目录/TensorBoard初始化 --------------------------
    writer = None  # 初始化writer，避免非主进程访问未定义变量
    log_dir_ = None
    if local_rank == 0:
        # 日志目录创建与清理
        log_dir_ = './output/log/segEM_3d(ninanjie)-prompt-3rd-1/'
        log_dir_ += (f'weighted-' if weighted else '') + f'w2-{WEIGHT_LOSS2}-w3-{WEIGHT_LOSS3}/{num_fmaps}_{fmap_inc_factors}'
        os.makedirs(log_dir_, exist_ok=True)
        # 清理目录下旧文件
        for filename in os.listdir(log_dir_):
            file_path = os.path.join(log_dir_, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # 日志系统初始化
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logfile = f'{log_dir_}/log.txt'
        writer = SummaryWriter(log_dir=log_dir_)  # TensorBoard写入器
        # 日志文件处理器
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # 打印训练参数（仅主进程输出）
        logging.info(f'''Starting training:
            training_epochs:  {training_epochs}
            Batch size:       {batch_size}
            Learning rate:    {learning_rate}
            Device:           {device.type}
            Local Rank:       {local_rank}
            ''')
    # ------------------------------------------------------------------------------------------

    # 2. 所有进程共享：优化器、激活函数初始化（每个进程需独立创建优化器）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    activation = torch.nn.Sigmoid()

    # 3. 所有进程共享：训练超参数初始化（但仅主进程用Best_val_loss等做保存判断）
    model.train()
    epoch = 0
    Best_val_loss = 100000  # 初始最佳损失（所有进程同步初始值，避免判断偏差）
    Best_epoch = 0
    early_stop_count = 50
    no_improve_count = 0

    # -------------------------- 主进程专属：进度条初始化（避免多进程重复打印进度条） --------------------------
    pbar = tqdm(total=training_epochs) if local_rank == 0 else None
    try:
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            ################### 阶段1：训练（所有进程同步执行，数据并行加载） ###################
            model.train()
            # 每个进程独立设置随机种子（保证数据增强随机性不同）
            np.random.seed(epoch + local_rank)  # 加local_rank避免多进程种子重复
            random.seed(epoch + local_rank)
            tmp_loader = iter(train_loader)
            count, total = 0, len(tmp_loader)

            for raw, labels, mask, affinity, point_map in tmp_loader:
                # 数据转Tensor并绑定到当前进程的GPU（所有进程执行）
                raw = torch.as_tensor(raw, dtype=torch.float, device=device)
                point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)
                mask = torch.as_tensor(mask, dtype=torch.float, device=device)
                affinity = torch.as_tensor(affinity, dtype=torch.float, device=device)

                # 模型训练步骤（所有进程同步执行，分布式反向传播）
                model_step(
                    model, optimizer, raw, point_map, mask, affinity, activation,
                    scaler, train_step=True, weighted=weighted
                )

                count += 1
                # -------------------------- 主进程专属：更新训练进度条描述 --------------------------
                if local_rank == 0:
                    train_desc = f"loss: {Best_val_loss:.4f} in {Best_epoch}, "
                    progress_desc = f"Train {count}/{total}, "
                    pbar.set_description(progress_desc + train_desc + val_desc)

            ################### 阶段2：验证（所有进程同步执行，但仅主进程记录指标/可视化） ###################
            model.eval()
            # 验证集固定种子（所有进程用相同种子，保证验证数据一致）
            seed = 98  # 加epoch避免每次验证数据相同
            np.random.seed(seed)
            random.seed(seed)
            tmp_val_loader = iter(val_loader)
            acc_loss = []
            detailed_losses = np.array([0.] * 3)
            voi_val = 0.
            judgement_rates = np.array([0.] * 4)
            count, total = 0, len(tmp_val_loader)

            with torch.no_grad():  # 验证阶段禁用梯度（所有进程执行）
                for raw, labels, mask, affinity, point_map in tmp_val_loader:
                    # 数据转Tensor（所有进程执行）
                    raw = torch.as_tensor(raw, dtype=torch.float, device=device)
                    point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)
                    mask = torch.as_tensor(mask, dtype=torch.float, device=device)
                    affinity = torch.as_tensor(affinity, dtype=torch.float, device=device)

                    # 模型验证步骤（所有进程执行，计算损失和预测结果）
                    loss_value, pred, losses = model_step(
                        model, optimizer, raw, point_map, mask, affinity, activation,
                        scaler, train_step=False, weighted=weighted
                    )

                    # 计算验证指标（所有进程执行，但仅主进程汇总）
                    binary_y_pred = np.asarray(pred.cpu(), dtype=np.uint8).flatten()
                    binary_gt_mask = np.asarray(mask.cpu(), dtype=np.uint8).flatten()
                    voi_val += np.sum(variation_of_information(binary_y_pred, binary_gt_mask))
                    judgement_rates += np.asarray(get_acc_prec_recall_f1(binary_y_pred, binary_gt_mask))
                    acc_loss.append(loss_value.cpu().detach().numpy())
                    detailed_losses += np.asarray([loss_.cpu().detach().numpy() for loss_ in losses])

                    count += 1
                    # -------------------------- 主进程专属：更新验证进度条描述 --------------------------
                    if local_rank == 0:
                        progress_desc = f'Val: {count}/{total}, '
                        pbar.set_description(progress_desc + train_desc + val_desc)

            # -------------------------- 主进程专属：验证指标汇总+可视化+模型保存 --------------------------
            if local_rank == 0:
                # 1. 指标平均值计算
                detailed_losses /= (total + 1e-8)  # 加1e-8避免除零
                bce_, dice_, affinity_ = detailed_losses[:]
                voi_val /= (total + 1e-8)
                judgement_rates /= (total + 1e-8)
                acc, prec, recall, f1 = judgement_rates[:]
                val_loss = torch.as_tensor(acc_loss).mean().item()  # 验证集平均损失
                val_desc = f'acc: {acc:.3f}, prec: {prec:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, VOI: {voi_val:.5f}'

                # 2. 可视化分割结果（仅主进程保存图片到TensorBoard）
                sample_idx = random.randint(0, raw.shape[1] - 1)  # 随机选一个样本可视化
                # 数据格式转换（适配可视化函数）
                raw_vis = raw.squeeze()[sample_idx, ...].transpose(2, 0, 1)[-2, ...].cpu().numpy()
                pred_vis = pred.squeeze()[sample_idx, ...].transpose(2, 0, 1)[-2, ...].cpu().numpy()
                label_vis = labels.squeeze()[sample_idx, ...].transpose(2, 0, 1)[-2, ...].cpu().numpy()
                # 调用可视化函数（写入TensorBoard）
                visualize_and_save_mask(raw_vis, pred_vis, mode='visual/seg_only', idx=epoch, writer=writer)
                visualize_and_save_mask(raw_vis, pred_vis, mode='visual/raw_seg', idx=epoch, writer=writer)
                visualize_and_save_mask(raw_vis, label_vis, mode='visual/gt', idx=epoch, writer=writer)
                # 后处理后的可视化
                pred_aftercare = aftercare(torch.as_tensor(pred_vis).unsqueeze(0).unsqueeze(0).unsqueeze(4)).squeeze()
                visualize_and_save_mask(raw_vis, pred_aftercare, mode='aftercare/seg_only', idx=epoch, writer=writer)
                visualize_and_save_mask(raw_vis, pred_aftercare, mode='aftercare/raw_seg', idx=epoch, writer=writer)

                # 3. 最佳模型保存判断
                if Best_val_loss > val_loss:
                    Best_val_loss = val_loss
                    Best_epoch = epoch
                    # 保存分布式模型的module参数（避免保存DDP包装器）
                    torch.save(model.module.state_dict(), f'{log_dir_}/Best_in_val.model')
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # 4. 指标记录（TensorBoard + 日志文件）
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar('loss/bce', bce_, epoch)
                writer.add_scalar('loss/dice', dice_, epoch)
                writer.add_scalar('loss/affinity', affinity_, epoch)
                writer.add_scalar('metrics/accuracy', acc, epoch)
                writer.add_scalar('metrics/precision', prec, epoch)
                writer.add_scalar('metrics/recall', recall, epoch)
                writer.add_scalar('metrics/f1_score', f1, epoch)
                writer.add_scalar('VOI', voi_val, epoch)
                logging.info(f"Epoch {epoch}: val_loss = {val_loss:.6f}, best val_loss = {Best_val_loss:.6f} in epoch {Best_epoch}")

                # 5. 早停判断（仅主进程输出早停日志）
                if no_improve_count == early_stop_count:
                    logging.info("Early stop!")
                    break

            # -------------------------- 所有进程同步：更新epoch和进度条 --------------------------
            epoch += 1
            if local_rank == 0:
                pbar.update(1)  # 仅主进程更新进度条

    finally:
        # -------------------------- 主进程专属：资源清理（避免内存泄漏） --------------------------
        if local_rank == 0:
            pbar.close()  # 关闭进度条
            writer.close()  # 关闭TensorBoard写入器
            # 保存最终模型（仅主进程执行）
            if log_dir_ is not None:
                torch.save(model.module.state_dict(), f'{log_dir_}/final.model')

    # 所有进程共享：资源清理（每个进程独立释放GPU内存）
    del model, optimizer, activation, scaler
    torch.cuda.empty_cache()
    gc.collect()


def trainer(num_fmaps, fmap_inc_factors, weighted=False, use_single=False):
    model = segEM_3d_single(num_fmaps, fmap_inc_factors) if use_single else segEM_3d(num_fmaps, fmap_inc_factors)
    model = nn.DataParallel(model.cuda(), device_ids=device_ids, output_device=device_ids[0])
    scaler = torch.cuda.amp.GradScaler()
    log_dir_ = './output/log/segEM_3d(ninanjie)-prompt-3rd-1/'
    log_dir_ += ((f'single-' if use_single else '') + (f'weighted-' if weighted else '')
                 + f'w2-{WEIGHT_LOSS2}-w3-{WEIGHT_LOSS3}/{num_fmaps}_{fmap_inc_factors}')
    os.makedirs(log_dir_, exist_ok=True)
    for filename in os.listdir(log_dir_):
        file_path = os.path.join(log_dir_, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    ##创建log日志
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    logfile = '{}/log.txt'.format(log_dir_)
    writer = SummaryWriter(log_dir=log_dir_)
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info(f'''Starting training:
                training_epochs:  {training_epochs}
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
    early_stop_count = 50
    no_improve_count = 0
    with tqdm(total=training_epochs) as pbar:
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
            for raw, labels, mask, affinity, point_map in tmp_loader:
                ##Get Tensor
                raw = torch.as_tensor(raw,dtype=torch.float, device= device)
                point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)
                mask = torch.as_tensor(mask, dtype=torch.float, device=device)
                affinity = torch.as_tensor(affinity, dtype=torch.float, device=device) #(batch, 2, height, width)
                model_step(model, optimizer, raw, point_map, mask, affinity, activation,
                           scaler, train_step=True, weighted=weighted, use_single=use_single)

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
            voi_val, aftercare_voi_val = 0., 0.
            judgement_rates = np.array([0.] * 4)
            aftercare_judgement_rates = np.array([0.] * 4)
            count, total = 0, len(tmp_val_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            for raw, labels, mask, affinity, point_map in tmp_val_loader:
                raw = torch.as_tensor(raw, dtype=torch.float, device=device) #(batch, 1, height, width, depth)
                point_map = torch.as_tensor(point_map, dtype=torch.float, device=device) #(batch, height, width, depth)
                mask = torch.as_tensor(mask, dtype=torch.float, device=device) #(batch, 2, height, width, depth)
                affinity = torch.as_tensor(affinity, dtype=torch.float, device=device) #(batch, 3, height, width, depth)

                with torch.no_grad():
                    loss_value, (pred, pred_2d, pred_affinity), losses = model_step(model, optimizer, raw, point_map,
                                                                                    mask, affinity, activation, scaler,
                                                                                    train_step=False, weighted=weighted)

                    binary_y_pred = np.asarray(pred.cpu(), dtype=np.uint8).flatten()
                    binary_gt_mask = np.asarray(mask.cpu(), dtype=np.uint8).flatten()
                    y_pred_aftercare = aftercare(pred.squeeze())
                    binary_y_pred_aftercare = np.asarray(y_pred_aftercare, dtype=np.uint8).flatten()

                    voi_val += np.sum(variation_of_information(binary_y_pred, binary_gt_mask))
                    aftercare_voi_val += np.sum(variation_of_information(binary_y_pred_aftercare, binary_gt_mask))
                    judgement_rates += np.asarray(get_acc_prec_recall_f1(binary_y_pred, binary_gt_mask))
                    aftercare_judgement_rates += np.asarray(get_acc_prec_recall_f1(binary_y_pred_aftercare,
                                                                                  binary_gt_mask))
                    acc_loss.append(loss_value.cpu().detach().numpy())
                    detailed_losses += np.asarray([loss_.cpu().detach().numpy() for loss_ in losses])

                count += 1
                progress_desc = f'Val: {count}/{total}, '
                pbar.set_description(progress_desc + train_desc + val_desc)

            detailed_losses /= (total + 0.)
            bce_, dice_, affinity_ = detailed_losses[:]
            voi_val /= (total + 0.)
            aftercare_voi_val /= (total + 0.)
            judgement_rates /= (total + 0.)
            aftercare_judgement_rates /= (total + 0.)
            acc, prec, recall, f1 = judgement_rates[:]
            aftercare_acc, aftercare_prec, aftercare_recall, aftercare_f1 = aftercare_judgement_rates[:]
            val_loss = torch.as_tensor([loss_value.item() for loss_value in acc_loss]).mean().item()
            val_desc = f'acc: {acc:.3f}, prec: {prec:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, VOI: {voi_val:.5f}'

            sample_idx = random.randint(0, raw.shape[0] - 1)
            raw_ = raw.squeeze()[sample_idx, ..., -2].cpu().numpy()
            raw_slice_0 = raw.squeeze()[sample_idx, ..., 0].cpu().numpy()
            pred = pred.squeeze()[sample_idx, ..., -2].cpu().numpy() # (H, W)
            pred_2d = pred_2d.squeeze()[sample_idx, ...].cpu().numpy() # (B, H, W) -> (H, W)
            pred_affinity_np = np.asarray(pred_affinity[sample_idx, :, :, :, -2].cpu().numpy() * 255.0,
                                          dtype=np.uint8)  # (B, 3, H, W, N) -> (3, H, W)
            gt_affinity_np = affinity[sample_idx, :, :, :, -2].cpu().numpy() * 255.0
            label = labels.squeeze()[sample_idx, ..., -2]
            gt_label_2d = labels.squeeze()[sample_idx, ..., 0]
            visualize_and_save_mask(raw_, pred, mode='y_mask/seg_only', idx=epoch, writer=writer)
            visualize_and_save_mask(raw_, pred, mode='y_mask/raw_seg', idx=epoch, writer=writer)
            visualize_and_save_mask(raw_, label, mode='y_mask/gt', idx=epoch, writer=writer)
            visualize_and_save_mask(raw_slice_0, pred_2d, mode='slice_0/seg_only', idx=epoch, writer=writer)
            visualize_and_save_mask(raw_slice_0, pred_2d, mode='slice_0/raw_seg', idx=epoch, writer=writer)
            visualize_and_save_mask(raw_slice_0, gt_label_2d, mode='slice_0/gt', idx=epoch, writer=writer)

            visual_pred_affinities = [normalize_affinity(np.sum(pred_affinity_np[layer, ...], axis=0))
                                      for layer in ([0, 1], [0, 2], [1, 2])]
            visual_gt_affinities = [normalize_affinity(np.sum(gt_affinity_np[layer, ...], axis=0))
                                    for layer in ([0, 1], [0, 2], [1, 2])]

            visualize_and_save_affinity(visual_pred_affinities[0], epoch, 'y_affinity/pred_xy', writer)
            visualize_and_save_affinity(visual_gt_affinities[0], epoch, 'y_affinity/gt_xy', writer)
            visualize_and_save_affinity(visual_pred_affinities[1], epoch, 'y_affinity/pred_xz', writer)
            visualize_and_save_affinity(visual_gt_affinities[1], epoch, 'y_affinity/gt_xz', writer)
            visualize_and_save_affinity(visual_pred_affinities[2], epoch, 'y_affinity/pred_yz', writer)
            visualize_and_save_affinity(visual_gt_affinities[2], epoch, 'y_affinity/gt_yz', writer)

            pred = y_pred_aftercare.squeeze()[sample_idx, ..., -2]
            visualize_and_save_mask(raw_, pred, mode='aftercare/seg_only', idx=epoch, writer=writer)
            visualize_and_save_mask(raw_, pred, mode='aftercare/raw_seg', idx=epoch, writer=writer)
            writer.add_scalar('aftercare/accuracy', aftercare_acc, epoch)
            writer.add_scalar('aftercare/precision', aftercare_prec, epoch)
            writer.add_scalar('aftercare/recall', aftercare_recall, epoch)
            writer.add_scalar('aftercare/f1_score', aftercare_f1, epoch)
            writer.add_scalar('aftercare/VOI', aftercare_voi_val, epoch)

            writer.flush()

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.state_dict(), '{}/Best_in_val.model'.format(log_dir_))
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

            ##Early stop
            if no_improve_count == early_stop_count:
                logging.info("Early stop!")
                break

    torch.save(model.module.state_dict(), f'{log_dir_}/final.model')
    del model, optimizer, activation
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 18
    # Save_Name = 'segEM_3d(hemi+fib25+cremi)'
    # Save_Name = 'segEM_3d(hemi+fib25)faster_wloss3({})'.format(WEIGHT_LOSS_AFFINITY)

    set_seed()

    ###创建模型
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,5,0,3'  # 设置所有可以使用的显卡，共计四块
    device_ids = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]  # 选中显卡
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    dataset_names = ['best_val_3_3d_cpu']
    crop_size = 384
    crop_xyz = [9, 9, 1]
    train_dataset, val_dataset = [], []

    for dataset_name, i, j, k in itertools.product(
            dataset_names,
            range(crop_xyz[0]),
            range(crop_xyz[1]),
            range(crop_xyz[2])
    ):
        train_, val_ = _load_datasets(dataset_name, crop_size, crop_xyz, i, j, k)
        train_dataset.append(train_)
        val_dataset.append(val_)

    sys.exit(0)
    train_dataset, val_dataset = ConcatDataset(train_dataset), ConcatDataset(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn_3D_ninanjie_Train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
                            collate_fn=collate_fn_3D_ninanjie_Train)

    # 3. 超参数设置
    num_fmaps = 24
    fmap_inc_factors = 4
    learning_rate = 1e-4
    training_epochs = 1000

    # trainer(num_fmaps, fmap_inc_factors)
    trainer(num_fmaps, fmap_inc_factors, weighted=True, single=True)


# if __name__ == '__main__':
#     set_seed()
#
#     # 显式设置环境变量（关键）
#     os.environ['CUDA_VISIBLE_DEVICES'] = "4,1,5,0,3"  # 使用所有6张卡
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29502'
#     gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
#     dist.init_process_group(
#         backend='nccl',
#         init_method='env://',
#         world_size=gpu_num,  # 总进程数=显卡数=6
#         rank=int(os.environ['RANK'])
#     )
#
#     # 2. 每个进程绑定到不同显卡（0-5号卡）
#     local_rank = int(os.environ['LOCAL_RANK'])
#     torch.cuda.set_device(local_rank)
#     device = torch.device('cuda', local_rank)
#
#     # 3. 超参数设置
#     num_fmaps = 12
#     fmap_inc_factors = 4
#     batch_size = 1  # 单进程batch size，总batch size要乘显卡数
#     learning_rate = 1e-4
#     training_epochs = 10000
#
#     trainer_distribute(num_fmaps, fmap_inc_factors)
#     trainer_distribute(num_fmaps, fmap_inc_factors, weighted=True)
