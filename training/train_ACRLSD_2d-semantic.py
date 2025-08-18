import logging
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from skimage.metrics import variation_of_information
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

# from torchmetrics import functional
from models.unet2d import UNet2d
from training.utils.dataloader_ninanjie import Dataset_2D_ninanjie_Train, load_train_dataset, collate_fn_2D_fib25_Train, \
    get_acc_prec_recall_f1, load_semantic_dataset
from utils.dataloader_ninanjie import Dataset_2D_ninanjie_Train
from utils.dataloader_hemi_better import Dataset_2D_hemi_Train, collate_fn_2D_hemi_Train


# from utils.dataloader_cremi import Dataset_2D_cremi_Train,collate_fn_2D_cremi_Train

## CUDA_VISIBLE_DEVICES=1 python train_ACRLSD_2d.py &

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


def model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation, train_step=True):
    if train_step:
        optimizer.zero_grad()

    # forward
    lsd_logits, affinity_logits = model(raw)

    loss_value = loss_fn(lsd_logits, gt_lsds) + loss_fn(affinity_logits, gt_affinity)

    if train_step:
        loss_value.backward()
        optimizer.step()

    lsd_output = activation(lsd_logits)
    affinity_output = activation(affinity_logits)

    outputs = {
        'pred_lsds': lsd_output,
        'lsds_logits': lsd_logits,
        'pred_affinity': affinity_output,
        'affinity_logits': affinity_logits,
    }

    if not train_step:
        outputs.update({
            'voi': sum(variation_of_information(affinity_output.squeeze(), gt_affinity.squeeze()))
        })

    return loss_value, outputs


def weighted_model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation, train_step=True):
    if train_step:
        optimizer.zero_grad()

    # 前向传播
    lsd_logits, affinity_logits = model(raw)

    # --------------------------
    # 1. 提取前景区域并计算权重
    # --------------------------
    # 假设亲和度图中前景相关区域为非零值（可根据实际标签定义调整）
    # 生成前景掩码（1表示前景相关区域，0表示背景）
    fg_mask_affinity = (gt_affinity > 0).float()  # 亲和度前景掩码
    fg_mask_lsds = (gt_lsds > 0).float()  # LSDS前景掩码（根据实际标签调整）

    # 计算前景/背景像素数量（亲和度）
    fg_pixels_affinity = fg_mask_affinity.sum()
    bg_pixels_affinity = fg_mask_affinity.numel() - fg_pixels_affinity
    # 计算前景权重（避免除零）
    fg_weight_affinity = bg_pixels_affinity / (fg_pixels_affinity + 1e-5) if fg_pixels_affinity > 0 else 1.0

    # 计算前景/背景像素数量（LSDS）
    fg_pixels_lsds = fg_mask_lsds.sum()
    bg_pixels_lsds = fg_mask_lsds.numel() - fg_pixels_lsds
    # 计算前景权重
    fg_weight_lsds = bg_pixels_lsds / (fg_pixels_lsds + 1e-5) if fg_pixels_lsds > 0 else 1.0

    # 生成像素级权重矩阵（前景区域用fg_weight，背景用1.0）
    weights_affinity = torch.where(fg_mask_affinity == 1.0, fg_weight_affinity, torch.tensor(1.0, device=device)).to(device)
    weights_lsds = torch.where(fg_mask_lsds == 1.0, fg_weight_lsds, torch.tensor(1.0, device=device)).to(device)

    # --------------------------
    # 2. 计算加权损失
    # --------------------------
    # 定义加权损失计算函数（以MSE为例，可根据实际loss_fn类型调整）
    def weighted_loss(pred, target, weights):
        # 计算基础损失（如MSE）
        base_loss = loss_fn(pred, target)
        # 应用权重（对每个元素的损失加权）
        weighted = base_loss * weights
        return weighted.mean()  # 加权后的平均损失

    # 计算加权损失（LSDS和亲和度分别加权）
    loss_lsd = weighted_loss(lsd_logits, gt_lsds, weights_lsds)
    loss_affinity = weighted_loss(affinity_logits, gt_affinity, weights_affinity)

    # 总损失（可根据任务重要性调整两者比例）
    loss_value = loss_lsd + loss_affinity  # 或 loss_lsd * 0.5 + loss_affinity * 0.5

    # 反向传播与参数更新
    if train_step:
        loss_value.backward()
        optimizer.step()

    # 激活输出
    lsd_output = activation(lsd_logits)
    affinity_output = activation(affinity_logits)

    affinity_output = affinity_output.detach().cpu()
    gt_affinity = gt_affinity.cpu()

    outputs = {
        'pred_lsds': lsd_output,
        'lsds_logits': lsd_logits,
        'pred_affinity': affinity_output,
        'affinity_logits': affinity_logits,
    }
    if not train_step:
        outputs.update({
            'voi': sum(variation_of_information(np.asarray((affinity_output.squeeze() > 0.5) + 0),
                                                np.asarray(gt_affinity.squeeze(), dtype=np.int16)))
        })
    return loss_value, outputs


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 32

    set_seed()

    ###创建模型
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    gpus = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]

    model = ACRLSD().to(device)
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    dataset_names = ['second_6', 'fourth_1']

    Save_Name = f'ACRLSD_2D(ninanjie)_semantic'

    ##装载数据
    crop_xyz = [3, 3, 1]
    train_dataset, val_dataset = [], []
    for dataset_name in dataset_names:
        for i in range(crop_xyz[0]-1):
            for j in range(crop_xyz[1]):
                for k in range(crop_xyz[2]):
                    train_tmp, val_tmp = load_semantic_dataset(dataset_name, raw_dir='raw', label_dir='export',
                                                      require_xz_yz=False, from_temp=True, crop_size=600,
                                                      crop_xyz=crop_xyz, chunk_position=[i, j, k])
                    train_dataset.append(train_tmp)
                    val_dataset.append(val_tmp)

    train_dataset = ConcatDataset(train_dataset)
    val_dataset = ConcatDataset(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=48, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn_2D_fib25_Train)
    val_loader = DataLoader(val_dataset, batch_size=(batch_size // 2) + 1, shuffle=False, num_workers=48,
                            pin_memory=True, collate_fn=collate_fn_2D_fib25_Train)

    def load_data_to_device(loader):
        res = []
        for raw, labels, point_map, mask, gt_affinity, gt_lsds in loader:
            raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
            gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
            gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                                          device=device)  # (batch, 2, height, width)
            res.append((raw, labels, point_map, mask, gt_affinity, gt_lsds))
        return res

    train_loader, val_loader = load_data_to_device(train_loader), load_data_to_device(val_loader)

    ##创建log日志
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logfile = './output/log/log_{}.txt'.format(Save_Name)
    csvfile = './output/log/log_{}.csv'.format(Save_Name)
    fh = logging.FileHandler(logfile, mode='a', delay=False)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info(f'''Starting training:
    training_epochs:  {training_epochs}
    Num_slices_train: {len(train_dataset)}
    Num_slices_val:   {len(val_dataset)}
    Batch size:       {batch_size}
    Learning rate:    {learning_rate}
    Device:           {device.type}
    ''')

    ##开始训练
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # set activation
    activation = torch.nn.Sigmoid()
    # set loss function
    loss_fn = torch.nn.MSELoss().to(device)

    # training loop
    model.train()
    loss_fn.train()
    epoch = 0
    Best_val_loss = 100000
    Best_epoch = 0
    Best_model = None
    early_stop_count = 100
    no_improve_count = 0
    with tqdm(total=training_epochs) as pbar:
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            epoch += 1
            ###################Train###################
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            count, total = 0, len(train_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
            for raw, labels, point_map, mask, gt_affinity, gt_lsds in train_loader:
                ##Get Tensor
                # raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
                # gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
                # gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                #                               device=device)  # (batch, 2, height, width)
                count = count + 1
                progress_desc = f"Train {count}/{total}, "
                train_desc = f"loss: {Best_val_loss:.4f}, "
                loss_value, pred = weighted_model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation)
                pbar.set_description(progress_desc + train_desc + val_desc)

            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 98
            np.random.seed(seed)
            random.seed(seed)
            acc_loss, voi_vals = [], []
            metrics = np.array([0,0,0,0], dtype=np.float32)
            count, total = 0, len(val_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            for raw, labels, point_map, mask, gt_affinity, gt_lsds in val_loader:
                # raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
                # gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
                # gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                #                               device=device)  # (batch, 2, height, width)
                with torch.no_grad():
                    loss_value, output = weighted_model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation,
                                               train_step=False)
                    y_pred = np.asarray(output['pred_affinity'].detach().cpu()).squeeze().flatten()
                    gt_affinity = np.asarray(gt_affinity.detach().cpu()).squeeze().flatten()
                count += 1
                acc_loss.append(loss_value)
                voi_vals.append(output['voi'])
                metrics += np.asarray(get_acc_prec_recall_f1(y_pred, gt_affinity))
                progress_desc = f"Val {count}/{total}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

            # val_loss = np.mean(np.array([loss_value.cpu().numpy() for loss_value in acc_loss]))
            val_loss = torch.stack([loss_value.cpu() for loss_value in acc_loss]).mean().item()
            voi = np.mean(voi_vals)
            metrics /= (total + 0.)

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                Best_model = model.state_dict()
                no_improve_count = 0
            else:
                no_improve_count += 1

            val_desc = f'best {Best_epoch}, {"-".join([f"{metric:.4f}" for metric in metrics])}, VOI: {voi:.4f}'
            ## Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, Best_val_loss, Best_epoch))

            pbar.update(1)

            ##Early stop
            if no_improve_count == early_stop_count and epoch > 100:
                logging.info("Early stop!")
                break
    fh.flush()
    torch.save(Best_model, './output/checkpoints/{}_Best_in_val.model'.format(Save_Name))
