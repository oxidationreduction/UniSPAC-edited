import logging
import os
import random
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

# from torchmetrics import functional
from models.unet2d import UNet2d
from training.utils.dataloader_ninanjie import Dataset_2D_ninanjie_Train, load_dataset, collate_fn_2D_fib25_Train
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
    def focal_loss(logits, targets, gamma=1.):
        pt = torch.sigmoid(logits)
        loss = -torch.mean((1.-pt) ** gamma * (targets * torch.log(pt + 1e-10) + (1-targets) * torch.log(1. - pt + 1e-10)))

        return loss

    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    lsd_logits, affinity_logits = model(raw)

    loss_value = loss_fn(lsd_logits, gt_lsds) + loss_fn(affinity_logits, gt_affinity)

    # loss_value = (loss_fn(lsd_logits, gt_lsds)
    #               + loss_fn(affinity_logits[gt_affinity == 0], gt_affinity[gt_affinity == 0])
    #               + loss_fn(affinity_logits[gt_affinity == 1], gt_affinity[gt_affinity == 1]))

    # loss_value = loss_fn(lsd_logits, gt_lsds) + focal_loss(affinity_logits, gt_affinity, gamma=1.)

    # backward if training mode
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

    results_ = {}
    if not train_step:
        affinity_output = (affinity_output.squeeze() >= 0.5).float()
        gt_affinity = gt_affinity.squeeze()
        # results_['accuracy'] = functional.accuracy(affinity_output, gt_affinity, task='binary').item()
        # results_['precision'] = functional.precision(affinity_output, gt_affinity, task='binary').item()
        # results_['recall'] = functional.recall(affinity_output, gt_affinity, task='binary').item()
        # results_['f1_score'] = functional.f1_score(affinity_output, gt_affinity, task='binary').item()
        # 计算真正例(TP)、假正例(FP)、真反例(TN)、假反例(FN)
        TP = torch.sum((affinity_output == 1) & (gt_affinity == 1)).float()
        FP = torch.sum((affinity_output == 1) & (gt_affinity == 0)).float()
        TN = torch.sum((affinity_output == 0) & (gt_affinity == 0)).float()
        FN = torch.sum((affinity_output == 0) & (gt_affinity == 1)).float()

        # 计算分母（避免除零）
        total = TP + FP + TN + FN
        positives = TP + FN
        predicted_positives = TP + FP

        # 计算四项指标
        results_['accuracy'] = (TP + TN) / total if total > 0 else torch.tensor(0.0)
        results_['precision'] = TP / predicted_positives if predicted_positives > 0 else torch.tensor(0.0)
        results_['recall'] = TP / positives if positives > 0 else torch.tensor(0.0)
        results_['f1_score'] = 2 * (results_['precision'] * results_['recall']) / (
                    results_['precision'] + results_['recall']) \
            if (results_['precision'] + results_['recall']) > 0 else torch.tensor(0.0)

        # 转换为标量值
        results_ = {k: v.item() for k, v in results_.items()}

    return (loss_value, outputs) if train_step else (loss_value, outputs, results_)


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 192

    set_seed()

    ###创建模型
    # set device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = ACRLSD()

    ##单卡
    # model = model.to(device)

    ##多卡训练
    gpus = [3,4,5]#选中显卡
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    Save_Name = 'ACRLSD_2D(ninanjie)'

    ##装载数据
    # train_dataset_1 = Dataset_2D_hemi_Train(data_dir='./data/hemi/training/', split='train', crop_size=128,
    #                                         require_lsd=True, require_xz_yz=True)
    # val_dataset_1 = Dataset_2D_hemi_Train(data_dir='./data/hemi/training/', split='val', crop_size=128,
    #                                       require_lsd=True, require_xz_yz=True)

    ninanjie_data = './data/ninanjie'
    ninanjie_save = './data/ninanjie-save'
    train_dataset, val_dataset = load_dataset('first', require_xz_yz=True, from_temp=True,
                                              crop_xyz=[3,3,2], chunk_position=[1,1,0])

    # train_dataset_3 = Dataset_2D_cremi_Train(data_dir='../data/CREMI/', split='train', crop_size=128, require_lsd=True)
    # val_dataset_3 = Dataset_2D_cremi_Train(data_dir='../data/CREMI/', split='val', crop_size=128, require_lsd=True)

    # train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    # val_dataset = ConcatDataset([val_dataset_1, val_dataset_2])

    # train_dataset = train_dataset_1
    # val_dataset = val_dataset_1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=48, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn_2D_fib25_Train)
    val_loader = DataLoader(val_dataset, batch_size=(batch_size // 2) + 1, shuffle=False, num_workers=48, pin_memory=True,
                            collate_fn=collate_fn_2D_fib25_Train)

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
    logger.setLevel(logging.INFO)
    logfile = './output/log/log_{}.txt'.format(Save_Name)
    csvfile = './output/log/log_{}.csv'.format(Save_Name)
    fh = logging.FileHandler(logfile, mode='a', delay=False)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

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
        analysis = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1'])
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
                train_desc = f"loss: {Best_val_loss:.2f}, "
                loss_value, pred = model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation)
                pbar.set_description(progress_desc + train_desc + val_desc)

            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 98
            np.random.seed(seed)
            random.seed(seed)
            acc_loss = []
            count, total = 0, len(val_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            for raw, labels, point_map, mask, gt_affinity, gt_lsds in val_loader:
                # raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
                # gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
                # gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                #                               device=device)  # (batch, 2, height, width)
                with torch.no_grad():
                    loss_value, _, results = model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation,
                                               train_step=False)
                count += 1
                acc_loss.append(loss_value)
                progress_desc = f"Val {count}/{total}, "
                analysis.loc[len(analysis)] = [results['accuracy'], results['precision'], results['recall'], results['f1_score']]

            # val_loss = np.mean(np.array([loss_value.cpu().numpy() for loss_value in acc_loss]))
            val_loss = torch.stack([loss_value.cpu() for loss_value in acc_loss]).mean().item()

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                Best_model = model.state_dict()
                no_improve_count = 0
            else:
                no_improve_count += 1

            val_desc = (f"acc: {results['accuracy']:.4f}, prec: {results['precision']:.4f}, "
                        f"recall: {results['recall']:.4f}, f1: {results['f1_score']:.4f}  ({epoch - Best_epoch})")
            pbar.set_description(progress_desc + train_desc + val_desc)

            ## Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, Best_val_loss, Best_epoch))

            pbar.update(1)

            ##Early stop
            if no_improve_count == early_stop_count and epoch > 100:
                logging.info("Early stop!")
                break
    fh.flush()
    ch.flush()
    # writer.add_scalar('val_loss', val_loss, epoch)
    analysis.to_csv(csvfile)
    torch.save(Best_model, './output/checkpoints/{}_Best_in_val.model'.format(Save_Name))
