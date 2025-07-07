import logging
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from torchmetrics import functional
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from utils.dataloader_fib25_better import Dataset_2D_fib25_Train
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
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    lsd_logits, affinity_logits = model(raw)

    loss_value = loss_fn(lsd_logits, gt_lsds) + loss_fn(affinity_logits, gt_affinity)

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
        affinity_output = affinity_output.squeeze()
        gt_affinity = gt_affinity.squeeze()
        results_['accuracy'] = functional.accuracy(affinity_output, gt_affinity, task='binary')
        results_['precision'] = functional.precision(affinity_output, gt_affinity, task='binary')
        results_['recall'] = functional.recall(affinity_output, gt_affinity, task='binary')
        results_['f1_score'] = functional.f1_score(affinity_output, gt_affinity, task='binary')

    return (loss_value, outputs) if train_step else (loss_value, outputs, results_)


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 1000
    learning_rate = 1e-4
    batch_size = 80
    # Save_Name = 'ACRLSD_2D(hemi+fib25+cremi)'
    Save_Name = 'ACRLSD_2D(ninanjie)'

    set_seed()

    ###创建模型
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = ACRLSD()

    ##单卡
    # model = model.to(device)

    ##多卡训练
    gpus = [2]#选中显卡
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    Save_Name = 'ACRLSD_2D(fib25)_multigpu'

    ##装载数据
    # train_dataset_1 = Dataset_2D_hemi_Train(data_dir='./data/hemi/training/', split='train', crop_size=128,
    #                                         require_lsd=True, require_xz_yz=True)
    # val_dataset_1 = Dataset_2D_hemi_Train(data_dir='./data/hemi/training/', split='val', crop_size=128,
    #                                       require_lsd=True, require_xz_yz=True)

    fib25_data = '/home/liuhongyu2024/sshfs_share/liuhongyu2024/project/unispac/UniSPAC-edited/data/fib25'
    if os.path.exists(os.path.join(fib25_data, 'fib25_train.joblib')):
        print("Load data from disk...")
        train_dataset_2 = joblib.load(os.path.join(fib25_data, 'fib25_train.joblib'))
        val_dataset_2 = joblib.load(os.path.join(fib25_data, 'fib25_val.joblib'))
    else:
        train_dataset_2 = Dataset_2D_fib25_Train(data_dir=os.path.join(fib25_data, 'training'), split='train', crop_size=128,
                                                 require_lsd=True, require_xz_yz=True)
        joblib.dump(train_dataset_2, os.path.join(fib25_data, 'fib25_train.joblib'))
        val_dataset_2 = Dataset_2D_fib25_Train(data_dir=os.path.join(fib25_data, 'training'), split='val', crop_size=128,
                                               require_lsd=True, require_xz_yz=True)
        joblib.dump(val_dataset_2, os.path.join(fib25_data, 'fib25_val.joblib'))

    # train_dataset_3 = Dataset_2D_cremi_Train(data_dir='../data/CREMI/', split='train', crop_size=128, require_lsd=True)
    # val_dataset_3 = Dataset_2D_cremi_Train(data_dir='../data/CREMI/', split='val', crop_size=128, require_lsd=True)

    # train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    # val_dataset = ConcatDataset([val_dataset_1, val_dataset_2])
    train_dataset = train_dataset_2
    val_dataset = val_dataset_2

    # train_dataset = train_dataset_1
    # val_dataset = val_dataset_1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=48, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn_2D_hemi_Train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size // 2, shuffle=False, num_workers=48, pin_memory=True,
                            collate_fn=collate_fn_2D_hemi_Train)

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
    early_stop_count = 100
    no_improve_count = 0
    with tqdm(total=training_epochs) as pbar:
        analysis = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1'])
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            ###################Train###################
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            count, total = 0, len(train_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
            for raw, labels, point_map, mask, gt_affinity, gt_lsds in train_loader:
                loss_value, pred = model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation)
                count = count + 1
                progress_desc = f"Train {count}/{total}, "
                train_desc = f"Best: {epoch}, loss: {Best_val_loss:.2f}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 98
            np.random.seed(seed)
            random.seed(seed)
            count, total = 0, len(val_loader)
            acc_loss = []
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            for raw, labels, point_map, mask, gt_affinity, gt_lsds in val_loader:
                with torch.no_grad():
                    loss_value, _, results = model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation,
                                               train_step=False)
                count += 1
                acc_loss.append(loss_value)
                progress_desc = f"Val {count}/{total}, "
                val_desc = (f"acc: {results['accuracy']:.4f}, prec: {results['precision']:.4f}, "
                            f"recall: {results['recall']:.4f}, f1: {results['f1_score']:.4f}")
                analysis.loc[len(analysis)] = [results['accuracy'], results['precision'], results['recall'],
                                               results['f1_score']]

                pbar.set_description(progress_desc + train_desc + val_desc)

            val_loss = np.mean(np.array([loss_value.cpu().numpy() for loss_value in acc_loss]))

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.state_dict(), '/home/liuhongyu2024/Downloads/UniSPAC-edited/output/checkpoints/{}_Best_in_val.model'.format(Save_Name))
                no_improve_count = 0
            else:
                no_improve_count = no_improve_count + 1

            ##Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, Best_val_loss, Best_epoch))
            fh.flush()
            ch.flush()
            # writer.add_scalar('val_loss', val_loss, epoch)
            analysis.to_csv(csvfile)
            epoch += 1
            pbar.update(1)
            ##Early stop
            if no_improve_count == early_stop_count:
                logging.info("Early stop!")
                break
