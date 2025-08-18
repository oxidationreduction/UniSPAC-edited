import logging
import os
import random

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from utils.dataloader_ninanjie import Dataset_2D_ninanjie_Train
from utils.dataloader_hemi_better import Dataset_2D_hemi_Train, collate_fn_2D_hemi_Train


## CUDA_VISIBLE_DEVICES=0 python main_segEM_2d_train_zebrafinch.py &

# WEIGHT_LOSS_AFFINITY = 100

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


####ACRLSD模型
class segEM2d(torch.nn.Module):
    def __init__(
            self,
    ):
        super(segEM2d, self).__init__()

        ##For affinity prediction
        self.model_affinity = ACRLSD()
        # model_path = './output/checkpoints/ACRLSD_2D(hemi+fib25+cremi)_Best_in_val.model' 
        model_path = ('/home/liuhongyu2024/Documents/UniSPAC-edited/'
                      'output/checkpoints/ACRLSD_2D(ninanjie)_Best_in_val.model')
        weights = torch.load(model_path, map_location=torch.device('cpu'))
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


def model_step(model, optimizer, input_image, input_prompt, gt_binary_mask, gt_affinity, activation, train_step=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    # lsd_logits,affinity_logits = model(raw)
    y_mask, _, _ = model(input_image, input_prompt)

    loss1 = F.binary_cross_entropy(y_mask.squeeze(), gt_binary_mask.squeeze())
    Diceloss_fn = DiceLoss().to(device)
    loss2 = Diceloss_fn(1 - y_mask.squeeze(), 1 - gt_binary_mask.squeeze())

    # loss3 = torch.sum(y_pred * gt_affinity)/torch.sum(gt_affinity)

    # loss = loss1 + loss2 + loss3 * WEIGHT_LOSS_AFFINITY
    loss = loss1 + loss2

    # backward if training mode
    if train_step:
        loss.backward()
        optimizer.step()

    return loss, y_mask


def load_dataset(dataset_name: str, split: str='train', from_temp=True):
    if os.path.exists(os.path.join(ninanjie_save, dataset_name)) and from_temp:
        print(f"Load {dataset_name} from disk...")
        train_dataset_2 = joblib.load(os.path.join(ninanjie_save, dataset_name))
    else:
        train_dataset_2 = Dataset_2D_ninanjie_Train(data_dir=os.path.join(ninanjie_data), split=split, crop_size=256,
                                                 require_lsd=True, require_xz_yz=True)
        joblib.dump(train_dataset_2, os.path.join(ninanjie_save, dataset_name))
    return train_dataset_2


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 96
    Save_Name = 'segEM2d(ninanjie)'

    set_seed()

    ###创建模型
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = segEM2d()
    ##多卡训练
    # 一机多卡设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,4'#设置所有可以使用的显卡，共计四块
    device_ids = [0,1,4]#选中显卡
    torch.cuda.set_device(device_ids[0])
    model = nn.DataParallel(model.cuda(), device_ids=device_ids, output_device=device)
    # model = torch.nn.DataParallel(model)  # 默认使用所有的device_ids

    # model = model.to(device)

    ##装载数据
    ninanjie_data = '/home/liuhongyu2024/sshfs_share/liuhongyu2024/project/unispac/UniSPAC-edited/data/ninanjie/fourth'
    ninanjie_save = '/home/liuhongyu2024/sshfs_share/liuhongyu2024/project/unispac/UniSPAC-edited/data/ninanjie-save'

    train_dataset, val_dataset = load_dataset('first', require_xz_yz=True, from_temp=True,
                                              crop_xyz=[3, 3, 2], chunk_position=[1, 1, 0])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=14, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn_2D_hemi_Train)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=14, pin_memory=True,
                            collate_fn=collate_fn_2D_hemi_Train)

    ##创建log日志
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = './output/log/log_{}.txt'.format(Save_Name)
    fh = logging.FileHandler(logfile, mode='a')
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

    # training loop
    model.train()
    epoch = 0
    Best_val_loss = 100000
    Best_epoch = 0
    early_stop_count = 100
    no_improve_count = 0
    with tqdm(total=training_epochs) as pbar:
        while epoch < training_epochs:
            ###################Train###################
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            tmp_loader = iter(train_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
            for raw, labels, point_map, mask, gt_affinity, _ in tmp_loader:
                ##Get Tensor
                raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
                point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)  # (batch, 6, height, width)
                mask = torch.as_tensor(mask, dtype=torch.float, device=device)  # (batch, 2, height, width)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                                              device=device)  # (batch, 2, height, width)

                loss_value, pred = model_step(model, optimizer, raw, point_map, mask, gt_affinity, activation,
                                              train_step=True)
            epoch += 1
            pbar.update(1)

            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 98
            np.random.seed(seed)
            random.seed(seed)
            tmp_val_loader = iter(val_loader)
            acc_loss = []
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            for raw, labels, point_map, mask, gt_affinity, _ in tmp_val_loader:
                raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
                point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)  # (batch, 6, height, width)
                mask = torch.as_tensor(mask, dtype=torch.float, device=device)  # (batch, 2, height, width)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                                              device=device)  # (batch, 2, height, width)

                with torch.no_grad():
                    loss_value, _ = model_step(model, optimizer, raw, point_map, mask, gt_affinity, activation,
                                               train_step=False)
                acc_loss.append(loss_value.cpu().detach().numpy())
            val_loss = np.mean(acc_loss)

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.state_dict(), './output/checkpoints/{}_Best_in_val.model'.format(Save_Name))
                no_improve_count = 0
            else:
                no_improve_count = no_improve_count + 1

            ##Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, Best_val_loss, Best_epoch))
            # writer.add_scalar('val_loss', val_loss, epoch)

            ##Early stop
            if no_improve_count == early_stop_count:
                logging.info("Early stop!")
                break
