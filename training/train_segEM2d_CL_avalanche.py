import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,ConcatDataset
from tqdm.auto import tqdm
import logging
# import argparse
# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d

from utils.dataloader_hemi_better import Dataset_2D_hemi_Train,collate_fn_2D_hemi_Train
from utils.dataloader_fib25_better import Dataset_2D_fib25_Train,collate_fn_2D_fib25_Train
from utils.dataloader_zebrafinch_better import Dataset_2D_zebrafinch_Train_CL_avalanche


from avalanche.benchmarks import nc_benchmark
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Cumulative, AGEM, GEM, Replay, Naive, EWC, LwF, DER, GDumb, SynapticIntelligence
from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin


## CUDA_VISIBLE_DEVICES=0 python main_segEM_2d_train_zebrafinch.py &

WEIGHT_LOSS3 = 1

def set_seed(seed = 1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True
    
    
import argparse
def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--strategy', dest='strategy', type=str, default='Cumulative',
                        help='strategy',metavar='E')

    return parser.parse_args()

    
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
            constant_upsample=True).to(device)
        
        self.lsd_predict = torch.nn.Conv2d(in_channels=12,out_channels=6, kernel_size=1)  #最终输出层的卷积操作


        # create our network, 6 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet2d(
            in_channels=7, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2],[2,2],[2,2]], #降采样的因子
            padding='same',
            constant_upsample=True).to(device)
        
        self.affinity_predict = torch.nn.Conv2d(in_channels=12,out_channels=2, kernel_size=1)  #最终输出层的卷积操作
    
    def forward(self, x):

        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x,y_lsds.detach()],dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds,y_affinity
    
    
####ACRLSD模型
class segEM2d(torch.nn.Module):
    def __init__(
        self,
    ):
        super(segEM2d, self).__init__()
        
        ##For affinity prediction
        self.model_affinity = ACRLSD()
        # model_path = './output/checkpoints/ACRLSD_2D(hemi+fib25)_Best_in_val.model' 
        # weights = torch.load(model_path,map_location=torch.device('cpu'))
        # self.model_affinity.load_state_dict(weights)
        # for param in self.model_affinity.parameters():
        #     param.requires_grad = False

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
        
        y_concat = torch.cat([x_prompt.unsqueeze(1),y_affinity.detach()],dim=1)

        y_mask = self.mask_predict(self.model_mask(y_concat))
        y_mask = self.sigmoid(y_mask)

        return y_mask,y_lsds,y_affinity



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

def model_step(model, optimizer, input_image, input_prompt, gt_binary_mask, gt_affinity, gt_lsds, activation, train_step=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()
        
    # forward
    # lsd_logits,affinity_logits = model(raw)
    y_mask,y_lsds,y_affinity= model(input_image,input_prompt)

    loss1 = F.binary_cross_entropy(y_mask.squeeze(),gt_binary_mask.squeeze())
    Diceloss_fn = DiceLoss().to(device)
    loss2 = Diceloss_fn(1-y_mask.squeeze(), 1-gt_binary_mask.squeeze())
    
    loss3 = torch.sum(y_mask * gt_affinity)/torch.sum(gt_affinity)
    
    loss_mask = loss1 + loss2 + loss3 * WEIGHT_LOSS3
    
    
    loss_fn = torch.nn.MSELoss().to(device)
    loss_affinity = loss_fn(y_lsds, gt_lsds) + loss_fn(y_affinity, gt_affinity)
    

    loss = loss_mask + loss_affinity
    
    # backward if training mode
    if train_step:
        loss.backward()
        optimizer.step()
    
    return loss, y_mask



if __name__ == '__main__':
    
    args = get_args()
    
    ##设置超参数
    training_epochs = 100
    learning_rate = 1e-4
    batch_size = 32

    Save_Name = 'segEM2d(star)(hemi+fib25)_zebrafinch1-6_{}'.format(args.strategy)
    
    set_seed()
    

    ###创建模型
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = segEM2d()
    # model_path = './output/checkpoints/segEM2d(hemi+fib25)_Best_in_val.model' 
    model_path = './output/checkpoints/segEM2d(hemi+fib25)wloss-1_Best_in_val.model' 
    weights = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    
    ##多卡训练
    #一机多卡设置
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'#设置所有可以使用的显卡，共计四块
    # device_ids = [0,1,2,3]#选中显卡
    # model = nn.DataParallel(model, device_ids=device_ids)#并行使用两块
    # model = torch.nn.DataParallel(model)  # 默认使用所有的device_ids

    model = model.to(device)


    # ##装载数据
    # train_dataset_1 = Dataset_2D_hemi_Train(data_dir='./data/funke/hemi/training/', split='train', crop_size=128, require_lsd=True,require_xz_yz=True)
    # val_dataset_1 = Dataset_2D_hemi_Train(data_dir='./data/funke/hemi/training/', split='val', crop_size=128, require_lsd=True,require_xz_yz=True)
    
    # train_dataset_2 = Dataset_2D_fib25_Train(data_dir='./data/funke/fib25/training/', split='train', crop_size=128, require_lsd=True,require_xz_yz=True)
    # val_dataset_2 = Dataset_2D_fib25_Train(data_dir='./data/funke/fib25/training/', split='val', crop_size=128, require_lsd=True,require_xz_yz=True)
    

    train_dataset_3 = Dataset_2D_zebrafinch_Train_CL_avalanche(data_dir='./data/funke/zebrafinch/training/', split='train', crop_size=128, require_lsd=True, data_idxs=list(range(6)))
    val_dataset_3 = Dataset_2D_zebrafinch_Train_CL_avalanche(data_dir='./data/funke/zebrafinch/training/', split='val', crop_size=128, require_lsd=True,data_idxs=list(range(6)))
    
        
    # train_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])
    # val_dataset   = ConcatDataset([val_dataset_1, val_dataset_2, val_dataset_3])
    
    train_dataset = train_dataset_3
    val_dataset = val_dataset_3

    # 创建基准
    benchmark = nc_benchmark(
        train_dataset = train_dataset,
        test_dataset = val_dataset,
        n_experiences=6, # ignored if one_dataset_per_exp==True
        task_labels=True,
        fixed_class_order=[0,1,2,3,4,5], ##按照dataloader中设置的顺序依次学习每个volumes
    )
    
    # train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=14,pin_memory=True,drop_last=True,collate_fn=collate_fn_2D_hemi_Train)
    # val_loader = DataLoader(val_dataset,batch_size=8, shuffle=False, num_workers=14,pin_memory=True, collate_fn=collate_fn_2D_hemi_Train)
    val_loader = DataLoader(val_dataset,batch_size=8, shuffle=False, num_workers=14,pin_memory=True)

    # 设置评估插件
    evaluator = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger(), TensorboardLogger()]
    )


    ##创建log日志
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = './output/log/log_{}.txt'.format(Save_Name)
    fh = logging.FileHandler(logfile,mode='a')
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

    if args.strategy == 'Cumulative':
        # 创建Cumulative策略实例
        strategy = Cumulative(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(), # ignore, no use!!!
            train_mb_size=batch_size,
            train_epochs=training_epochs,
            eval_mb_size=8,
            device=device,
            plugins=None,
            evaluator=evaluator,
            eval_every=1
        )
    elif args.strategy == 'AGEM':
        strategy = AGEM(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(), # ignore, no use!!!
            patterns_per_exp=200,  # 设置每个经验存储的样本数量
            sample_size=64,        # 从记忆中采样的样本数量
            train_mb_size=batch_size,
            train_epochs=training_epochs,
            eval_mb_size=8,
            evaluator=evaluator,
            device=device
        )
    elif args.strategy == 'GEM':
        strategy = GEM(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(), # ignore, no use!!!
            patterns_per_exp=200,  # 设置每个经验存储的样本数量
            train_mb_size=batch_size,
            train_epochs=training_epochs,
            eval_mb_size=8,
            evaluator=evaluator,
            device=device
        )
    elif args.strategy == 'Replay':
        # 创建Replay策略
        strategy = Replay(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(), # ignore, no use!!!
            mem_size=200,  # 设置Replay缓冲区大小
            train_mb_size=batch_size,
            train_epochs=training_epochs,
            eval_mb_size=8,
            evaluator=evaluator,
            device=device
        )
    elif args.strategy == 'Naive':
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(), # ignore, no use!!!
            train_mb_size=batch_size,
            train_epochs=training_epochs,
            eval_mb_size=8,
            evaluator=evaluator,
            device=device
        )
    elif args.strategy == 'EWC':
        strategy = EWC(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(), # ignore, no use!!!
            ewc_lambda=0.4,  # EWC正则化强度
            train_mb_size=batch_size,
            train_epochs=training_epochs,
            eval_mb_size=8,
            evaluator=evaluator,
            device=device
        )
    elif args.strategy == 'LwF':
        strategy = LwF(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(), # ignore, no use!!!
            alpha = 1,
            temperature = 2,
            train_mb_size=batch_size,
            train_epochs=training_epochs,
            eval_mb_size=8,
            evaluator=evaluator,
            device=device
        )
    elif args.strategy == 'DER':
        strategy = DER(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(), # ignore, no use!!!
            mem_size=200,  # 设置内存大小
            alpha=0.1,  # 知识蒸馏损失权重
            beta=0.5,  # 分类损失权重
            train_mb_size=batch_size,
            train_epochs=training_epochs,
            eval_mb_size=8,
            evaluator=evaluator,
            device=device
        )
    elif args.strategy == 'GDumb':
        strategy = GDumb(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(), # ignore, no use!!!
            mem_size=1000,
            train_mb_size=batch_size,
            train_epochs=training_epochs,
            eval_mb_size=8,
            evaluator=evaluator,
            device=device
        )
    elif args.strategy == 'SynapticIntelligence':
        strategy = SynapticIntelligence(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(), # ignore, no use!!!
            si_lambda=0.1,  # SI regularization strength
            train_mb_size=batch_size,
            train_epochs=training_epochs,
            eval_mb_size=8,
            evaluator=evaluator,
            device=device
        )
        

    
    
    # 训练和评估循环
    for experience in benchmark.train_stream:
        logging.info(f"Starting training on experience {experience.current_experience}")
        strategy.train(experience)
        
        logging.info("Starting evaluation")
        strategy.eval(benchmark.test_stream)
        
        
        
        ###################Validate###################
        strategy.model.eval()
        ##Fix validation set
        seed = 98
        np.random.seed(seed)
        random.seed(seed)
        tmp_val_loader = iter(val_loader)
        acc_loss = []
        for raw,_,labels,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            with torch.no_grad():
                raw = raw.to(device)
                point_map = point_map.to(device)
                mask = mask.to(device)
                gt_affinity = gt_affinity.to(device)
                gt_lsds = gt_lsds.to(device)
                loss_value, _ = model_step(strategy.model, optimizer, raw, point_map, mask, gt_affinity, gt_lsds, activation, train_step=False)
            acc_loss.append(loss_value.cpu().detach().numpy())
        val_loss = np.mean(acc_loss)
        
        ##Record
        logging.info("experience {}: val_loss = {:.6f}".format(
            experience.current_experience,val_loss))
    
        
        # torch.save(model.state_dict(),'./output/checkpoints/{}-{}_Best_in_val.model'.format(Save_Name,count))
        # save_checkpoint(strategy, './output/checkpoints/{}-{}_Best_in_val.model'.format(Save_Name,count))
        torch.save(strategy.model.state_dict(),'./output/checkpoints/{}-{}_Best_in_val.model'.format(Save_Name,experience.current_experience))



