from collections import OrderedDict
import torch

# 加载权重文件
file_name = 'ACRLSD_2D(ninanjie)_multigpu-no-xz-yz_no-crop_Best_in_val0.model'
state_dict = torch.load(file_name)

# 去掉 'module.' 前缀
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # 去掉 'module.' 前缀
    new_state_dict[name] = v

torch.save(new_state_dict, file_name)



# 将修改后的权重加载到模型中
