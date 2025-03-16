from collections import OrderedDict
import torch

# 加载权重文件
state_dict = torch.load('ACRLSD_2D(fib25)_multigpu_Best_in_val.model')

# 去掉 'module.' 前缀
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # 去掉 'module.' 前缀
    new_state_dict[name] = v

torch.save(new_state_dict, 'ACRLSD_2D(fib25)_Best_in_val.model')



# 将修改后的权重加载到模型中
