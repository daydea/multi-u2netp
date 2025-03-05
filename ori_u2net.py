import os
import torch
from model.u2net import U2NET  # 确保这里导入的是 U2NET 而不是 U2NETP
from tools.saving_utils import save_checkpoint

# 创建保存目录
os.makedirs("saved_models/pretrain_model", exist_ok=True)

# 创建 U2NET 模型实例
u_net = U2NET(in_ch=3, out_ch=5)  # 根据你的需求设置输入和输出通道数

# 保存随机初始化的权重
save_checkpoint(u_net, os.path.join("saved_models/pretrain_model", "u2net_random.pth"))
