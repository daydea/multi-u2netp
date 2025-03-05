import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import glob
import os
from utils_metrics import show_results,compute_mIoU
from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

import random
import csv
import time
from model.u2net_FPN import U2NETP_FPN
from u2net_val import eval_print_miou


def save_metrics_to_csv(csv_path, epoch, loss, miou, mpa, precision, recall):
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # 如果文件为空，写入表头
            writer.writerow(["Epoch", "Train Loss", "mIoU", "mPA", "Precision", "Recall"])
        writer.writerow([epoch, loss, miou, mpa, precision, recall])
    print(f"Metrics saved to {csv_path}")

# ------- 0. set random seed --------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(1000)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# ------- 1. define loss function --------
# CrossEntropyLoss
device = torch.device("cuda:0")
# background 的权重是 1 ，其他类别的为  1.5，总共的数量和 U2NET4P(3, 3)  的第二个参数对应
weights = np.array([1, 2, 3, 2, 2], dtype=np.float32)
weights = torch.from_numpy(weights).to(device)
loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)


def muti_CrossEntropyLoss_loss(d0, d1, d2, d3, d4, d5, d6, labels_v):
    labels_v = labels_v.squeeze(1).long()
    loss0 = loss_CE(d0, labels_v)
    loss1 = loss_CE(d1, labels_v)
    loss2 = loss_CE(d2, labels_v)
    loss3 = loss_CE(d3, labels_v)
    loss4 = loss_CE(d4, labels_v)
    loss5 = loss_CE(d5, labels_v)
    loss6 = loss_CE(d6, labels_v)

    loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"
          % (loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(),
             loss4.data.item(), loss5.data.item(), loss6.data.item()))

    return loss0, loss


# ------- 2. set the directory of training process --------
model_name = 'u2netp_fpn'  # 'u2net'
model_dir = os.path.join('saved_models', model_name + os.sep)

# 图片的文件类型
image_ext = '.png'
label_ext = '.png'

# train阶段
# 数据集路径
data_dir = os.path.join("datasets/aug_NGnOK/train_data" + os.sep)
# 原始图片路径
tra_image_dir = os.path.join('images' + os.sep)
# 图片的标签路径
tra_label_dir = os.path.join('masks' + os.sep)

# val阶段
# 数据集类别
num_classes = 5
name_classes = ["background", "quejiao", "feibian", "liuhen","lasi"]
# 原始图片路径
images_path = "datasets/aug_NGnOK/test_data/images/"
# 图片的标签路径
gt_dir = "datasets/aug_NGnOK/test_data/masks/"
# 存放推理结果图片的路径
pred_dir = "datasets/aug_NGnOK/test_data/predict_masks/"
predict_label = "datasets/aug_NGnOK/test_data/predict_labels/"
# 存放 miou 计算结果的 图片
miou_out_path = "miou_out_tab"

'''
# 预训练权重
seg_pretrain_u2netp_path = 'saved_models/pretrain_model/segm2_u2net.pth'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
'''
epoch_num = 50
batch_size = 2
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(512),
        RandomCrop(488),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# ------- 3. define model --------
# define the net
if model_name == 'u2net':
    #          img channel, classes number
    net = U2NET(3, 5)
elif model_name == 'u2netp_fpn':
    net = U2NETP_FPN(3, 5)
    # net.load_state_dict(torch.load(seg_pretrain_u2netp_path, map_location=torch.device("cpu")))

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 4480

# Print model's state dict keys and shapes
print("Model's state dict keys and shapes:")
for k, v in net.state_dict().items():
    print(f"{k}: {v.shape}")

'''
# Load pre-trained model weights
pretrained_dict = torch.load(seg_pretrain_u2netp_path, map_location=device)
# Print pre-trained model's state dict keys and shapes
print("Pre-trained model's state dict keys and shapes:")
# Check if the shapes match
update_count = 0
total_count = 0
for k, v in pretrained_dict.items():
    total_count += 1
    if k in net.state_dict() and v.shape == net.state_dict()[k].shape:
        update_count += 1
        print(f"Matched: {k} with shape {v.shape}")
    else:
        print(f"Mismatched: {k} with shape {v.shape} in pre-trained model, but {net.state_dict()[k].shape} in current model")

print(f"Out of {total_count} layers in pre-trained model, {update_count} layers weights are recovered and matched in the current model.")
'''
# 定义 CSV 文件路径
csv_path = os.path.join(model_dir, f"{model_name}_metrics.csv")

# 确保保存 CSV 文件的目录存在
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

print(f"DataLoader length: {len(salobj_dataloader)}")
for epoch in range(0, epoch_num):
    net.train()
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        optimizer.zero_grad()
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_CrossEntropyLoss_loss(d0, d1, d2, d3, d4, d5, d6, labels_v)
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d.pth" % ite_num)
            model_path_save = model_dir + model_name + "_bce_itr_" + str(ite_num) + ".pth"
            eval_print_miou(num_classes, name_classes, images_path, gt_dir, pred_dir, predict_label, miou_out_path, model_path_save)
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()
            ite_num4val = 0

    # 在每个 epoch 结束时保存模型权重
    model_path_save = os.path.join(model_dir, f"{model_name}_epoch_{epoch + 1}.pth")
    torch.save(net.state_dict(), model_path_save)

    # 在每个 epoch 结束时计算验证指标
    if ite_num4val > 0:
        epoch_loss = running_loss / ite_num4val
        print(f"Epoch {epoch + 1}/{epoch_num} finished. Train Loss: {epoch_loss:.4f}")
    else:
        print(f"Epoch {epoch + 1}/{epoch_num} finished. No data processed. Skipping loss calculation.")
    epoch_loss = 0.0

    # 调用验证函数并获取指标
    results = eval_print_miou(
        num_classes=5,
        name_classes=["background", "quejiao", "feibian", "liuhen", "lasi"],
        images_path="datasets/aug_NGnOK/test_data/images/",
        gt_dir="datasets/aug_NGnOK/test_data/masks/",
        pred_dir="datasets/aug_NGnOK/test_data/predict_masks/",
        predict_label="datasets/aug_NGnOK/test_data/predict_labels/",
        miou_out_path="miou_out",
        model_dir=model_path_save
    )

    if results is not None:
        miou, mpa, precision, recall = results
        # 将指标保存到 CSV 文件
        save_metrics_to_csv(csv_path, epoch + 1, epoch_loss, miou, mpa, precision, recall)
    else:
        print("eval_print_miou returned None. Skipping metrics saving.")
    # 清理 CUDA 缓存
    torch.cuda.empty_cache()
    
# 在所有 epoch 训练结束后调用 show_results
if __name__ == "__main__":
    # 读取 CSV 文件中的指标数据
    df = pd.read_csv(csv_path)
    epochs = df['Epoch'].values
    miou = df['mIoU'].values
    mpa = df['mPA'].values
    precision = df['Precision'].values
    recall = df['Recall'].values

    # 计算所有 epoch 的平均指标
    avg_miou = np.nanmean(miou)
    avg_mpa = np.nanmean(mpa)
    avg_precision = np.nanmean(precision)
    avg_recall = np.nanmean(recall)

    print("Final Metrics:")
    print(f"Average mIoU: {avg_miou * 100:.2f}%")
    print(f"Average mPA: {avg_mpa * 100:.2f}%")
    print(f"Average Precision: {avg_precision * 100:.2f}%")
    print(f"Average Recall: {avg_recall * 100:.2f}%")

    # 调用 show_results 生成最终的柱状图和混淆矩阵
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, os.listdir(images_path), num_classes, name_classes)
    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)    
    
            
