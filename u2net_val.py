import os
import cv2 as cv
from skimage import io
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from model.u2net_FPN import U2NETP_FPN
from utils_metrics import compute_mIoU, show_results
from data_loader import RescaleT, ToTensorLab

def label_deal(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])
    if len(label_3.shape) == 3:
        label = label_3[:, :, 0]
    elif len(label_3.shape) == 2:
        label = label_3
    if len(image.shape) == 3 and len(label.shape) == 2:
        label = label[:, :, np.newaxis]
    elif len(image.shape) == 2 and len(label.shape) == 2:
        label = label[:, :, np.newaxis]
    return label

def eval_print_miou(num_classes, name_classes, images_path, gt_dir, pred_dir, predict_label, miou_out_path, model_dir):
    miou_mode = 0
    val_list_path = "/home/sherlock/下载/图像分类网络/deep-learning-for-image-processing-master/pytorch_segmentation/multi-u2net/multi_U2NET/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
    if not os.path.exists(val_list_path):
        raise FileNotFoundError(f"Validation list file not found: {val_list_path}")
    
    with open(val_list_path, 'r') as f:
        val_img_names = [line.strip() for line in f.readlines()]
    
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model ", model_dir)
        net = U2NETP_FPN(3, num_classes)
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        model = net.eval()
        print("Load model done.")

        print("Get predict result.")
        for image_name in tqdm(val_img_names):
            # 尝试加载 .jpg 文件
            image_path = os.path.join(images_path, image_name + '.jpg')
            if not os.path.exists(image_path):
                # 如果 .jpg 文件不存在，尝试加载 .png 文件
                image_path = os.path.join(images_path, image_name + '.png')
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
            
            image = io.imread(image_path)
            
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]  # 去除Alpha通道
            
            label = label_deal(image)
            if len(image.shape) == 2 and len(label.shape) == 2:
                image = image[:, :, np.newaxis]
            imidx = np.array([0])

            sample = {'imidx': imidx, 'image': image, 'label': label}
            sample = RescaleT(512)(sample)
            sample = ToTensorLab(flag=0)(sample)

            inputs_test = sample['image'].unsqueeze(0).type(torch.FloatTensor)
            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)
            d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)
            d1 = torch.softmax(d1, dim=1)
            predict_np = torch.argmax(d1, dim=1, keepdim=True).cpu().detach().numpy().squeeze()

            predict_np = cv.resize(predict_np, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)
            cv.imwrite(os.path.join(pred_dir, image_name + '.png'), predict_np)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, val_img_names, num_classes, name_classes)
        print("Get miou done.")
        print(IoUs, num_classes)
        # show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
        return np.nanmean(IoUs), np.nanmean(PA_Recall), np.nanmean(Precision), np.nanmean(IoUs)

    return 0.0, 0.0, 0.0, 0.0

if __name__ == "__main__":
    num_classes = 21
    name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    images_path = "datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/"
    gt_dir = "datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/"
    pred_dir = "datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/predict_masks/"
    predict_label = "datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/predict_labels/"
    miou_out_path = "miou_out"
    model_dir = './saved_models/u2net/u2netp.pth'
    eval_print_miou(num_classes, name_classes, images_path, gt_dir, pred_dir, predict_label, miou_out_path, model_dir)
