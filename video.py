# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color   #scikit-image是基于scipy的一款图像处理包，它将图片作为numpy数组进行处理
import numpy as np
import random
import math
import matplotlib.pyplot as plt
 
from torchvision import transforms, utils
from PIL import Image
import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import glob
from model.u2net import U2NET,U2NETP #导入两个网络
import cv2
 
 
capture = cv2.VideoCapture(0)
class RescaleT(object): #此处等比缩放原始图像位指定的输出大
 
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size  #获得输出的图片的大小
 
    def __call__(self,sample):
        imidx, image, label,frame = sample['imidx'], sample['image'],sample['label'],sample['frame'] #获取到图片的索引、图片名和标签
 
        h, w = image.shape[:2] #获取图片的形状
 
        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size #根据输出图片的大小重新分配宽和高
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size
 
        new_h, new_w = int(new_h), int(new_w)
 
        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)
 
        img = transform.resize(image,(self.output_size,self.output_size),mode='constant') #此处等比缩放原始图像位指定的输出大小
        lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)  #等比缩放标签图像    #skimage.transform import resize
 
        return {'imidx':imidx, 'image':img,'label':lbl,"frame":frame}
 
class Rescale(object): #重新缩放至指定大小
 
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
 
    def __call__(self,sample): #使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
 
        imidx, image, label,frame = sample['imidx'], sample['image'],sample['label'],sample['frame']
 
        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]
 
        h, w = image.shape[:2]
 
        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size
 
        new_h, new_w = int(new_h), int(new_w)
 
        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        img = transform.resize(image,(new_h,new_w),mode='constant')
        lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)
 
        return {'imidx':imidx, 'image':img,'label':lbl,"frame":frame}
 
class RandomCrop(object): #返回经过随机裁剪后的图像和标签，指定输出大小
 
    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self,sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
 
        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]
 
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
 
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
 
        image = image[top: top + new_h, left: left + new_w] #对原始的图片进行随机裁
        label = label[top: top + new_h, left: left + new_w] #对原始标签随机裁剪
 
        return {'imidx':imidx,'image':image, 'label':label} #返回经过随机裁剪后的图像和标签
 
class ToTensor(object): #对图像和标签归一化
    """Convert ndarrays in sample to Tensors."""
 
    def __call__(self, sample):
 
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
 
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        tmpLbl = np.zeros(label.shape)
 
        image = image/np.max(image) #归一化图片
        if(np.max(label)<1e-6):
            label = label
        else:
            label = label/np.max(label)
 
        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
 
        tmpLbl[:,:,0] = label[:,:,0]
 
        # change the r,g,b to b,r,g from [0,255] to [0,1]
        #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))
 
        return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}
 
class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,flag=0):
        self.flag = flag
 
    def __call__(self, sample):
 
        imidx, image, label,frame =sample['imidx'], sample['image'], sample['label'], sample['frame']
 
        tmpLbl = np.zeros(label.shape)
 
        if(np.max(label)<1e-6):
            label = label
        else:
            label = label/np.max(label)
 
        # change the color space
        if self.flag == 2: # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0],image.shape[1],6))
            tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
            if image.shape[2]==1:
                tmpImgt[:,:,0] = image[:,:,0]
                tmpImgt[:,:,1] = image[:,:,0]
                tmpImgt[:,:,2] = image[:,:,0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)
 
            # nomalize image to range [0,1]
            tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
            tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
            tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
            tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
            tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
            tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))
 
            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))
 
            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
            tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
            tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
            tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])
 
        elif self.flag == 1: #with Lab color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
 
            if image.shape[2]==1:
                tmpImg[:,:,0] = image[:,:,0]
                tmpImg[:,:,1] = image[:,:,0]
                tmpImg[:,:,2] = image[:,:,0]
            else:
                tmpImg = image
 
            tmpImg = color.rgb2lab(tmpImg)
 
            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))
 
            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))
 
            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
 
        else: # with rgb color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
            image = image/np.max(image)
            if image.shape[2]==1:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
            else:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
 
        tmpLbl[:,:,0] = label[:,:,0]
 
        # change the r,g,b to b,r,g from [0,255] to [0,1]
        #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))
 
        return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl),"frame":frame}
 
class SalObjDataset(Dataset): #返回归一化后的图片索引，图片，标签图片
    def __init__(self,img_name_list,lbl_name_list,transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list #获取到所有的图片名绝对路径
        self.label_name_list = lbl_name_list #获取到所有的标签绝对路径
        self.transform = transform                                                               #transform包括裁剪缩放转tensor
 
    def __len__(self):
        return len(self.image_name_list)
 
    def __getitem__(self,idx):
        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])
        # image = io.imread(self.image_name_list[idx]) #通过每张的绝对路径读取到每一张图片
        # print(type(image))  #<class 'numpy.ndarray'>
 
 
        while True:
            ref, frame = capture.read()  # 读取某一帧
            image = frame
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 格式转变，BGRtoRGB
 
            print(image.shape)          #(480, 640, 3)
            print("=======================================================")
            # imname = self.image_name_list[idx]
            imidx = np.array([idx]) #图片的索引转化numpy的数组
 
            if(0==len(self.label_name_list)): #如果没有标签则创建一个0标签
                label_3 = np.zeros(image.shape)
            else: #如果有标签则获取对应的标签
                label_3 = io.imread(self.label_name_list[idx])
 
            label = np.zeros(label_3.shape[0:2]) #将标签数据用不同维度的0表示
 
            if(3==len(label_3.shape)):
                label = label_3[:,:,0]
            elif(2==len(label_3.shape)):
                label = label_3
 
            if(3==len(image.shape) and 2==len(label.shape)):
                label = label[:,:,np.newaxis] #np.newaxis的作用就是在这一位置增加一个一维，这一位置指的是np.newaxis所在的位置
            elif(2==len(image.shape) and 2==len(label.shape)):
                image = image[:,:,np.newaxis]
                label = label[:,:,np.newaxis]
 
            sample = {'imidx':imidx, 'image':image, 'label':label,"frame":frame}
 
            if self.transform:
                sample = self.transform(sample)  #对图像transform
            return sample
 
def main():
 
    model_name = 'u2net'#u2netp #保存的模型的名称
    model_dir = r"/home/sherlock/下载/图像分类网络/deep-learning-for-image-processing-master/pytorch_segmentation/multi-u2net/multi_U2NET/saved_models/u2net/u2net_bce_itr_44800.pth" #模型参数的路径
    img_name_list = [i for i in range(10000)]
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
 
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)     #加载数据
 
 
    if(model_name=='u2net'): #分辨使用的是哪一个模型参数
        print("...load U2NET---173.6 MB")
        net = U2NET(3,5)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,5)
    net.load_state_dict(torch.load(model_dir)) #加载训练好的模型
    if torch.cuda.is_available():
        net.cuda()                             #网络转移至GPU
    net.eval()                                 #测评模式
 
    for i_test, data_test in enumerate(test_salobj_dataloader):
 
        inputs_test = data_test['image']                   #测试的是图片
        inputs_test = inputs_test.type(torch.FloatTensor)  #转为浮点型
 
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
            #Variable是对Tensor的一个封装，操作和Tensor是一样的，但是每个Variable都有三个属性，
            # tensor不能反向传播，variable可以反向传播。它会逐渐地生成计算图。
            # 这个图就是将所有的计算节点都连接起来，最后进行误差反向传递的时候，
            # 一次性将所有Variable里面的梯度都计算出来，而tensor就没有这个能力
        else:
            inputs_test = Variable(inputs_test)
 
        d1,d2,d3,d4,d5,d6,d7 = net(inputs_test) #将图片传入网络
 
        pred = d1[:,0,:,:]
        pred = (pred-torch.min(pred))/(torch.max(pred)-torch.min(pred))  #对预测的结果做归一化
 
 
        predict = pred.squeeze()  # 删除单维度
        predict_np = predict.cpu().data.numpy()  # 转移到CPU上
        im = Image.fromarray(predict_np * 255).convert('RGB')  # 转为PIL，从归一化的图片恢复到正常0到255之间
 
        imo = im.resize((640, 480), resample=Image.BILINEAR)  # 得到的掩码！！！！！！！！
 
        # img_array = np.asarray(Image.fromarray(np.uint8(data_test['image'])))
 
        img_array = np.uint8(data_test["frame"][0])
        # print(data_test)
        # print(data_test["frame"][0])
        # print(data_test["frame"][0].shape)
        # cv2.imshow("", np.uint8(data_test["frame"][0]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
 
 
        mask = np.asarray(Image.fromarray(np.uint8(imo)))
 
        # cv2.imshow("", np.uint8(mask))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(img_array.shape)
        # print("ccccccccccccccccccc")
        # res = np.concatenate((img_array, mask[:, :, [0]]), -1)  # 将原图和掩码进行数组拼接
        # img = cv2.cvtColor(res, cv2.COLOR_RGB2BGRA)
        # img = Image.fromarray(img.astype('uint8'), mode='RGBA')
        # img.show()
        # b, g, r, a = cv2.split(img)
        # img = cv2.merge([a,r,b,g,])
 
        img = Image.fromarray(np.uint8(img_array * (mask / 255)))
 
        cv2.imshow("",np.uint8(img))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
 
        del d1,d2,d3,d4,d5,d6,d7 #del 用于删除对象。在 Python，一切都是对象，因此 del 关键字可用于删除变量、列表或列表片段等。
 
 
if __name__ == "__main__":
    main() #调用
