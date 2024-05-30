# -*- coding: utf-8 -*-
import os

# 设置CUDA_VISIBLE_DEVICES环境变量为0，表示只使用第一个GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms #, utils
import torch.optim as optim
import numpy as np
import glob
import matplotlib.pyplot as plt

from data_loader import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
from model import BASNet

import pytorch_ssim
import pytorch_iou
"""loss function"""

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

# ------- 1. define loss function --------
print("---define loss function...")
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
	loss7 = bce_ssim_loss(d7,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))
	# print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

	return loss0, loss

def f_score_loss(pred, target, beta_square=0.3):
    # if not isinstance(pred, torch.Tensor):
    #     pred = torch.tensor(pred)
    # if not isinstance(target, torch.Tensor):
    #     target = torch.tensor(target)
    # print(f"pred shape: {pred.shape}, target shape: {target.shape}")
    tp = torch.sum(pred * target).float()
    fp = torch.sum(pred * (1 - target)).float()
    fn = torch.sum((1 - pred) * target).float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_score = (1 + beta_square) * (precision * recall) / (beta_square* precision + recall + 1e-8)
    
    return 1 - f_score

def extract_edges(gray):
    gray = normPRED(gray)
    tensor_cpu = gray.cpu()
    gray = tensor_cpu.detach().numpy()
    
    # 将灰度图像转为8位整数类型
    gray = (gray * 255).astype(np.uint8)
    
    binary_image = gray
    # 二值化
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 定义结构元素
    kernel = np.ones((5, 5), np.uint8)
    
    # 进行腐蚀操作，不进行padding
    erosion = cv2.erode(binary_image, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    # 提取边框（原始图像 - 腐蚀后的图像）
    edges = cv2.absdiff(binary_image, erosion)
    
    # 将边框结果转换回PyTorch张量，并转到GPU（如果可用）
    edges_tensor = torch.tensor(edges / 255.0, dtype=torch.float32)
    if torch.cuda.is_available():
        edges_tensor = edges_tensor.cuda()
    
    return edges_tensor


print("---set the directory of training dataset...")
# ------- 2. set the directory of training dataset --------
data_dir = 'dataset/train'
tra_image_dir = 'img'
tra_bbox_dir = 'label_img'
tra_label_dir = 'new_label_img'

image_ext = '.jpg'
label_ext = '.png'
bbox_ext = '.png'

epoch_num = 15
batch_size_train = 45
model_dir = f"./saved_model/"
# batch_size_val = 1
# val_num = 0
tra_img_name_list = glob.glob(f"{data_dir}/{tra_image_dir}/TRA_RI_*{image_ext}")
# tra_img_name_list = sorted(tra_img_name_list, key=lambda name: int(name[-11:-4]))[:-5]
tra_img_name_list = sorted(tra_img_name_list, key=lambda name: int(''.join(filter(str.isdigit, name))))

tra_lbl_name_list = glob.glob(f"{data_dir}/{tra_label_dir}/TRA_RI_*{label_ext}")
# tra_lbl_name_list = sorted(tra_lbl_name_list, key=lambda name: int(name[-11:-4]))[:-5]
tra_lbl_name_list = sorted(tra_lbl_name_list, key=lambda name: int(''.join(filter(str.isdigit, name))))

tra_bbox_name_list = glob.glob(f"{data_dir}/{tra_bbox_dir}/TRA_RI_*{bbox_ext}")
# tra_bbox_name_list = sorted(tra_bbox_name_list, key=lambda name: int(name[-11:-4]))[:-5]
tra_bbox_name_list = sorted(tra_bbox_name_list, key=lambda name: int(''.join(filter(str.isdigit, name))))


print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    bbx_name_list=tra_bbox_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        # RandomCrop(224),
        # Normalize(),
        ToTensor()
    ]))

salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

# ------- 3. define model --------
# define the net
print("---define model...")
net = BASNet(3, 1)
if torch.cuda.is_available():
    net.cuda()
pretrained_model_path = 'saved_model/train_4river_last.pth'
net.load_state_dict(torch.load(pretrained_model_path))
# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = CosineAnnealingLR(optimizer, T_max=5)  # T_max是半個週期的迭代數
# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
loss_list = []
for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels, bboxes = data['image'], data['label'], data['bbox']
        

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        bboxes = bboxes.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v, bboxes_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False), Variable(bboxes.cuda(), requires_grad=False)
        else:
            inputs_v, labels_v, bboxes_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False), Variable(bboxes, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)

        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v)
        
        # normalization #copy from predicted.py
        pred = d0[:,0,:,:] #copy from predicted.py
        pred = normPRED(pred) #copy from predicted.py
        bbox_pred  = extract_edges(pred)
        
        fscore_loss_value = f_score_loss(bbox_pred , bboxes_v)
        total_loss = loss + fscore_loss_value*1.75

        total_loss.backward()
        optimizer.step()

        # # print statistics
        # running_loss += loss.data[0]
        # running_tar_loss += loss2.data[0]
        running_loss += total_loss.data.item()
        running_tar_loss += loss2.data.item()
        # del temporary outputs and loss
        p = total_loss.data.item()
        del d0, d1, d2, d3, d4, d5, d6, d7, loss2, loss

        #
        loss_list.append(running_loss / ite_num4val)
        if ite_num % 10 == 0:
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %.3f, tar: %.3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
    scheduler.step()
# tra_lbl_name_list = []
torch.save(net.state_dict(), model_dir + "train_4river_new.pth")
plt.plot(loss_list)
plt.title("train_4river_loss_curve")
plt.savefig('train_4river_loss_curve.png')
print('-------------Congratulations! Training Done!!!-------------')