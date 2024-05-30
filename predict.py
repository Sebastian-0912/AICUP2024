# -*- coding: utf-8 -*-
import os

# 设置CUDA_VISIBLE_DEVICES环境变量为0，表示只使用第一个GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/Shareddrives/AICUP/BASNet

from data_loader import RescaleT

from data_loader import ToTensor
from data_loader import SalObjDataset

from model import BASNet
from skimage import io
import torch

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
# ------- 2. set the directory of training dataset --------




def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')

# --------- 1. get image path and name ---------

data_dir = f'dataset/test'
prediction_dir = f'./test/submit_F1.5_tune/'

#### River

model_dir = './saved_model/train_4river_F1.5_tune.pth'
image_ext = '.jpg'

pri_name_list = glob.glob(f"{data_dir}/PRI_RI_*{image_ext}")
pub_name_list = glob.glob(f"{data_dir}/PUB_RI_*{image_ext}")
img_name_list = pri_name_list + pub_name_list
img_name_list = sorted(img_name_list, key=lambda name: int(name[-11:-4]))

# --------- 2. dataloader ---------
#1. dataload
test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [], bbx_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensor()]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)

# --------- 3. model define ---------
print("...load BASNet for river...")
net = BASNet(3,1)
net.load_state_dict(torch.load(model_dir))
if torch.cuda.is_available():
  net.cuda()
net.eval()

# --------- 4. inference for each image ---------
for i_test, data_test in enumerate(test_salobj_dataloader):

  print("inferencing:",img_name_list[i_test].split("/")[-1])

  inputs_test = data_test['image']
  inputs_test = inputs_test.type(torch.FloatTensor)

  if torch.cuda.is_available():
    inputs_test = Variable(inputs_test.cuda())
  else:
    inputs_test = Variable(inputs_test)

  d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)

  # normalization
  pred = d1[:,0,:,:]
  pred = normPRED(pred)

  # save results to test_results folder
  save_output(img_name_list[i_test],pred,prediction_dir)

  del d1,d2,d3,d4,d5,d6,d7,d8

#### Road

model_dir = './saved_model/train_4road_F1.5_tune.pth'
image_ext = '.jpg'

pri_name_list = glob.glob(f"{data_dir}/PRI_RO_*{image_ext}")
pub_name_list = glob.glob(f"{data_dir}/PUB_RO_*{image_ext}")
img_name_list = pri_name_list + pub_name_list
img_name_list = sorted(img_name_list, key=lambda name: int(name[-11:-4]))

# --------- 2. dataloader ---------
#1. dataload
test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [], bbx_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensor()]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)

# --------- 3. model define ---------
print("...load BASNet for road...")
net = BASNet(3,1)
net.load_state_dict(torch.load(model_dir))
if torch.cuda.is_available():
  net.cuda()
net.eval()

# --------- 4. inference for each image ---------
for i_test, data_test in enumerate(test_salobj_dataloader):

  print("inferencing:",img_name_list[i_test].split("/")[-1])

  inputs_test = data_test['image']
  inputs_test = inputs_test.type(torch.FloatTensor)

  if torch.cuda.is_available():
    inputs_test = Variable(inputs_test.cuda())
  else:
    inputs_test = Variable(inputs_test)

  d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)

  # normalization
  pred = d1[:,0,:,:]
  pred = normPRED(pred)

  # save results to test_results folder
  save_output(img_name_list[i_test],pred,prediction_dir)

  del d1,d2,d3,d4,d5,d6,d7,d8
