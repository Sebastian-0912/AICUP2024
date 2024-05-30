import glob
import numpy as np
import cv2
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# data_dir = 'dataset/train'
# tra_image_dir = 'img'
# tra_img_name_list = glob.glob(f"{data_dir}/{tra_image_dir}/*")
# images = []
# for path in tra_img_name_list:
#     image = cv2.imread(path)
#     image = cv2.resize(image, (240, 428))
#     # print(image.shape)
#     images.append(image)
# mean = np.mean(images, axis=(0, 1, 2))
# std = np.std(images, axis=(0, 1, 2))
mean = np.array([112.7168997, 136.27366714, 132.31856602])/255
std = np.array([52.96647579, 44.90719295, 50.38616624])/255
print(mean)
print(std)