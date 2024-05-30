import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def flip_images(input_dir):

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 讀取圖像
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # 水平翻轉圖像
            flipped_img = cv2.flip(img, 1)

            # 構造新文件名
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_1{ext}"
            output_path = os.path.join(input_dir, new_filename)

            # 保存翻轉後的圖像
            cv2.imwrite(output_path, flipped_img)
            print(f'{new_filename} is saved')
            
            # 垂直翻轉圖像
            flipped_img = cv2.flip(img, 0)

            new_filename = f"{name}_2{ext}"
            output_path = os.path.join(input_dir, new_filename)

            # 保存翻轉後的圖像
            cv2.imwrite(output_path, flipped_img)
            print(f'{new_filename} is saved')

            # 垂直+水平翻轉圖像
            tmp_img = cv2.flip(img, 0)
            flipped_img = cv2.flip(tmp_img, 1)

            new_filename = f"{name}_3{ext}"
            output_path = os.path.join(input_dir, new_filename)

            # 保存翻轉後的圖像
            cv2.imwrite(output_path, flipped_img)
            print(f'{new_filename} is saved')

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 讀取圖像
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            upper_half = img[:height//2, :, :]
            lower_half = img[height//2:, :, :]

            name, ext = os.path.splitext(filename)
            
            upper_half_path = os.path.join(input_dir, f"{name}_4{ext}")
            lower_half_path = os.path.join(input_dir, f"{name}_5{ext}")

            cv2.imwrite(upper_half_path, upper_half)
            cv2.imwrite(lower_half_path, lower_half)

import numpy as np

pp = ["img", "label_img", "new_label_img"]
for p in pp:
    input_dir = f'/home/aicup/AICUP_2024/BASNet/dataset/train/{p}'
    flip_images(input_dir)
def extract_and_save_edges(image_path):
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # 將圖片轉換為灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 定義結構元素
    kernel = np.ones((5, 5), np.uint8)

    # 進行腐蝕操作
    # erosion = cv2.erode(binary, kernel, iterations=1)

    # 进行腐蚀操作，不进行padding
    erosion = cv2.erode(gray, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)


    # 提取邊框（原始圖像 - 腐蝕後的圖像）
    edges = cv2.absdiff(gray, erosion)
    name = image_path.split("/")[-1]
    output = f'/home/aicup/AICUP_2024/BASNet/dataset/train/label_img/{name}'

    # 儲存結果圖片
    cv2.imwrite(output, edges)

    print(f"Edges extracted and saved to {output}")


import glob
a_files = glob.glob('/home/aicup/AICUP_2024/BASNet/dataset/train/new_label_img/*.png')
files = a_files
for file in files:
    extract_and_save_edges(file)