import cv2
import numpy as np

# 讀取兩張圖像
image_path1 = '/home/aicup/AICUP_2024/BASNet/TRA_RI_2002157_2.png'  # 替換為第一張圖像的路徑
image_path2 = '/home/aicup/AICUP_2024/BASNet/dataset/train/label_img/TRA_RI_2002157.png'  # 替換為第二張圖像的路徑

image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
_, image1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

# 檢查圖像是否成功加載
if image1 is None or image2 is None:
    print("Error: Could not load one of the images.")
    exit()

# 確保兩張圖像具有相同的大小
if image1.shape != image2.shape:
    print("Error: Images do not have the same dimensions.")
    exit()

# 計算兩張圖像的差異
difference = image1.astype(int) - image2.astype(int)

# 創建結果圖像，初始化為白色
result = np.ones((*difference.shape, 3), dtype=np.uint8) * 255

# 將 > 0 的部分設為藍色
result[difference > 0] = [255, 0, 0]

# 將 < 0 的部分設為紅色
result[difference < 0] = [0, 0, 255]

# 將 = 0 的部分保持為白色（可以省略，因為初始化為白色）

# 保存結果圖像
output_path = '/home/aicup/AICUP_2024/BASNet/TRA_RI_2002157_diff.png'  # 替換為你想保存結果圖像的路徑
cv2.imwrite(output_path, result)
