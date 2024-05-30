import cv2
import numpy as np
import glob
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

    # 儲存結果圖片
    cv2.imwrite(image_path, edges)

    print(f"Edges extracted and saved to {image_path}")

# 使用範例
input_image_path = glob.glob(f"test/submit_F1.5_tune/*")

for i in input_image_path:
    extract_and_save_edges(i)
