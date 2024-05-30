import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

def fill_image(image_name, img, w, h):
    # Load the provided image
    img_path = f"train/label_img/{image_name}"
    img = Image.open(img_path)

    # Since the image is essentially a silhouette with a boundary to fill, we need to find the boundary coordinates.
    # Convert image to binary mode to easily identify the boundary
    binary_img = img.convert("1")

    # Since the desired region to fill is bounded by white, we should use white as the fill color.
    # Prepare a drawing context
    draw = ImageDraw.Draw(binary_img)

    # Assuming the polygon is centrally located and bounded by white lines, let's try filling from the center outward
    width, height = binary_img.size
    center_point = (w, h)

    # Using a flood fill operation starting from the center of the image
    ImageDraw.floodfill(binary_img, xy=center_point, value=255, border=None)

    # Save or display the result
    output_path = f"train/new_label_img/{image_name}"
    binary_img.save(output_path)
# binary_img.show()

images_dir = "train/label_img"
image_names = os.listdir( images_dir )

for image_name in image_names:
    image = cv2.imread( os.path.join( images_dir, image_name ), 0 )
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    heigh, width = image.shape[0], image.shape[1]
    new_image = np.zeros_like(image)
    jjj = 0
    for h in range( heigh ):
        for w in range( width ):
            """如果是白點"""
            if image[h][w] > 127:
                 new_image[h][w] = 255
            else:
                """ up """
                judge_up = 0
                for i in range(1, h+1):
                    if image[h-i][w] > 127:
                        judge_up = 1
                        break
                """ down """
                judge_down = 0
                for i in range(1, heigh-h):
                    if image[h+i][w] > 127:
                        judge_down = 1
                        break
                """ left """
                judge_left = 0
                for i in range(1, w+1):
                    if image[h][w-i] > 127:
                        judge_left = 1
                        break
                """ right """
                judge_right = 0
                for i in range(1, width-w):
                    if image[h][w+i] > 127:
                        judge_right = 1
                        break
                
                if judge_up and judge_down and judge_left and judge_right:
                    os.makedirs( "train/new_label_img", exist_ok=True )
                    fill_image(image_name, image, w, h)
                    jjj = 1
                    break
        if jjj:
            break

    
    # break
