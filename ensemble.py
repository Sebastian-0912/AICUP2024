import os
from PIL import Image
import numpy as np
import glob
import cv2

a = glob.glob("test/submit_new/*.png")
qq = ["submit_new", "submit_F1.75d", "submit_F1.75c", "submit_F2a", "submit_F2b", "submit_last", "submit_F2c_de", "submit_F1.5_tune"]
b = []
for i in a:
    b.append(i.split("/")[-1])
for p in b:
    images = []
    for q in qq:
        image = cv2.imread(f"test/{q}/{p}", cv2.IMREAD_GRAYSCALE)
        images.append(image)
    ensemble_image = np.mean(images, axis=0)
    cv2.imwrite(f'test/answer/{p}', ensemble_image)

    

# Function to read an image and convert it to a numpy array
# def read_image(image_path):
#     return np.array(Image.open(image_path))

# # Function to save a numpy array as an image
# def save_image(image_array, save_path):
#     image = Image.fromarray(np.uint8(image_array))
#     image.save(save_path)

# # Root directory containing model subdirectories
# root_dir = 'path_to_your_root_directory'

# # List of model directories
# model_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# # Assuming all model directories have the same number of images with the same names
# image_names = os.listdir(model_dirs[0])

# # Directory to save the ensemble images
# output_dir = os.path.join(root_dir, 'ensemble_output')
# os.makedirs(output_dir, exist_ok=True)

# # Process each image
# for image_name in image_names:
#     # Read all images with the same name from different model directories
#     images = [read_image(os.path.join(model_dir, image_name)) for model_dir in model_dirs]
    
#     # Ensure all images have the same shape
#     for img in images:
#         assert img.shape == images[0].shape, f"All images must have the same dimensions. Issue with {image_name}."

#     # Convert list of images to a numpy array for easy manipulation
#     images_array = np.array(images, dtype=np.float32)

#     # Compute the ensemble image by averaging
#     ensemble_image = np.mean(images_array, axis=0)

#     # Save the ensemble image
#     save_path = os.path.join(output_dir, image_name)
#     save_image(ensemble_image, save_path)

# print(f"Ensemble images saved successfully in '{output_dir}'")
