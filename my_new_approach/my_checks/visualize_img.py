import os
import os.path as osp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# from utils.config import Config

# config = Config()

imgs_path = "/home/jsg2/Desktop/rhome/jsg2/prototype_learning_seque/data/Breast_US_Proto_Learning/images_and_masks" 
all_imgs = os.listdir(imgs_path)
just_us = [img for img in all_imgs if (not "tumor" in img) and (not "other" in img)]

print(just_us)

output_folder_path = f"./sample_imgs"
os.makedirs(output_folder_path,exist_ok=True)

for img_name in just_us[:4]:
    img_path = osp.join(imgs_path, img_name)

    img = Image.open(img_path)
    print(img.size)
    print(img.mode)

    # img_gray = img.convert("L").resize((config.img_size, config.img_size))
    
    img_gray = img.convert("L").resize((256, 256))

    img_gray.save(osp.join(output_folder_path, img_name))
