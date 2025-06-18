import os
import pandas as pd
import shutil
import os.path as osp
import random

root = r"C:\Users\seque\OneDrive - UAM\Escritorio\IPCV MASTER\Research_Lisbon_2025\ProtoPNet_seque_\prototype_learning_seque\data"
img_dir = r"US_breast"
imgs_path = osp.join(root, img_dir)
info_df = pd.read_excel(r"C:\Users\seque\OneDrive - UAM\Escritorio\IPCV MASTER\Research_Lisbon_2025\ProtoPNet_seque_\prototype_learning_seque\data\US_breast_data_csv.xlsx", sheet_name="BrEaST-Lesions-USG clinical (3)")
normal_samples = info_df["Image_filename"][info_df["Classification"] == "normal"].values

# print(normal_samples)
# print(info_df.head())

img_files = [file for file in sorted(os.listdir(imgs_path)) if (not "tumor" in file) and (not "other" in file)and (not file in normal_samples)]

os.makedirs(osp.join(root, "US_breast_train_val"), exist_ok=True)
os.makedirs(osp.join(root, "US_breast_train_val", "train"),exist_ok=True)
os.makedirs(osp.join(root, "US_breast_train_val", "val"),exist_ok=True)

train_size = 0.85

random.seed(33)
train_set = random.sample(img_files, int(len(img_files)*train_size))
test_set = [file for file in img_files if file not in train_set]

print(f" Number Training Samples: {len(sorted(train_set))}")
print(f" Number Test Samples: {len(sorted(test_set))}")

for file in train_set:
    shutil.copy(osp.join(root,img_dir,file), osp.join(root,"US_breast_train_val","train",file))

for file in test_set:
    shutil.copy(osp.join(root,img_dir,file), osp.join(root,"US_breast_train_val","val",file))