import os
from dataset import USG_Dataset
root_dir = r"C:\Users\seque\OneDrive - UAM\Escritorio\IPCV MASTER\Research_Lisbon_2025\ProtoPNet_seque_\prototype_learning_seque\data\US_breast_train_val"
mode = "train"
excel_file = "US_breast_data_csv.xlsx"
transform = False
feature = "Shape" #"Margin"

dataset = USG_Dataset(root_dir=root_dir,
                      mode=mode,
                      is_push=False,
                      excel_file=excel_file,
                      feature=feature,
                      OA_transform=transform)

img, label = dataset[111]
print(label)

