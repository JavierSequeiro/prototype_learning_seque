import os
import os.path as  osp
import pandas as pd


data_root = r"C:\Users\seque\OneDrive - UAM\Escritorio\IPCV MASTER\Research_Lisbon_2025\ProtoPNet_seque_\prototype_learning_seque\data"

csv_name = "US_breast_data_csv.xlsx"
data_csv = pd.read_excel(osp.join(data_root,csv_name), sheet_name="BrEaST-Lesions-USG clinical (3)")

# Shape	Margin	Echogenicity	Posterior_features	Halo	Calcifications	Skin_thickening
shape_ = data_csv["Shape"]
margin = data_csv["Margin"]
echogenicity = data_csv["Echogenicity"]
posterior_features = data_csv["Posterior_features"]
halo = data_csv["Halo"]
calcifications = data_csv["Calcifications"]
skin_thickening = data_csv["Skin_thickening"]

if __name__ == "__main__":
    print(shape_.value_counts(), "\n")
    print(margin.value_counts(), "\n")
    print(echogenicity.value_counts(), "\n")
    print(posterior_features.value_counts(), "\n")
    print(halo.value_counts(), "\n")
    print(calcifications.value_counts(), "\n")
    print(skin_thickening.value_counts(), "\n")
    