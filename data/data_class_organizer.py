import os
import os.path as osp
import pandas as pd
import shutil
from tqdm import tqdm

data_root = r"C:\Users\seque\OneDrive - UAM\Escritorio\IPCV MASTER\Research_Lisbon_2025\ProtoPNet_seque_\prototype_learning_seque\data"
imgs_folder = "US_breast"
imgs = os.listdir(osp.join(data_root, imgs_folder))

csv_name = "US_breast_data_csv.xlsx"
data_csv = pd.read_excel(osp.join(data_root,csv_name), sheet_name="BrEaST-Lesions-USG clinical (3)")

if __name__ == "__main__":
    
    # Generate folders
    os.makedirs(osp.join(data_root,"US_breast_classes", "benign"), exist_ok=True)
    os.makedirs(osp.join(data_root, "US_breast_classes", "malignant"), exist_ok=True)
    # os.makedirs(osp.join(data_root, "US_breast_classes", "normal"), exist_ok=True)
    
    org_folder = "US_breast_classes"
    benign_path = osp.join(data_root,org_folder,"benign")
    malignant_path = osp.join(data_root,org_folder,"malignant")
    # normal_path = osp.join(data_root,org_folder,"normal")

    # Iterate over images and send them to their respective folders
    print("Starting transfer process...")
    for i, img in tqdm(enumerate(imgs), total=len(imgs), desc="Transfer Images"):

        if "tumor" not in img and "other" not in img:
            file_path = osp.join(data_root, imgs_folder, img)
   
            classif = data_csv[data_csv["Image_filename"] == img]["Classification"]
            classif = classif.item()

            if classif == "benign":
                if not os.path.exists(osp.join(benign_path, img)):
                    tqdm.write(f"File {img} copied to BENIGN folder")
                    shutil.copyfile(file_path, osp.join(benign_path, img))
                else:
                    tqdm.write(f"File {img} ALREADY copied")
            # elif classif == "normal":
            #     if not os.path.exists(osp.join(normal_path, img)):
            #         tqdm.write(f"File {img} copied to NORMAL folder")
            #         shutil.copyfile(file_path, osp.join(normal_path, img))

            #     else:
            #         tqdm.write(f"File {img} ALREADY copied")
            elif classif == "malignant":
                if not os.path.exists(osp.join(malignant_path, img)):
                    tqdm.write(f"File {img} copied to MALIGNANT folder")
                    shutil.copyfile(file_path, osp.join(malignant_path, img))

                else:
                    tqdm.write(f"File {img} ALREADY copied")    
        else:
            continue

    print("Finished Process!")

    print("#"*5)
    print("#"*5)
    print("Stats:")
    # print(f"Normal subjects = {len(os.listdir(normal_path))}")
    print(f"Benign subjects = {len(os.listdir(benign_path))}")
    print(f"Malignant subjects = {len(os.listdir(malignant_path))}")
    print("#"*5)
    print("#"*5)
            # shutil.copy(file_path, osp.join())
    # print(imgs)
    # print(data_csv["BIRADS"][data_csv["Classification"]=="benign"])

    print(data_csv.head())

    ## Develop Function to generate dataset based on feature

    


