# General
import os
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Torch
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from sklearn.metrics._classification import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score

# Own implemented stuff
from settings import data_path, train_perc, batch_size, sample_display, class_dict, data_info_display, img_size
from settings import lr, base_architecture, experiment_run, num_train_epochs
from utils import * #samples_per_class, show_sample, get_dimensions
from dataset import USG_Dataset, USG_Dataset_all_feat


#############################################################################
############## ############################################ #################
#############################################################################

#############################################################################
############ DATASET CREATION + DATALOADERS(TRAIN, PUSH, VAL) ###############
#############################################################################

root_dir = r"C:\Users\seque\OneDrive - UAM\Escritorio\IPCV MASTER\Research_Lisbon_2025\ProtoPNet_seque_\prototype_learning_seque\data\US_breast_train_val"
mode = "train"
excel_file = "US_breast_data_csv.xlsx"
transform = True
feature = "Echogenicity" #"Margin"

# load the data

train_batch_size = 8
test_batch_size = 8
train_push_batch_size = 8
# all datasets
# train set 
# Augmentation just in this set to avoid setting augmented samples as prototypes
# train_dataset = USG_Dataset(train_dir, train_mask_dir, is_train=True,number_classes=num_classes,augmentation=online_augmentation)
# train_dataset = USG_Dataset_all_feat(root_dir=root_dir,
#                     mode="train",
#                     is_push=False,
#                     excel_file=excel_file,
#                 #   feature=feature,
#                     OA_transform=transform)
train_dataset = USG_Dataset(root_dir=root_dir,
                    mode="train",
                    is_push=False,
                    excel_file=excel_file,
                    feature=feature,    
                    OA_transform=transform)
train_loader = DataLoader(
train_dataset, batch_size=train_batch_size, shuffle=True,
pin_memory=False) # num_workers=4,

# mode = "val"
test_dataset = USG_Dataset(root_dir=root_dir,
                    mode="val",
                    is_push=False,
                    excel_file=excel_file,
                    feature=feature,
                    OA_transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    pin_memory=False) # num_workers=4,
# print([data.targets[i] for i in train_data.indices])

# (Optional) Display sample
if sample_display: show_sample(dataloader=train_loader, samp_num=batch_size-2, class_dict=class_dict)

#############################################################################
############## ############################################ #################
#############################################################################

#############################################################################
############ COMPUTE CLASS WEIGHTS TO BALANCE TRAINING (DIFF) ###############
#############################################################################
# Balance training  (Imbalanced classes)
# train_labels = np.array([data.targets[i] for i in train_data.indices])

# all_labels = []
# for _, label in train_dataset:
#     all_labels.append(label.numpy())
# unique_labels = np.unique(np.array(all_labels), axis=0)

# int_values = range(0,13,1)
# labels_dict = dict(zip(int_values, unique_labels))
# labels_int = []
# for vector in all_labels:
#     # print(vector.tolist())
#     for key, value in labels_dict.items():
#         if np.array_equal(vector, value):
#             # print(f'{value} == {vector}')
#             labels_int.append(key)


# weights = compute_class_weight(class_weight='balanced',
#                                classes= np.array(int_values),
#                                y=labels_int)

# for i, w in enumerate(weights):
#     # print(f'Class {class_dict[i]}: {w:.2f}')
    
#     print(f'Class {i}: {w:.2f}')
# print("\n")

# num_classes = len(weights)
# Load model(s)

if feature != "Margin":
    train_labels = []
    for _, labels in train_dataset:
        train_labels.append(np.argmax(labels.numpy()))

    weights = compute_class_weight(class_weight='balanced',
                               classes= np.unique(train_labels),
                               y=train_labels)

    for i, w in enumerate(weights):
    # print(f'Class {class_dict[i]}: {w:.2f}')
    
        print(f'Class {i}: {w:.2f}')
        print("\n")
#############################################################################
############## ############################################ #################
#############################################################################

#############################################################################
############ CONFIGURATION SETUP (DEVICE, MODEL OPTIM,CRIT..) ###############
#############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device}")
#############################

# Multiclass
if feature != "Margin":
    num_classes = len(weights)
    class_weights = torch.FloatTensor(weights).to(device)
    model = load_model(model_name="resnet18",
           num_classes= num_classes,
           model_path="").to(device)
    
    # Define loss function with weighted cross entropy
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Weighted Cross Entropy

# Multilabel
else:
    model = load_model(model_name="resnet18",
           num_classes= 5,#num_classes,
           model_path="").to(device)
    criterion = nn.BCELoss()

import torch.optim as optim
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=lr) #, momentum=0.9)

# Train the model
main_model_folder = "models"
# model_dir = folder_path_to_save_runs + '/' + base_architecture + '/' + experiment_run + '/'
model_dir = f'./{main_model_folder}/{base_architecture}/{experiment_run}'
print("saving models to: ", model_dir)
os.makedirs(model_dir, exist_ok=True)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
currbest_ba,best_epoch_ba=0,-1
currbest, best_epoch = 0, -1
currbest_F1,best_epoch_F1=0, -1

train_accu=[]
train_F1_score=[]
train_ba=[]

test_accu=[]
test_F1_score=[]
test_ba=[]

train_loss=[]
val_loss=[]

#############################################################################
############## ############################################ #################
#############################################################################

#############################################################################
############                 MODEL TRAINING                   ###############
#############################################################################

for epoch in range(num_train_epochs):
    log('\n\t ### Epoch: \t{0}'.format(epoch))
    # Train
    log('\ttrain')
    all_predicted_train, all_target_train = [], []
    model.train()
    n_batches_train=0
    total_cross_entropy_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        _, target = torch.max(labels,1)
        
        # argmax_vectors[outputs.argmax()] = 1
        if feature != "Margin":
            # all_predicted_train.extend(argmax_vectors.detach().cpu().numpy())
            # all_target_train.extend(labels.detach().cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            _, target = torch.max(labels,1)
            all_predicted_train.extend(predicted.detach().cpu().numpy())
            all_target_train.extend(target.detach().cpu().numpy())
            
        else:
            all_predicted_train.extend(outputs.detach().cpu().numpy())
            all_target_train.extend(labels.detach().cpu().numpy())



        loss = criterion(outputs, labels)
        n_batches_train += 1
        total_cross_entropy_train += loss.item()
        loss.backward()
        optimizer.step()

    # print(predicted)
    # print(target)
    # print(labels)
    # print(outputs)
   

    # all_predicted_train = np.vstack(all_predicted_train)
    # all_target_train = np.vstack(all_target_train)
    if feature == "Margin":
        # all_predicted_train = np.vstack(all_predicted_train)
        # all_target_train = np.vstack(all_target_train)
        
        # print(outputs)
        # print(labels)
        print(all_predicted_train)
        print(all_target_train)
        all_binary_preds_train = (all_predicted_train > 0.5).astype(int)

        f1_train = f1_score(all_target_train, all_binary_preds_train, average='macro')
        auc_train = roc_auc_score(all_target_train, all_binary_preds_train, average='macro')

        log('\tmacro-averaged F1 score: \t{0}'.format(f1_train))
        log('\tmacro-averaged AUC: \t{0}'.format(auc_train))

    loss_training=total_cross_entropy_train / n_batches_train
    log('\tcross ent train: \t{0}'.format(total_cross_entropy_train / n_batches_train))
    bal_acc_train = balanced_accuracy_score(all_target_train, all_predicted_train)
    log('\tmacro-averaged AUC: \t{0}'.format(bal_acc_train))
    # for a in range(num_classes):    
    #     log('\t{0}'.format( np.array2string(confusion_matrix(all_target_train, all_predicted_train)[a])))

    # log('{0}'.format( classification_report(all_target_train, all_predicted_train) ))

    pr_train, rc_train, f1_train, sp_train = precision_recall_fscore_support(all_target_train, all_predicted_train,
                                                    average='macro')
    accu_train=accuracy_score(all_target_train, all_predicted_train)
    
    log('\tmacro-averaged precision : \t{0}'.format(pr_train))
    log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc_train))
    log('\tmacro-averaged F1 score: \t{0}'.format(f1_train))
    log('\ttrain accuracy: \t{0}'.format(accu_train))
    
    
    # Validate
    log('\n\tvalidation')
    all_predicted_val, all_target_val = [], []
    model.eval()
    n_batches_val=0
    total_cross_entropy_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            n_batches_val += 1
            total_cross_entropy_val += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            # all_predicted_val.extend(predicted.detach().cpu().tolist())
            # all_target_val.extend(labels.detach().cpu().tolist())
            all_predicted_val.extend(outputs.detach().cpu().numpy())
            all_target_val.extend(labels.detach().cpu().numpy())
    
    all_predicted_val = np.vstack(all_predicted_val)
    all_target_val = np.vstack(all_target_val)

    loss_validation=total_cross_entropy_val / n_batches_val
    log('\tcross ent val: \t{0}'.format(total_cross_entropy_val / n_batches_val))
    

    all_binary_preds_val = (all_predicted_val > 0.5).astype(int)
    # print(all_binary_preds_train)
    f1_val = f1_score(all_target_val, all_binary_preds_val, average='macro')
    auc_val = roc_auc_score(all_target_val, all_binary_preds_val, average='macro')

    log('\tmacro-averaged F1 score: \t{0}'.format(f1_val))
    log('\tmacro-averaged AUC: \t{0}'.format(auc_val))
    # for a in range(num_classes):    
    #     log('\t{0}'.format( np.array2string(confusion_matrix(all_target_val, all_predicted_val)[a])))

    # log('{0}'.format( classification_report(all_target_val, all_predicted_val) ))

    # pr_val, rc_val, f1_val, sp_val = precision_recall_fscore_support(all_target_val, all_predicted_val,
    #                                                 average='macro')
    # accu_val=accuracy_score(all_target_val, all_predicted_val)
    
    # log('\tmacro-averaged precision : \t{0}'.format(pr_val))
    # log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc_val))
    # log('\tmacro-averaged F1 score: \t{0}'.format(f1_val))
    # log('\tvalidation accuracy: \t{0}'.format(accu_val))

    # if(num_classes==8):
    #     log('\n\tMalignant (1) vs Benign (0) information')
    #     BM_all_target_val = np.where(np.isin(all_target_val, [0,1, 4, 6]), 1, 0)
    #     BM_all_predicted_val = np.where(np.isin(all_predicted_val, [0,1, 4, 6]), 1, 0)
    #     for a in range(2):    
    #         log('\t{0}'.format( np.array2string(confusion_matrix(BM_all_target_val, BM_all_predicted_val)[a])))
    #     log('{0}'.format( classification_report(BM_all_target_val, BM_all_predicted_val) ))

    # save_model_w_condition(model=model, model_dir=model_dir, model_name=str(epoch),ba=rc_val,target_ba=currbest_ba,log=log)
    # if currbest_ba < rc_val:
    #     currbest_ba  = rc_val
    #     best_epoch_ba = epoch

    # if currbest < accu_val:
    #     currbest = accu_val
    #     best_epoch = epoch

    # if currbest_F1 < f1_val:
    #     currbest_F1  = f1_val
    #     best_epoch_F1 = epoch
    
    # log("\n\tcurrent best accuracy is: \t\t{} at epoch {}".format(currbest, best_epoch))
    # log("\tcurrent best F1 is: \t\t{} at epoch {}".format(currbest_F1, best_epoch_F1))
    # log("\tcurrent best BA is: \t\t{} at epoch {}".format(currbest_ba, best_epoch_ba))
    

    # train_accu.append(accu_train)
    # train_F1_score.append(f1_train)
    # train_ba.append(rc_train)

    # test_accu.append(accu_val)
    # test_F1_score.append(f1_val)
    # test_ba.append(rc_val)

    # train_loss.append(loss_training)
    # val_loss.append(loss_validation)
 
    # save_metrics_plots(model_dir=model_dir,
    #                    train_acc=train_accu,
    #                    test_acc=test_accu,
    #                    train_F1=train_F1_score,
    #                    test_F1=test_F1_score,
    #                    train_ba=train_ba,
    #                    test_ba=test_ba,
    #                    train_loss=train_loss,
    #                    val_loss=val_loss)