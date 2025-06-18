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
from sklearn.metrics import classification_report, accuracy_score 

# Own implemented stuff
from settings import data_path, train_perc, batch_size, sample_display, class_dict, data_info_display, img_size
from settings import lr, base_architecture, experiment_run, num_train_epochs
from utils import * #samples_per_class, show_sample, get_dimensions

# Data loading
data_transform = transforms.Compose([transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                                     transforms.Grayscale(),
                                     transforms.ToTensor(),
                                    ])
data_trans_no_resize = transforms.Compose([transforms.Grayscale(),
                                     transforms.ToTensor(),
                                    ])

data = datasets.ImageFolder(data_path,transform=data_transform)
data_4dims = datasets.ImageFolder(data_path,transform=data_trans_no_resize)

# Train-Test split
train_samples = int(train_perc * len(data))
test_samples = len(data) - train_samples

generator = torch.Generator().manual_seed(14)
train_data , test_data = random_split(data, [train_samples, test_samples])


if data_info_display:
    get_dimensions(dataset=data_4dims)
    line_break()


    # EDA (samples per class)
    samples_per_class(set_name="Training Set", dataset=train_data)
    samples_per_class(set_name="Test Set", dataset=test_data)

# Dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False)

# print([data.targets[i] for i in train_data.indices])

# (Optional) Display sample
if sample_display: show_sample(dataloader=train_loader, samp_num=batch_size-2, class_dict=class_dict)

# Balance training  (Imbalanced classes)
train_labels = np.array([data.targets[i] for i in train_data.indices])
weights = compute_class_weight(class_weight='balanced',
                               classes= np.unique(train_labels),
                               y=train_labels)

for i, w in enumerate(weights):
    print(f'Class {class_dict[i]}: {w:.2f}')
print("\n")

num_classes = len(weights)
# Load model(s)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_name="resnet18",
           num_classes=num_classes,
           model_path="")
model = model.to(device)
#############################
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Define loss function with weighted cross entropy
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights) # Weighted Cross Entropy

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
        all_predicted_train.extend(predicted.detach().cpu().tolist())
        all_target_train.extend(labels.detach().cpu().tolist())



        loss = criterion(outputs, labels)
        n_batches_train += 1
        total_cross_entropy_train += loss.item()
        loss.backward()
        optimizer.step()
    
    loss_training=total_cross_entropy_train / n_batches_train
    log('\tcross ent train: \t{0}'.format(total_cross_entropy_train / n_batches_train))

    for a in range(num_classes):    
        log('\t{0}'.format( np.array2string(confusion_matrix(all_target_train, all_predicted_train)[a])))

    log('{0}'.format( classification_report(all_target_train, all_predicted_train) ))

    pr_train, rc_train, f1_train, sp_train = precision_recall_fscore_support(all_target_train, all_predicted_train,
                                                    average='macro')
    accu_train=accuracy_score(all_target_train, all_predicted_train)
    
    log('\tmacro-averaged precision : \t{0}'.format(pr_train))
    log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc_train))
    log('\tmacro-averaged F1 score: \t{0}'.format(f1_train))
    log('\ttrain accuracy: \t{0}'.format(accu_train))

    # if(num_classes==8):
    #     log('\n\tMalignant (1) vs Benign (0) information')
    #     BM_all_target_train = np.where(np.isin(all_target_train, [0,1, 4, 6]), 1, 0)
    #     BM_all_predicted_train = np.where(np.isin(all_predicted_train, [0,1, 4, 6]), 1, 0)
    #     for a in range(2):    
    #         log('\t{0}'.format( np.array2string(confusion_matrix(BM_all_target_train, BM_all_predicted_train)[a])))
    #     log('{0}'.format( classification_report(BM_all_target_train, BM_all_predicted_train) ))
    
    
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

            all_predicted_val.extend(predicted.detach().cpu().tolist())
            all_target_val.extend(labels.detach().cpu().tolist())
    
    loss_validation=total_cross_entropy_val / n_batches_val
    log('\tcross ent val: \t{0}'.format(total_cross_entropy_val / n_batches_val))

    for a in range(num_classes):    
        log('\t{0}'.format( np.array2string(confusion_matrix(all_target_val, all_predicted_val)[a])))

    log('{0}'.format( classification_report(all_target_val, all_predicted_val) ))

    pr_val, rc_val, f1_val, sp_val = precision_recall_fscore_support(all_target_val, all_predicted_val,
                                                    average='macro')
    accu_val=accuracy_score(all_target_val, all_predicted_val)
    
    log('\tmacro-averaged precision : \t{0}'.format(pr_val))
    log('\tmacro-averaged recall or Balanced Accuracy (BA) : \t{0}'.format(rc_val))
    log('\tmacro-averaged F1 score: \t{0}'.format(f1_val))
    log('\tvalidation accuracy: \t{0}'.format(accu_val))

    # if(num_classes==8):
    #     log('\n\tMalignant (1) vs Benign (0) information')
    #     BM_all_target_val = np.where(np.isin(all_target_val, [0,1, 4, 6]), 1, 0)
    #     BM_all_predicted_val = np.where(np.isin(all_predicted_val, [0,1, 4, 6]), 1, 0)
    #     for a in range(2):    
    #         log('\t{0}'.format( np.array2string(confusion_matrix(BM_all_target_val, BM_all_predicted_val)[a])))
    #     log('{0}'.format( classification_report(BM_all_target_val, BM_all_predicted_val) ))

    save_model_w_condition(model=model, model_dir=model_dir, model_name=str(epoch),ba=rc_val,target_ba=currbest_ba,log=log)
    if currbest_ba < rc_val:
        currbest_ba  = rc_val
        best_epoch_ba = epoch

    if currbest < accu_val:
        currbest = accu_val
        best_epoch = epoch

    if currbest_F1 < f1_val:
        currbest_F1  = f1_val
        best_epoch_F1 = epoch
    
    log("\n\tcurrent best accuracy is: \t\t{} at epoch {}".format(currbest, best_epoch))
    log("\tcurrent best F1 is: \t\t{} at epoch {}".format(currbest_F1, best_epoch_F1))
    log("\tcurrent best BA is: \t\t{} at epoch {}".format(currbest_ba, best_epoch_ba))
    

    train_accu.append(accu_train)
    train_F1_score.append(f1_train)
    train_ba.append(rc_train)

    test_accu.append(accu_val)
    test_F1_score.append(f1_val)
    test_ba.append(rc_val)

    train_loss.append(loss_training)
    val_loss.append(loss_validation)
 
    save_metrics_plots(model_dir=model_dir,
                       train_acc=train_accu,
                       test_acc=test_accu,
                       train_F1=train_F1_score,
                       test_F1=test_F1_score,
                       train_ba=train_ba,
                       test_ba=test_ba,
                       train_loss=train_loss,
                       val_loss=val_loss)