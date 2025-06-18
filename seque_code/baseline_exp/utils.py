import torch
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import os

def line_break():
    print("\n")

def samples_per_class(set_name, dataset):

    tot_ben = 0
    tot_mal = 0 
    for i in range(len(dataset)):
        _, lab = dataset[i]
        if lab == 0: tot_ben += 1
        else: tot_mal += 1

    print(f"{set_name}")
    print(f'Benign: {tot_ben}')
    print(f'Malignant: {tot_mal} \n')

def get_dimensions(dataset):
    min_x, max_x = 2000, 0
    min_y, max_y = 2000, 0

    for i in range(len(dataset)):
        img, _ = dataset[i]

        if img.shape[1] > max_x:
            max_x = img.shape[1]
            
        if img.shape[2] > max_y:
            max_y = img.shape[2]

        if img.shape[1] < min_x:
            min_x = img.shape[1]

        if img.shape[2] < min_y:
            min_y = img.shape[1]
        
    print(f'Max X | Max Y --> ({max_x},{max_y})')
    print(f'Min X | Min Y --> ({min_x},{min_y})')

def show_sample(dataloader, samp_num:int, class_dict:dict):
    for i, (x,y) in enumerate(dataloader):
        assert samp_num <= x.shape[0], "Sample chosen must be lower or equal than Batch Size"
        if i == 0:
            img = x[samp_num]
            print(img.shape)

            img =  img.permute(1,2,0)
            plt.imshow(img, cmap="gray")
            plt.title(class_dict[y[0].item()])
            plt.show()
        else: break


def load_model(model_name, num_classes, model_path:str = "", load_model:bool = False):

    # models_dict = {"resnet18": models.resnet18(pretrained=True),
    #                "resnet50": models.resnet50(pretrained=True),
    #                "resnet101": models.resnet101(pretrained=True),
    #                "resnet152": models.resnet152(pretrained=True),
    #                "vit_b32": models.vit_b_32(pretrained=True),
    #                "resnet18": models.resnet18(pretrained=True),}

    if(load_model==False):
    #   Choose model
      if model_name == "resnet18":
          model = models.resnet18(pretrained=True)
          num_features = model.fc.in_features
          model.fc = nn.Linear(num_features, num_classes)

          old_weights = model.conv1.weight.data
        #   model.
          model.conv1 = nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3, bias=False)
          model.conv1.weight.data = old_weights.mean(dim=1, keepdim=True)


      elif model_name == "resnet50":
          model = models.resnet50(pretrained=True)
          num_features = model.fc.in_features
          model.fc = nn.Linear(num_features, num_classes)

          old_weights = model.conv1.weight.data
        #   model.
          model.conv1 = nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3, bias=False)
          model.conv1.weight.data = old_weights.mean(dim=1, keepdim=True)

      elif model_name == "densenet169":
          model = models.densenet169(pretrained=True)
          num_features = model.classifier.in_features
          model.classifier = nn.Linear(num_features, num_classes)

      elif model_name == "eb3":
          model = models.efficientnet_b3(pretrained=True)
          num_features = model.classifier[-1].in_features
          model.classifier[-1] = nn.Linear(num_features, num_classes)

      elif model_name == "vgg16":
          model = models.vgg16(pretrained=True)
          model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
      else:
          raise ValueError("Model not available. Choose an offered one.")
    else:
      model = torch.load(model_path)

    return model

def save_metrics_plots(model_dir, train_acc, test_acc, train_F1, test_F1, train_ba, test_ba, train_loss, val_loss):
    plt.plot(train_acc, "b", label="train")
    plt.plot(test_acc, "r", label="test")
    plt.legend()
    plt.savefig(model_dir + 'train_test_accu.png')
    plt.close()

    plt.plot(train_F1, "b", label="train")
    plt.plot(test_F1, "r", label="test")
    plt.legend()
    plt.savefig(model_dir + 'train_test_F1.png')
    plt.close()

    plt.plot(train_ba, "b", label="train")
    plt.plot(test_ba, "r", label="test")
    plt.legend()
    plt.savefig(model_dir + 'train_test_ba.png')
    plt.close()


    plt.plot(train_loss, "b", label="train")
    plt.plot(val_loss, "r", label="test")
    plt.legend()
    plt.savefig(model_dir + 'train_test_loss.png')
    plt.close()   

# Miguel

def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    counter = [0]
    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
        # Question: do we need to flush()
    return logger, f.close

def save_model_w_condition(model, model_dir, model_name, ba, target_ba, log=print):
    '''
    model: this is not the multigpu model
    '''
    if ba > target_ba:
        log('\tBA above {0:.2f}%'.format(target_ba * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '_{0:.4f}.pth').format(ba)))
