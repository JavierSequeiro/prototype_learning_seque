import os
import torch
import torchvision.transforms as transforms



class Config:

    def __init__(self) -> None:
        
        # Data Loading
        self.data_root = "/home/jsg2/Desktop/rhome/jsg2/prototype_learning_seque/data/Breast_US_Proto_Learning"
        
        #Training Params
        self.img_size = 256
        self.batch_size = 16

        # Model Loading
        self.model_name = "resnet18"
        