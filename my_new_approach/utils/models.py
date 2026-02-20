import torch
import torch.nn as nn
from torchvision import models

class ResNet:

    def __init__(self, config, logger) -> None:
        self.config = config
        self.logger = logger

    
    def load_model(self):

        models_dict = {"resnet18":models.resnet18(weights=self.config.weights),
                    "resnet34":models.resnet34(weights=self.config.weights),
                    "resnet50":models.resnet50(weights=self.config.weights),
                    "resnet101":models.resnet101(weights=self.config.weights),
                    "resnet152":models.resnet152(weights=self.config.weights),
                    }
        
        try:
            model = models_dict[self.config.model_name]
            architecture = self.config.model_name

            self.logger.info(f"Loading {architecture}!")

        except KeyError:

            model = models_dict["resnet18"]
            architecture = "resnet18"

            self.logger.info(f"Provided model ({self.config.model_name}) not available. Automatically loaded ResNet18...")

        
        model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)

        if not self.config.load_new_model:
            state_dict = torch.load(self.config.pretrained_model_path)
            model.load_state_dict(state_dict=state_dict)

            self.logger.info(f"Loading PRETRAINED model")


        return model, architecture