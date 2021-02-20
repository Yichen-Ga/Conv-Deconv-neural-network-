import torch 
import torch.nn as nn
import torchvision
import torchvision.models as models

class vgg16(nn.Module):
    def __init__(self, trained_model):
        super(vgg16, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,64,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),#4
            
            nn.Conv2d(64,128,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),#9
            
            nn.Conv2d(128,256,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),#16
            
            nn.Conv2d(256,512,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True), #23
            
            nn.Conv2d(512,512,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True), #30
        )
        self.load_pretrained(trained_model)
        
    def load_pretrained(self, trained_model):
        for i, layer in enumerate(trained_model.features):
            if isinstance(layer,nn.Conv2d):
                self.conv_layers[i].weight.data = layer.weight.data
                self.conv_layers[i].bias.data = layer.bias.data
    
    def forward(self,image):
        maxpool_indices = {}
        intermidiate_features = []
        for i,layer in enumerate(self.conv_layers):
            if isinstance(layer,nn.MaxPool2d):
                image, indices = layer(image)
                maxpool_indices[i] = indices
                intermidiate_features.append(image)
            else:
                image = layer(image)
                intermidiate_features.append(image)
        return intermidiate_features,maxpool_indices
    