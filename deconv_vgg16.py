import torch 
import torch.nn as nn
import torchvision
import torchvision.models as models
import ReverseConv2d as RevConv2d

class deconvolve_vgg16(nn.Module):
    def __init__(self, trained_model):
        super(deconvolve_vgg16, self).__init__()
        self.deconv_layers = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[28],512,512,3,padding = 1),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[26],512,512,3,padding = 1),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[24],512,512,3,padding = 1),
            
            nn.MaxUnpool2d(kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[21],512,512,3,padding = 1),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[19],512,512,3,padding = 1),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[17],512,256,3,padding = 1),
            
            nn.MaxUnpool2d(kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[14],256,256,3,padding = 1),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[12],256,256,3,padding = 1),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[10],256,128,3,padding = 1),
            
            nn.MaxUnpool2d(kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[7],128,128,3,padding = 1),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[5],128,64,3,padding = 1),
            
            nn.MaxUnpool2d(kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[2],64,64,3,padding = 1),
            nn.ReLU(inplace=True),
            RevConv2d.ReverseConv2d(trained_model.conv_layers[0],64,3,3,padding = 1),
        )
        
    def reconstruct(self,intermidiate_features,maxpool_indices,start_index):
        reconstructed_feature = intermidiate_features[start_index]
        deconv_start_index = len(self.deconv_layers) - start_index - 1
        for i in range(deconv_start_index, len(self.deconv_layers)):
            print(reconstructed_feature.shape,i)
            if isinstance (self.deconv_layers[i], nn.MaxUnpool2d):
                current_layer_index = len(self.deconv_layers) - i - 1
                current_indices = maxpool_indices[current_layer_index]
                reconstructed_feature = self.deconv_layers[i](reconstructed_feature, current_indices)
            else:
                reconstructed_feature = self.deconv_layers[i](reconstructed_feature)
        return reconstructed_feature
                        