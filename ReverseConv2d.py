import torch 
import torch.nn as nn
import torchvision
import torchvision.models as models

class ReverseConv2d(nn.Module):
    def __init__(self,trained_layer,in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0,groups=1,bias=True,dilation=1,
                 padding_mode='zeros'):
        super(ReverseConv2d, self).__init__()
       # self.transpose2d = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,output_padding,
       #                                               groups,False,dilation,padding_mode)
        self.transpose2d = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,bias=False,groups=groups,padding_mode=padding_mode)
        #self.transpose2d.weight.data = trained_layer.weight.data
        self.transpose2d.weight.data = trained_layer.weight.data.permute(1, 0, 3, 2)
        self.bias = trained_layer.bias.data
        self.use_bias = bias
        
    def forward(self,deconv_input):
        print(self.transpose2d.weight.data.shape)
        if (self.use_bias):
            return self.transpose2d(deconv_input-self.bias[None,:,None,None])
        return self.transpose2d(deconv_input)
    
