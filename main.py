import torch 
import torch.nn as nn
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import numpy as np
import PIL
import ReverseConv2d as RevConv2d
import conv_vgg16 as conv
import deconv_vgg16 as deconv

#prepare to use vgg16
input_image = input('Enter path of the image:')
pretrained_vgg16_model = torchvision.models.vgg16(pretrained=True, progress=True)
img = cv2.imread(input_image, 1) #try 'D:/CNN/Dataset/849970-gr.jpg'

#resize img, normalization, and axis transpose -> prepare image I/O
default_vgg16_input_size = (224, 224) #(width, height)
img = cv2.resize(img, default_vgg16_input_size)/255.0
img_height,img_width,img_channels = img.shape
plt.imshow(img)
print('image loaded')
img = img.transpose(2,0,1).reshape((1,img_channels,img_height,img_width))
#print(img) #is used when we want to value it numeratly.
img = torch.FloatTensor(img)

#operating on vgg16
conv_VGG16_model = conv.vgg16(pretrained_vgg16_model)
intermidiate_features,maxpool_indices = conv_VGG16_model.forward(img)
print('vgg16 loaded')
#Change start_index here to adjust the model to get a clear image
input_start = input('Enter start_index:')
start_index = input_start
deconv_VGG16_model = deconv.deconvolve_vgg16(conv_VGG16_model)
result_img = deconv_VGG16_model.reconstruct(intermidiate_features,maxpool_indices,start_index)
print('model reconstructed')
#revise the output to print a understandable output.
def process_raw(raw):
    img = raw.data.numpy()[0].transpose(1,2,0)
    diff = (img.max()-img.min())
    if diff == 0:
        diff = 1e-6
    img = (img-img.min())/diff
    return img
#plot the reconstructed image out
plt.imshow(process_raw(result_img))
result_img = process_raw(result_img)
output_image = input('Enter the path to save the image(including its name):')
#change tensor into array, so that we can print it out
print (result_img.shape)
result_img = (result_img * 255).astype(np.uint8)  
result_img = PIL.Image.fromarray(result_img)
result_img.save(output_image)
print('Successfully completed')