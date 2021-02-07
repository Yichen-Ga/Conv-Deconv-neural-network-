# Conv-Deconv-neural-network-

  # Introduction
 This project implements zeiler's method to visualize deconvolutional neural network using Pytorch [1]. A convolutional neural network is also implemented to help to test the result. A pretrained VGG16 model and a few pictures was been used for the networks. I wish to demonstrate an example of writing multiple neural networks using Pytorch to vision people, and a practice of using theories of CNN in real coding. 
 
 Although this tutorial only shows how to do Conv/ Deconv neural network in VGG16. People can certainly look through the process and modify the code to apply a new type of model.
 I will also include some instructions to help people modify to apply their model.

  Note: This project is based on Windows operating system. The code itself may noet be effected much, but errors caused by environment may happen in a different operating system.

[1] Zeiler, Matthew D. and R. Fergus. "Visualizing and Understanding Convolutional Networks." ECCV (2014).

  # Steps
  # 1.Install Anaconda
  
  Anaconda is a software toolkit that creates virtual Python environments. We will use python libraries installed through Anaconda to do this project.
  
  Download the [Anaconda for Windows](https://www.anaconda.com/products/individual) in this link (which direct to their website). Once it is donwloaded, execute the installer and follow the instructions to complete the installing process.
  
  # 2.Build up environment
  
  I will include a environment file in the files. If the files aren't working in your computer or you want to build your own environment follow instructions below
  
  **2a.How to create an environment from an environment.yml file**
  
  Use the file by issuring:
  
  ```
  conda env create -f environment.yml
  ```
  
  Then activate the environment file and check the environment:
  
 ```
 activate environmentName
 conda env list
 ```
  
  **2b. Set up a new Anaconda virtual environment**
  
  We are going to use 3.0 version of python and newest pytorch and torchvision. Use the following code to create an environment named CNN for your project:
  
  ```
  conda create -n CNN python=3
  ```
  
  Then activate the environment to use and install libraries in this environment by issuing:
  
  ```
  activate CNN
  ```
  
  Install pytorch and torchvision by issuing:
  
  ```
  conda install pytorch torchvision -c pytorch
  ```
  
  Install the other necessary packages by issuing the following commands:
  
  ```
  conda install jupyter
  conda install numpy scipy scikit-learn
  conda install cv2 pickle matplotlib
  ```
  
  Note: If the cv2 and pickle cannot be installed try following commands:
  ```
  conda install opencv-python
  conda install -c menpo opencv
  conda install -c conda-forge pickle5
  ```
  
  # 3. Write a convolutional neural network depending on VGG16
  
  **3a.Some Insights about VGG16 Structure**
  
  Generally speaking convolutional neural networks (CNN) is a class of deep neural networks to analysis visual imagary. CNN is commonly used to deal with relatively small dataset. For more detailed information about CNN check [wikipedia on CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network).
  
  A general summary about architechture is shown by Figure1 and Figure2:
  
  VGG16 is one of VGG net. VGGs are a newly developed convolutional neural network which has high accuracy in ImageNet (which is one the on the largest data-set available). The VGG16 is a most commanly used in VGGs which is much deeper consisting 16 weight layers. 
  
  A summary about architechture is shown by Figure3 and Figure4:
  
  **3b. Building VGG16 convolutional neural networks class**
  In this section, we will build a VGG16 convolutional neural network depending on pretrained network in PyTorch. We will also write a function to load pretrained network and a forword to prepare parameters for deconvolution.
  
  
  
  
  
  

  
  
