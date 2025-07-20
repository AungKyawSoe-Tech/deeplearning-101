# deeplearning-101
DeepLearning 101 (Hello World in Deep Learning)

1. Introduction

This is to document learn how to create a HelloWorld version of Deep Learning. It is based on Code examples of Keras library [https://keras.io/examples/#code-examples]


2. Setting up the development environment

It is tried in WSL Windows Subsystem for Linux

Installation Steps for WSL 2.0 can be read below. First, enable WSL: Open PowerShell as the Administrator and run:
“dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart”
Next, enable Virtual Machine Platform: Run the following command to enable virtualization:
“dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart”
Update Linux Kernel: Download and install the latest WSL 2.0 Linux kernel update package from Microsoft's official site
wsl –install
Install a Linux Distribution: Use the Microsoft Store to install your preferred Linux distribution (e.g., Ubuntu). Alternatively, use:
wsl --install -d <distribution_name>
Set WSL 2.0 as Default: Run this command to set WSL 2.0 as the default version:
wsl --set-default-version 2

Then check Linux version by “uname -a”, install following modulees on WSL:
sudo apt install curl
sudo apt install unzip
sudo apt install gedit
sudo apt install python3-pip
sudo apt install python3-full
sudo apt install graphviz
pip3 install matplotlib
pip3 install pydot
python3 -m pip install tensorflow
python3 -m pip install keras

Setting and sourcing virtual environment can be done as follows:
learning@LAPTOP-O7E2D0MF:/mnt/c/WINDOWS/system32$ python3 --version
Python 3.12.3
sudo apt install python3.12-venv
python3 -m venv venv312; source venv312/bin/activate

learning @LAPTOP-O7E2D0MF:/mnt/c/WINDOWS/system32$ source venv312/bin/activate
(venv312) learning@LAPTOP-O7E2D0MF:/mnt/c/WINDOWS/system32$
Confirm keras is installed correctly as below:
python3 -m pip show keras

3. Running a sample program
Try Simple MNIST convnet [https://keras.io/examples/vision/mnist_convnet/], python module can be downloaded from: 
https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py
The same of running it can bee seen below:
python3 mnist_convnet.py


4. Creating a Hello World program based on an example from Keras

First “Hello World” program can be created based on this Image classification from scratch
[URI: https://keras.io/examples/vision/image_classification_from_scratch/]

First download data set as follows
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
Then unzip and confirm
unzip -q kagglecatsanddogs_5340.zip
ls

Then run the helloworld.py which is here Editing deeplearning-101/README.md at main · AungKyawSoe-Tech/deeplearning-101


Run the program as 
python3 helloworld.py


5. References
   
https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download
https://datasets.activeloop.ai/docs/ml/datasets/mnist/
https://codeburst.io/how-to-install-the-python-environment-for-ai-and-machine-learning-on-wsl2-612240cb8c0c



