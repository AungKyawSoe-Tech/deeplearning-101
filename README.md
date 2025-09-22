# DeepLearning 101 (Hello World in Deep Learning)

## 1. Introduction

This is to document learn how to create a HelloWorld version of Deep Learning. It is based on Code examples of Keras library [https://keras.io/examples/#code-examples]


## 2. Setting up the development environment

It is tried in WSL Windows Subsystem for Linux

Use WSL 2.0

Install Ubuntu on WSL 

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

Confirm keras library is installed correctly as partt of Python 3 by following command

python3 -m pip show keras

## 3. Downloading data set

First download data set as follows
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
Then unzip and confirm "unzip -q kagglecatsanddogs_5340.zip && ls"

### Working across Windows and Linux file systems
[How to see folders in WSL from Windows] (https://learn.microsoft.com/en-us/windows/wsl/filesystems)

## 4. Running the Helloworld script which will train the model using the data set
Then run the helloworld.py which is here 

[AungKyawSoe-Tech/deeplearning-101] https://github.com/AungKyawSoe-Tech/deeplearning-101/

Run the program as 
python3 helloworld.py


## 5. References

https://keras.io/examples/vision/mnist_convnet/
https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download
https://datasets.activeloop.ai/docs/ml/datasets/mnist/
https://codeburst.io/how-to-install-the-python-environment-for-ai-and-machine-learning-on-wsl2-612240cb8c0c



