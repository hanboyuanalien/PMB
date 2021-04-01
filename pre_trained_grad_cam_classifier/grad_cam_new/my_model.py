import torch.nn as nn 
import torch.nn.functional as F
import numpy as np


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 2)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = relu_fc1(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels = 1 , num_classes = 2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), bias=False) 
        self.batch1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.batch2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3))
        self.batch3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3))
        self.batch4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3))
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        

        self.fc1 = nn.Linear(512*10*10, num_classes)
        self.soft = nn.Softmax(dim=1)
        

    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)

        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.pool5(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.soft(x)
        
        return x