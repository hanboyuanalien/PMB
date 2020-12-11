import pandas as pd
import requests
import csv
import SimpleITK as sitk
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torch

input_path = []
patient = []
out_path = []
for patient_ix in range(1,3):
    ac_path = 'Patient'+f'{patient_ix:02d}' + '/' +'Patient' + f'{patient_ix:02d}' + '_AC.gipl'
    itk_image = sitk.ReadImage(ac_path)
    ac_image = sitk.GetArrayFromImage(itk_image)
    nac_path = 'Patient'+f'{patient_ix:02d}' + '/' +'Patient' + f'{patient_ix:02d}' + '_NAC.gipl'
    itk_image1 = sitk.ReadImage(nac_path)
    nac_image = sitk.GetArrayFromImage(itk_image1)
    #for slice_ix in range(ac_image.shape[0]):
    patient.append('Patient'+f'{patient_ix:02d}')
    input_path.append(nac_path)
    out_path.append(ac_path)
    c ={'patient':patient,'input_path':input_path,'out_path':out_path}
df = pd.DataFrame(c)
af = df.to_csv('112.csv',header=None)

class MyDataset(Dataset):
    def __init__(self, csv, transform=None, target_transform=None):
        df = pd.read_table(csv,
                   sep=',', 
                   header=None, 
                   names=['patient', 'input_path', 'out_path'])
        trainin = []
        trainout = []
        testin = []
        testout = []
        kf = KFold(n_splits=2)
        kf.get_n_splits(df['patient'])
        for train_index, test_index in kf.split(df['patient']):
            X_train, X_test =df['input_path'][train_index], df['input_path'][test_index]
            y_train, y_test = df['out_path'][train_index], df['out_path'][test_index]
            for x in  X_train:
                itk_image = sitk.ReadImage(x)
                train_in_image = sitk.GetArrayFromImage(itk_image)
                for slice_ix in range(train_in_image.shape[0]):
                    trainin.append((x,slice_ix))
            for y in  y_train:
                itk_image1 = sitk.ReadImage(y)
                train_out_image = sitk.GetArrayFromImage(itk_image1)
                for slice_ix in range(train_out_image.shape[0]]):
                    trainout.append((y,slice_ix))
            for x1 in X_test:  
                itk_image2 = sitk.ReadImage(x1)
                test_in_image = sitk.GetArrayFromImage(itk_image2)
                for slice_ix in range(test_in_image.shape[0]):
                    testin.append((x1,slice_ix))
            for y1 in y_test:
                itk_image3 = sitk.ReadImage(y1)
                test_out_image = sitk.GetArrayFromImage(itk_image3)
                for slice_ix in range(test_out_image.shape[0]):
                    testout.append((y1,slice_ix))
 
        self.trainin = trainin
        self.trainout = trainout
        self.testin = testin
        self.testout = testout
        self.transform = transform
        self.target_transform = target_transform
       
        
        
        

    def __getitem__(self, index):
        path1, id1 = self.trainin[index]
        path2, id2= self.trainout[index]
        path3, id3= self.testin[index]
        path4, id4= self.testout[index]
        it1 = sitk.ReadImage(path1)
        s = sitk.GetArrayFromImage(it1)
        sl = s[id1,:,:]
        X_train = Image.fromarray((sl/np.max(sl)*0xff).astype(np.uint8), mode="L")
        it2 = sitk.ReadImage(path2)
        s1 = sitk.GetArrayFromImage(it2)
        sl1 = s1[id2,:,:]
        Y_train = Image.fromarray((sl1/np.max(sl1)*0xff).astype(np.uint8), mode="L")
        it3 = sitk.ReadImage(path3)
        s2 = sitk.GetArrayFromImage(it3)
        sl2 = s2[id3,:,:]
        X_test = Image.fromarray((sl2/np.max(sl2)*0xff).astype(np.uint8), mode="L")
        it4 = sitk.ReadImage(path4)
        s3 = sitk.GetArrayFromImage(it4)
        sl3 = s3[id4,:,:]
        Y_test = Image.fromarray((sl3/np.max(sl3)*0xff).astype(np.uint8), mode="L")
        X_train = self.transform(X_train)
        Y_train = self.transform(Y_train)
        X_test = self.transform(X_test)
        Y_test = self.transform(Y_test)
        return X_train,Y_train,X_test,Y_test

    def __len__(self):
        return len(self.trainin)
train_data=MyDataset(csv='112.csv', transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)