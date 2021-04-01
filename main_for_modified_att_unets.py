from SimpleITK.SimpleITK import TranslationTransform
from numpy.core.fromnumeric import mean
from modified_attention_unet_1 import modified_att_unet_1
from modified_attention_unet_2 import modified_att_unet_2
from modified_attention_unet_3 import modified_att_unet_3
from modified_attention_unet_4 import modified_att_unet_4
from modified_attention_unet_5 import modified_att_unet_5
from MyDataset import MyDataset 
from util import ssim_function
from torch.nn.modules import loss
from torch.optim.adam import Adam
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from math import log10
import os
import datetime
import time
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import SimpleITK as sitk
from MyDataset import resize_sitk_2D
import cv2
from grad_cam_classifier import get_grad_cam # prepared by Marie

# Training settings
def parser():
    parser = argparse.ArgumentParser(description='Please setting training parameters')
    parser.add_argument('--dataset', dest='data_path', default='/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl', help='dataset path')
    parser.add_argument('--batch_size', dest='batch_size', default=1, type=int, help='batch size for one iteration, default: 2')
    parser.add_argument('--epochs', dest='epochs', default=20, type=int, help='number of epochs, default: 10')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='value of learning rate, default: 0.001')
    parser.add_argument('--loss_function', dest='loss_function', default=nn.MSELoss(), help='type of loss function, default: MSELoss')
    parser.add_argument('--num_workers', dest='num_workers', default=2, type=int, help='value of numbers')
    parser.add_argument('--train', dest='train', default=True, help='do training')
    parser.add_argument('--checkpoint_file', dest='checkpoint_file', default=None, help='input the path of the model') # add address of checkpoint file.
    opt = parser.parse_args()
    args = vars(parser.parse_args())
    print(args)
    return opt

# evaluation maxtrics 
def matrics(img_AC,pred):
    mse = nn.MSELoss()(pred,img_AC) 
    mae = nn.L1Loss()(pred,img_AC)
    L = pow(2,18)-1
    psnr = 10 * log10(pow(L,2)/mse) # around.216373 is the biggest possible intensity of img_AC
    ssim = ssim_function(pred,img_AC,L=L)
    return mse,mae,psnr,ssim

# resize cam for replacing gating signal of modified_att_unet models, get specific pred
def process_cam(cam,model_id,model,img_NAC,device):
    if model_id == 1: # gating signals of all attention gates of att_unet were replaced by Grad-CAM
        cam1 = np.float32(resize_sitk_2D(cam, (48, 48)))
        cam2 = np.float32(resize_sitk_2D(cam, (96, 96)))
        cam3 = np.float32(resize_sitk_2D(cam, (192, 192)))
        cam4 = np.float32(resize_sitk_2D(cam, (384, 384)))

        cam1 = np.expand_dims(np.expand_dims(cam1,axis=0),axis=0)
        cam2 = np.expand_dims(np.expand_dims(cam2,axis=0),axis=0)
        cam3 = np.expand_dims(np.expand_dims(cam3,axis=0),axis=0)
        cam4 = np.expand_dims(np.expand_dims(cam4,axis=0),axis=0)

        cam1 = torch.from_numpy(cam1).to(device)       
        cam2 = torch.from_numpy(cam2).to(device)
        cam3 = torch.from_numpy(cam3).to(device)
        cam4 = torch.from_numpy(cam4).to(device)

        pred = model(img_NAC,cam1,cam2,cam3,cam4)
        return pred
    elif model_id == 2: # gating signal of 4th attention gate of att_unet was replaced by Grad-CAM
        cam = np.float32(resize_sitk_2D(cam, (384, 384)))
        cam = np.expand_dims(np.expand_dims(cam,axis=0),axis=0)
        cam = torch.from_numpy(cam).to(device)  
        pred = model(img_NAC,cam)
        return pred
    elif model_id == 3: # gating signal of 1st attention gate of att_unet was replaced by Grad-CAM
        cam = np.float32(resize_sitk_2D(cam, (48, 48)))
        cam = np.expand_dims(np.expand_dims(cam,axis=0),axis=0)
        cam = torch.from_numpy(cam).to(device)  
        pred = model(img_NAC,cam)
        return pred
    elif model_id == 4: # gating signals of 1st and 2nd attention gates of att_unet were replaced by Grad-CAM
        cam1 = np.float32(resize_sitk_2D(cam, (48, 48)))
        cam2 = np.float32(resize_sitk_2D(cam, (96, 96)))

        cam1 = np.expand_dims(np.expand_dims(cam1,axis=0),axis=0)
        cam2 = np.expand_dims(np.expand_dims(cam2,axis=0),axis=0)

        cam1 = torch.from_numpy(cam1).to(device)       
        cam2 = torch.from_numpy(cam2).to(device)

        pred = model(img_NAC,cam1,cam2)
        return pred
    elif model_id == 5: # gating signals of 1st, 2nd and 3rd attention gates of att_unet were replaced by Grad-CAM
        cam1 = np.float32(resize_sitk_2D(cam, (48, 48)))
        cam2 = np.float32(resize_sitk_2D(cam, (96, 96)))
        cam3 = np.float32(resize_sitk_2D(cam, (192, 192)))

        cam1 = np.expand_dims(np.expand_dims(cam1,axis=0),axis=0)
        cam2 = np.expand_dims(np.expand_dims(cam2,axis=0),axis=0)
        cam3 = np.expand_dims(np.expand_dims(cam3,axis=0),axis=0)

        cam1 = torch.from_numpy(cam1).to(device)       
        cam2 = torch.from_numpy(cam2).to(device)
        cam3 = torch.from_numpy(cam3).to(device)

        pred = model(img_NAC,cam1,cam2,cam3)
        return pred

# function for training and evaluation 
def train_val(loader=None,model=None,loss_function=None,optimizer=None,train_enable=None,device=None,model_classifier=None,model_id=None):
    sum_loss = 0.0
    sum_mse = 0.0
    sum_mae = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0

    if train_enable == 'True':
        model = model.train()
    else:
        model = model.eval() # default closing Dropout

    for img_NAC, img_AC,_ in loader:

        img_NAC = img_NAC.float()
        img_NAC = img_NAC.to(device)
        img_AC = img_AC.float()
        img_AC = img_AC.to(device)

        cam = get_grad_cam(model_classifier,img_NAC)
        pred = process_cam(cam,model_id,model,img_NAC,device)

        loss = loss_function(pred,img_AC) # Loss is just MSE
        mse,mae,psnr,ssim = matrics(img_AC,pred)
            
        if train_enable == 'True':
            optimizer.zero_grad()
            loss.backward() # back propagation
            optimizer.step()

        sum_loss += float(loss.item())
        sum_mse += float(mse.item())
        sum_mae += float(mae.item())
        sum_psnr += float(psnr)
        sum_ssim += float(ssim)

    epoch_loss = sum_loss/len(loader) 
    epoch_mse = sum_mse/len(loader)
    epoch_mae = sum_mae/len(loader)
    epoch_psnr = sum_psnr/len(loader)
    epoch_ssim = sum_ssim/len(loader)

    return epoch_loss,epoch_mse,epoch_mae,epoch_psnr,epoch_ssim

# function for testing
def test(loader=None,model=None,device=None,model_classifier=None,model_id=None):
    mse_list = []
    mae_list = []
    psnr_list = []
    ssim_list = []
    model = model.eval() 

    for img_NAC,img_AC,_ in loader:

        img_NAC = img_NAC.float()
        img_NAC = img_NAC.to(device)
        img_AC = img_AC.float()
        img_AC = img_AC.to(device)

        cam = get_grad_cam(model_classifier,img_NAC)
        pred = process_cam(cam,model_id,model,img_NAC,device)
        mse,mae,psnr,ssim = matrics(img_AC,pred)
            
        mse_list.append(float(mse.item()))
        mae_list.append(float(mae.item()))
        psnr_list.append(float(psnr))
        ssim_list.append(float(ssim))
    
    epoch_mse = np.sum(mse_list)/len(loader)
    epoch_mae = np.sum(mae_list)/len(loader)
    epoch_psnr = np.sum(psnr_list)/len(loader)
    epoch_ssim = np.sum(ssim_list)/len(loader)

    std_mse = np.std(mse_list)
    std_mae = np.std(mae_list)
    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)
    
    return epoch_mse,epoch_mae,epoch_psnr,epoch_ssim, std_mse,std_mae,std_psnr,std_ssim

def main():
    opt = parser()
    
    # complete dataset with fixed split method
    patient_list = [1,2,5,6,7,9,10,11,12,13,14,15] 
    train_list = [1,2,5,6,7,9,10,11]
    val_list = [12,13]
    test_list = [14,15] 

    train_dataset = MyDataset(data_path=opt.data_path, 
                            patientList =train_list, 
                            repeat=1,
                            transform=transforms.Compose([transforms.ToTensor()]))

    val_dataset = MyDataset(data_path=opt.data_path, 
                            patientList=val_list,
                            repeat=1,
                            transform=transforms.Compose([transforms.ToTensor()]))

    test_dataset = MyDataset(data_path= opt.data_path,
                            patientList=test_list,
                            repeat=1,
                            transform=transforms.Compose([transforms.ToTensor()]))

    # choose a model from modified att_unet models

    # replace all gating signals
    model = modified_att_unet_1() 
    model_id = 1

    # only replace gating signal of AG4
    # model = modified_att_unet_2() 
    # model_id =2

    # only replace gating signal of AG1
    # model = modified_att_unet_3() 
    # model_id = 3

    # only replace gating signals of AG 1 and 2
    # model = modified_att_unet_4() 
    # model_id = 4

    # only replace gating signals of AG 1 and 2 and 3
    # model = modified_att_unet_5() 
    # model_id = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-8) 
    
    # loading data
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_dataset, shuffle=False,batch_size=1, num_workers=opt.num_workers)

    model_classifier = torch.load('/home/WIN-UNI-DUE/sotiling/WS20/pmb_ws20_dl-pet-ac-attention/pre_trained_grad_cam_classifier/grad_cam_new/classification_rec_complete_neu3', map_location=torch.device('cuda'))
    model_classifier = model_classifier.float()

    checkpoint_file = opt.checkpoint_file
    if opt.train:
        if checkpoint_file == None:
            # timestamp
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            dir_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint'
            checkpoint_file = os.path.join(dir_path,'best_model_{}.pth'.format(time_stamp))

        print('='*5,'start of training','='*5)

        for epoch in range(opt.epochs):        
            train_loss,train_mse, train_mae,train_psnr,train_ssim = train_val(loader=train_loader,
                                                                            model=model,
                                                                            loss_function=opt.loss_function, 
                                                                            optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-8),
                                                                            train_enable='True',
                                                                            device=device,
                                                                            model_classifier = model_classifier,
                                                                            model_id = model_id)
            _,val_mse,val_mae,val_psnr,val_ssim = train_val(loader=val_loader,
                                                            model=model,
                                                            loss_function=opt.loss_function,
                                                            train_enable='False', 
                                                            device=device,
                                                            model_classifier = model_classifier,
                                                            model_id = model_id)
            min_loss = 10000000
            
            if train_loss < min_loss:
                min_loss = train_loss
                checkpoint = {
                            'epoch': epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict()
                }
                torch.save(checkpoint,checkpoint_file)
                print('checkpoint is saved in path',checkpoint_file)

            print('epoch:{}'.format(epoch+1))
            print('train_Loss: {:.6f} '.format(train_loss))
            print('train_MSE: {:.6f} val_MSE: {:.6f}'.format(train_mse, val_mse))
            print('train_MAE: {:.6f} val_MAE: {:.6f}'.format(train_mae, val_mae))
            print('train_PSNR: {:.6f} val_PSNR: {:.6f}'.format(train_psnr, val_psnr))
            print('train_SSIM: {:.6f} val_SSIM: {:.6f}'.format(train_ssim, val_ssim))

        print('='*5,'End of training ','='*5)
        print('name of saved best model',checkpoint_file)

    print('='*5,'Start of testing','='*5)
    if checkpoint_file == None:
        print('error! no model exists')
        return
    else:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint(['epoch'])

    test_mse,test_mae,test_psnr,test_ssim,std_mse,std_mae,std_psnr,std_ssim = test(loader=test_loader,
                                                                                    model=model,
                                                                                    device=device,
                                                                                    model_classifier = model_classifier,
                                                                                    model_id = model_id)
    print('test_MSE ± STD: {:.6f} ±  {:.6f}'.format(test_mse,std_mse))
    print('test_MAE ± STD: {:.6f}  ± {:.6f}'.format(test_mae,std_mae))
    print('test_PSNR ± STD: {:.6f} ± {:.6f}'.format(test_psnr,std_psnr))
    print('test_SSIM ± STD: {:.6f} ± {:.6f}'.format(test_ssim,std_ssim))
    print('='*5,'End of testing','='*5)

if __name__ == '__main__':
    starttime = time.time()
    main()
    endtime = time.time()
    print ('Time spent on this run',endtime-starttime,'seconds')