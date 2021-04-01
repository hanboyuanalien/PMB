from modified_attention_unet_1 import modified_att_unet_1
from modified_attention_unet_2 import modified_att_unet_2
from modified_attention_unet_3 import modified_att_unet_3
from modified_attention_unet_4 import modified_att_unet_4
from modified_attention_unet_5 import modified_att_unet_5
from MyDataset import MyDataset # self-defined Dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
from math import log10
from util import ssim_function
import numpy as np
from grad_cam_classifier import get_grad_cam 
from MyDataset import resize_sitk_2D

# evaluation maxtrics with no normalized data
def matrics(img_AC,pred):
    mse = nn.MSELoss()(pred,img_AC) 
    mae = nn.L1Loss()(pred,img_AC)
    L = pow(2,18)-1
    psnr = 10 * log10(pow(L,2)/mse) # ca.216373 is the biggest possible intensity of img_AC
    ssim = ssim_function(pred,img_AC,L=L)
    return mse,mae,psnr,ssim

# resize cam for replacing gating signal of modified_att_unet models, get specific pred
def process_cam(cam,model_id,model,img_NAC,device):
    if model_id == 1: # all AGs 
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
    elif model_id == 2: # AG4
        cam = np.float32(resize_sitk_2D(cam, (384, 384)))
        cam = np.expand_dims(np.expand_dims(cam,axis=0),axis=0)
        cam = torch.from_numpy(cam).to(device)  
        pred = model(img_NAC,cam)
        return pred
    elif model_id == 3: # AG1
        cam = np.float32(resize_sitk_2D(cam, (48, 48)))
        cam = np.expand_dims(np.expand_dims(cam,axis=0),axis=0)
        cam = torch.from_numpy(cam).to(device)  
        pred = model(img_NAC,cam)
        return pred
    elif model_id == 4: # AG1 and 2
        cam1 = np.float32(resize_sitk_2D(cam, (48, 48)))
        cam2 = np.float32(resize_sitk_2D(cam, (96, 96)))

        cam1 = np.expand_dims(np.expand_dims(cam1,axis=0),axis=0)
        cam2 = np.expand_dims(np.expand_dims(cam2,axis=0),axis=0)

        cam1 = torch.from_numpy(cam1).to(device)       
        cam2 = torch.from_numpy(cam2).to(device)

        pred = model(img_NAC,cam1,cam2)
        return pred
    elif model_id == 5: # AG 1,2,3
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
    

def test(loader=None,model=None,device=None,time_stamp=None,model_classifier=None,model_id=None):
    mse_list = []
    mae_list = []
    psnr_list = []
    ssim_list = []
    model = model.eval() 

    for img_NAC,img_AC,name_img_AC in loader:

        img_NAC = img_NAC.float()
        img_NAC = img_NAC.to(device)
        img_AC = img_AC.float()
        img_AC = img_AC.to(device)

        cam = get_grad_cam(model_classifier,img_NAC)
        pred = process_cam(cam,model_id,model,img_NAC,device)
        mse,mae,psnr,ssim = matrics(img_AC,pred)

        name_img_AC = "".join(name_img_AC)
        if name_img_AC == '/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl/Patient15/Patient15_AC.gipl140': # MyDataset
            print('mse',mse.item())
            print('mae',mae.item())
            print('psnr',psnr)
            print('ssim',ssim)
            print('difference image comes from',name_img_AC) 
            img_NAC = img_NAC.cpu().detach().numpy()
            img_AC = img_AC.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            difference_image = img_AC - pred
            a = np.ones(difference_image.shape)*1000
            difference_image = a + difference_image
            # difference_image = difference_image.cpu().detach().numpy()

            plt.figure(figsize=(20,12))
            plt.subplot(1,4,1)
            plt.imshow(img_NAC[0,0,20:-20,20:-20],cmap='gray')
            plt.title('img_NAC')

            plt.subplot(1,4,2)
            plt.imshow(img_AC[0,0,20:-20,20:-20],cmap='gray')
            plt.title('img_AC')

            plt.subplot(1,4,3)
            plt.imshow(pred[0,0,20:-20,20:-20],cmap='gray')
            plt.title('Prediction')

            plt.subplot(1,4,4)
            plt.imshow(difference_image[0,0,20:-20,20:-20],cmap ='jet')
            plt.title('difference_image')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.clim(-1000,7000)
            plt.show()

            dir_path = os.path.join('/home/WIN-UNI-DUE/sotiling/WS20/difference_images')
            difference_image_path = os.path.join(dir_path,str(time_stamp)+'difference.tif')
            plt.savefig(difference_image_path,bbox_inches="tight")
            print('difference_image is saved with following path')
            print(difference_image_path)
            
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
    
    return epoch_mse,epoch_mae,epoch_psnr,epoch_ssim,std_mse,std_mae,std_psnr,std_ssim


test_list = [15] 
data_path ='/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl'
time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

test_dataset = MyDataset(data_path= data_path,
                        patientList=test_list,
                        repeat=1,
                        transform=transforms.Compose([transforms.ToTensor()]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pools of models
model_id = 2
if model_id ==1:
    model = modified_att_unet_1()
    model_path = '/home/WIN-UNI-aDUE/sotiling/WS20/checkpoint/best_model_2021-02-02_20:54:23.pth' 
elif model_id ==2:
    model = modified_att_unet_2() 
    model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-02-03_11:03:39.pth' 
elif model_id ==3:
    model = modified_att_unet_3()
    model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-02-09_01:20:22.pth'
elif model_id ==4: # replace AG1&2
    model = modified_att_unet_4()
    model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-02-09_01:21:09.pth'
elif model_id ==5:
    model = modified_att_unet_5()
    model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-02-09_01:24:07.pth'

model.to(device=device)

# loading data
test_loader = DataLoader(test_dataset, shuffle=False,batch_size=1, num_workers=2)
model_classifier = torch.load('/home/WIN-UNI-DUE/sotiling/WS20/pmb_ws20_dl-pet-ac-attention/pre_trained_grad_cam_classifier/grad_cam_new/classification_rec_complete_neu3', map_location=torch.device('cuda'))
model_classifier = model_classifier.float()

print('='*5,'Start of testing','='*5)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
test_mse,test_mae,test_psnr,test_ssim,std_mse,std_mae,std_psnr,std_ssim= test(loader=test_loader,
                                                                                model=model,
                                                                                device=device,
                                                                                time_stamp = time_stamp,
                                                                                model_classifier=model_classifier,
                                                                                model_id = model_id
                                                                                )
print('test_MSE ± STD: {:.6f} ±  {:.6f}'.format(test_mse,std_mse))
print('test_MAE ± STD: {:.6f}  ± {:.6f}'.format(test_mae,std_mae))
print('test_PSNR ± STD: {:.6f} ± {:.6f}'.format(test_psnr,std_psnr)) 
print('test_SSIM ± STD: {:.6f} ± {:.6f}'.format(test_ssim,std_ssim))
print('='*5,'End of testing','='*5)
