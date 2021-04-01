from unet import unet
from attention_unet import att_unet
from resnet18_unet import resnet18_unet
from attention_resnet18_unet import att_resnet18_unet
from MyDataset import MyDataset
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


# evaluation maxtrics with no normalized data
def matrics(img_AC, pred):
    mse = nn.MSELoss()(pred, img_AC)
    mae = nn.L1Loss()(pred, img_AC)
    L = pow(2, 18) - 1
    psnr = 10 * log10(pow(L, 2) / mse)  # around.216373 is the biggest possible intensity of img_AC
    ssim = ssim_function(pred, img_AC, L=L)
    return mse, mae, psnr, ssim


def test(loader=None, model=None, device=None, time_stamp=None):
    mse_list = []
    mae_list = []
    psnr_list = []
    ssim_list = []
    model = model.eval()

    for img_NAC, img_AC, name_img_AC in loader:

        img_NAC = img_NAC.float()
        img_NAC = img_NAC.to(device)
        img_AC = img_AC.float()
        img_AC = img_AC.to(device)

        pred = model(img_NAC)
        mse, mae, psnr, ssim = matrics(img_AC, pred)

        name_img_AC = "".join(name_img_AC)
        # not smart method(modified from test function), need to be optimized
        if name_img_AC == '/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl/Patient15/Patient15_AC.gipl140':
            print('mse', mse.item())
            print('mae', mae.item())
            print('psnr', psnr)
            print('ssim', ssim)
            print('difference image comes from', name_img_AC)
            img_NAC = img_NAC.cpu().detach().numpy()
            img_AC = img_AC.cpu().detach().numpy()

            pred = pred.cpu().detach().numpy()
            difference_image = img_AC - pred
            # add bias to solve the negative part of difference_image could not be seen
            a = np.ones(difference_image.shape) * 1000
            difference_image = a + difference_image

            plt.figure(figsize=(21, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(img_NAC[0, 0, 20:-20, 20:-20], cmap='gray')
            plt.title('img_NAC')

            plt.subplot(1, 4, 2)
            plt.imshow(img_AC[0, 0, 20:-20, 20:-20], cmap='gray')
            plt.title('img_AC')

            plt.subplot(1, 4, 3)
            plt.imshow(pred[0, 0, 20:-20, 20:-20], cmap='gray')
            plt.title('Prediction')

            plt.subplot(1, 4, 4)
            plt.imshow(difference_image[0, 0, 20:-20, 20:-20], cmap='jet')
            plt.title('difference_image')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.clim(-1000, 7000)
            plt.show()

            dir_path = os.path.join('/home/WIN-UNI-DUE/sotiling/WS20/difference_images')
            difference_image_path = os.path.join(dir_path, str(time_stamp) + 'difference.tif')
            plt.savefig(difference_image_path, bbox_inches="tight")
            print('difference_image is saved with following path')
            print(difference_image_path)

        mse_list.append(float(mse.item()))
        mae_list.append(float(mae.item()))
        psnr_list.append(float(psnr))
        ssim_list.append(float(ssim))

    epoch_mse = np.sum(mse_list) / len(loader)
    epoch_mae = np.sum(mae_list) / len(loader)
    epoch_psnr = np.sum(psnr_list) / len(loader)
    epoch_ssim = np.sum(ssim_list) / len(loader)

    std_mse = np.std(mse_list)
    std_mae = np.std(mae_list)
    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)

    return epoch_mse, epoch_mae, epoch_psnr, epoch_ssim, std_mse, std_mae, std_psnr, std_ssim


test_list = [15]
data_path = '/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl'
time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

test_dataset = MyDataset(data_path=data_path,
                         patientList=test_list,
                         repeat=1,
                         transform=transforms.Compose([transforms.ToTensor()]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# choose a pre-trained model
model_id = 2
if model_id == 1:
    model = unet()
    model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-01-22_20:30:52.pth'  # unet
elif model_id == 2:
    model = att_unet()
    model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-01-31_21:15:58.pth'  # att unet
elif model_id == 3:
    model = resnet18_unet()
    model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-01-24_13:17:30.pth'  # resnet18
elif model_id == 4:
    model = att_resnet18_unet()
    model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-01-24_22:34:15.pth'  # att resnet18 unet

model.to(device=device)
# loading data
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=2)
print('=' * 5, 'Start of testing', '=' * 5)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
test_mse, test_mae, test_psnr, test_ssim, std_mse, std_mae, std_psnr, std_ssim = test(loader=test_loader,
                                                                                      model=model,
                                                                                      device=device,
                                                                                      time_stamp=time_stamp
                                                                                      )
print('test_MSE ± STD: {:.6f} ±  {:.6f}'.format(test_mse, std_mse))
print('test_MAE ± STD: {:.6f}  ± {:.6f}'.format(test_mae, std_mae))
print('test_PSNR ± STD: {:.6f} ± {:.6f}'.format(test_psnr, std_psnr))
print('test_SSIM ± STD: {:.6f} ± {:.6f}'.format(test_ssim, std_ssim))
print('=' * 5, 'End of testing', '=' * 5)