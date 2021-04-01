from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch
from unet import unet
from attention_unet import att_unet
from resnet18_unet import resnet18_unet
from attention_resnet18_unet import att_resnet18_unet
import cv2
import os
import SimpleITK as sitk
from MyDataset import resize_sitk_2D
from torch.optim.adam import Adam
import torch.nn as nn


def load_data(volume_path, slice_id):
    volume = sitk.ReadImage(volume_path)
    img = volume[:, :, slice_id]
    np_img = sitk.GetArrayFromImage(img)
    resize_img = resize_sitk_2D(np_img, (384, 384))
    transform = transforms.Compose([transforms.ToTensor()])
    tensor_img = transform(resize_img)
    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img


def img_transform(img, device):
    img = img.float()
    img = img.to(device=device)
    return img


def load_model(model, model_path, device):
    model.to(device=device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


# get gradients
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# get feature maps
def forward_hook(module, input, output):
    fmap_block.append(output)


def forward(tensor_img_NAC, tensor_img_AC, model, device):
    tensor_img_NAC = img_transform(tensor_img_NAC, device)
    tensor_img_AC = img_transform(tensor_img_AC, device)
    output = model(tensor_img_NAC)
    return output


def backward(loss_function, model, output, tensor_img_AC, device):
    tensor_img_AC = img_transform(tensor_img_AC, device)
    loss = loss_function(output, tensor_img_AC)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    optimizer.zero_grad()
    loss.backward()  # back propagation
    # optimizer.step() # actually weights are not changed


def gen_cam(feature_map, grads):
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 384x384
    weights = np.mean(grads, axis=(1, 2))  #
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = cv2.resize(cam, (384, 384))
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam


def show_cam_on_image(img_AC, img_NAC, cam, out_dir, time_stamp, model_name, slice_id):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    path_out = os.path.join(out_dir, str(time_stamp) + "_" + "attention_maps" + "_" + str(model_name) + "_" + str(
        slice_id) + ".tif")

    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # heatmap must in uint8
    cam_heatmap = np.float32(cam_heatmap) / 255

    fig = plt.figure(figsize=(26, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(img_NAC, cmap='gray', interpolation='none')
    plt.title('img_NAC')

    plt.subplot(1, 4, 2)
    plt.imshow(img_AC, cmap='gray', interpolation='none')
    plt.title('img_AC')

    plt.subplot(1, 4, 3)
    plt.imshow(cam_heatmap, 'jet', vmin=0, vmax=1, interpolation='none')
    plt.colorbar()
    plt.title('Activation map')

    plt.subplot(1, 4, 4)
    plt.imshow(img_NAC, 'gray', interpolation='none')
    plt.imshow(cam_heatmap, 'jet', vmin=0, vmax=1, interpolation='none', alpha=0.5)
    plt.colorbar()
    plt.title('Activation map and img_NAC')
    fig.savefig(path_out, bbox_inches='tight')


def store_cam(tensor_img_AC, tensor_img_NAC, output_dir, time_stamp, model_name, slice_id):
    grads_val = grad_block[0].cpu().data.numpy().squeeze(0)
    fmap = fmap_block[0].cpu().data.numpy().squeeze(0)
    cam = gen_cam(fmap, grads_val)
    cam = np.float32(resize_sitk_2D(cam, (344, 344)))
    tensor_img_NAC = tensor_img_NAC.cpu().data.numpy().squeeze()
    tensor_img_NAC = np.float32(resize_sitk_2D(tensor_img_NAC, (344, 344)))
    tensor_img_AC = tensor_img_AC.cpu().data.numpy().squeeze()
    tensor_img_AC = np.float32(resize_sitk_2D(tensor_img_AC, (344, 344)))
    show_cam_on_image(tensor_img_AC, tensor_img_NAC, cam, output_dir, time_stamp, model_name, slice_id)
    print('Activation Map of', str(model_name), 'with slice_id', str(slice_id), 'is saved!')


if __name__ == '__main__':

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = '/home/WIN-UNI-DUE/sotiling/WS20/activation_maps'
    patient_list = ['Patient15']
    data_path = '/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl'
    slice_list = [140]

    for i, slice_id in enumerate(slice_list):
        path_NAC = os.path.join(data_path, patient_list[0], patient_list[0] + "_NAC.gipl")
        tensor_img_NAC = load_data(path_NAC, slice_id)

        path_AC = os.path.join(data_path, patient_list[0], patient_list[0] + "_AC.gipl")
        tensor_img_AC = load_data(path_AC, slice_id)

        grad_block = []
        fmap_block = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_function = nn.MSELoss()

        model_id = 2
        if model_id == 1:
            model = unet()
            model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-01-22_20:30:52.pth'
            load_model(model, model_path, device)

            # for fair comparison
            model.Up_conv2.register_forward_hook(forward_hook)
            model.Up_conv2.register_backward_hook(backward_hook)
            output = forward(tensor_img_NAC, tensor_img_AC, model, device)
            backward(loss_function, model, output, tensor_img_AC, device)
            store_cam(tensor_img_AC, tensor_img_NAC, output_dir, time_stamp, 'att_unet', slice_id)

            # for self check
            # block = model.Conv5.conv
            # index = ['3']
            # for name, module in block._modules.items():
            #     if name in index:
            #         x = module
            #         x.register_forward_hook(forward_hook)
            #         x.register_backward_hook(backward_hook)
            #         output = forward(tensor_img_NAC,tensor_img_AC,model,device)
            #         backward(loss_function,model,output,tensor_img_AC,device)
            #         store_cam(tensor_img_NAC,output_dir,time_stamp,'unet',slice_id)

        elif model_id == 2:
            model = att_unet()
            model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-01-31_21:15:58.pth'
            load_model(model, model_path, device)

            # for fair comparison
            model.Up_conv2d_relu2.register_forward_hook(forward_hook)
            model.Up_conv2d_relu2.register_backward_hook(backward_hook)
            output = forward(tensor_img_NAC, tensor_img_AC, model, device)
            backward(loss_function, model, output, tensor_img_AC, device)
            store_cam(tensor_img_AC, tensor_img_NAC, output_dir, time_stamp, 'att_unet', slice_id)

            # for self check
            # block = model.Conv5.conv
            # block = model.Up_conv5.conv
            # block = model.Up_conv4.conv
            # block = model.Up_conv3.conv
            # index = ['3']
            # for name, module in block._modules.items():
            #     if name in index:
            #         x = module
            #         x.register_forward_hook(forward_hook)
            #         x.register_backward_hook(backward_hook)
            #         output = forward(tensor_img_NAC,tensor_img_AC,model,device)
            #         backward(loss_function,model,output,tensor_img_AC,device)
            #         store_cam(tensor_img_NAC,output_dir,time_stamp,'attention_unet',slice_id)

        elif model_id == 3:
            model = resnet18_unet()
            model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-01-24_13:17:30.pth'
            load_model(model, model_path, device)
            # for fair comparison
            model.decode0.register_forward_hook(forward_hook)
            model.decode0.register_backward_hook(backward_hook)
            output = forward(tensor_img_NAC, tensor_img_AC, model, device)
            backward(loss_function, model, output, tensor_img_AC, device)
            store_cam(tensor_img_AC, tensor_img_NAC, output_dir, time_stamp, 'resnet18_unet', slice_id)

            # for self check
            # block = model.decode0.conv
            # index = ['3']
            # for name, module in block._modules.items():
            #     if name in index:
            #         x = module
            #         x.register_forward_hook(forward_hook)
            #         x.register_backward_hook(backward_hook)
            #         output = forward(tensor_img_NAC,tensor_img_AC,model,device)
            #         backward(loss_function,model,output,tensor_img_AC,device)
            #         store_cam(tensor_img_AC,tensor_img_NAC,output_dir,time_stamp,'resnet18_unet',slice_id)

        elif model_id == 4:
            model = att_resnet18_unet()
            model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-01-24_22:34:15.pth'
            load_model(model, model_path, device)

            # for fair comparison
            model.Up_conv2.register_forward_hook(forward_hook)
            model.Up_conv2.register_backward_hook(backward_hook)
            output = forward(tensor_img_NAC, tensor_img_AC, model, device)
            backward(loss_function, model, output, tensor_img_AC, device)
            store_cam(tensor_img_AC, tensor_img_NAC, output_dir, time_stamp, 'att_resnet18_unet', slice_id)

            # for self check
            # block = model.Up_conv2.conv
            # index = ['3']
            # for name, module in block._modules.items():
            #     if name in index:
            #         x = module
            #         x.register_forward_hook(forward_hook)
            #         x.register_backward_hook(backward_hook)
            #         output = forward(tensor_img_NAC,tensor_img_AC,model,device)
            #         backward(loss_function,model,output,tensor_img_AC,device)
            #         store_cam(tensor_img_AC,tensor_img_NAC,output_dir,time_stamp,'attention_resnet18_unet',slice_id)


