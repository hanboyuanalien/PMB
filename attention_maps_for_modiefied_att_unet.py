from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch
from modified_attention_unet_1 import modified_att_unet_1
from modified_attention_unet_2 import modified_att_unet_2
from modified_attention_unet_3 import modified_att_unet_3
from modified_attention_unet_4 import modified_att_unet_4
from modified_attention_unet_5 import modified_att_unet_5
import cv2
import os
import SimpleITK as sitk
from MyDataset import resize_sitk_2D
from grad_cam_classifier import get_grad_cam # prepared by Marie
from torch.optim.adam import Adam
import torch.nn as nn

class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.model_features = model.features
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)
    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model_features._modules.items():
            print(x.shape)
            
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        x = x.view(x.size(0), -1) 
        x = self.model.classifier(x)
        return outputs, x
        
def load_data(volume_path,slice_id):
        volume = sitk.ReadImage(volume_path)
        img = volume[:,:,slice_id]
        np_img = sitk.GetArrayFromImage(img)
        resize_img = resize_sitk_2D(np_img, (384, 384))
        transform = transforms.Compose([transforms.ToTensor()])
        tensor_img = transform(resize_img)
        tensor_img = tensor_img.unsqueeze(0)
        return tensor_img

def img_transform(img,device):
    img = img.float()
    img = img.to(device=device)
    return img

def load_model(model,model_path,device):
    model.to(device=device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

# get gradients 
def backward_hook(module,grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# get feature maps
def forward_hook(module,input,output):
    fmap_block.append(output)

# resize cam for replacing gating signal of modified_att_unet models, get specific pred
def process_cam(cam,model_id,model,img_NAC,device):
    if model_id == 1:
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
    elif model_id == 2:
        cam = np.float32(resize_sitk_2D(cam, (384, 384)))
        cam = np.expand_dims(np.expand_dims(cam,axis=0),axis=0)
        cam = torch.from_numpy(cam).to(device)  
        pred = model(img_NAC,cam)
        return pred
    elif model_id == 3:
        cam = np.float32(resize_sitk_2D(cam, (48, 48)))
        cam = np.expand_dims(np.expand_dims(cam,axis=0),axis=0)
        cam = torch.from_numpy(cam).to(device)  
        pred = model(img_NAC,cam)
        return pred
    elif model_id == 4:
        cam1 = np.float32(resize_sitk_2D(cam, (48, 48)))
        cam2 = np.float32(resize_sitk_2D(cam, (96, 96)))

        cam1 = np.expand_dims(np.expand_dims(cam1,axis=0),axis=0)
        cam2 = np.expand_dims(np.expand_dims(cam2,axis=0),axis=0)

        cam1 = torch.from_numpy(cam1).to(device)       
        cam2 = torch.from_numpy(cam2).to(device)

        pred = model(img_NAC,cam1,cam2)
        return pred
    elif model_id == 5:
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

def forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id):
    tensor_img_NAC = img_transform(tensor_img_NAC,device)
    tensor_img_AC = img_transform(tensor_img_AC,device)
    cam = get_grad_cam(model_classifier,tensor_img_NAC)
    pred = process_cam(cam,model_id,model,tensor_img_NAC,device)
    return pred

def backward(loss_function,model,output,tensor_img_AC,device):
    tensor_img_AC = img_transform(tensor_img_AC,device)
    loss = loss_function(output,tensor_img_AC) 
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    optimizer.zero_grad()
    loss.backward() # back propagation
    # optimizer.step() # actually weights are not changed

def gen_cam(feature_map,grads):
    cam = np.zeros(feature_map.shape[1:],dtype=np.float32) # 384x384
    weights = np.mean(grads, axis=(1, 2))  #
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = cv2.resize(cam, (384, 384))
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def show_cam_on_image(img_AC,img_NAC, cam, out_dir,time_stamp,model_name,slice_id):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir) 
    path_out = os.path.join(out_dir, str(time_stamp)+"_"+"attention_maps"+"_"+str(model_name)+"_"+str(slice_id)+".tif")

    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET) # do heatmap must with uint8
    cam_heatmap = np.float32(cam_heatmap)/255

    # img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # cam_img_heatmap = 0.3*cam_heatmap + 0.7*np.float32(img_rgb)/255
    # cam_img_heatmap = cam_img_heatmap / np.max(cam_img_heatmap)

    fig = plt.figure(figsize=(28,5)) 
    plt.subplot(1,4,1)
    plt.imshow(img_NAC,cmap='gray',interpolation='none') 
    plt.title('img_NAC')

    plt.subplot(1,4,2)
    plt.imshow(img_AC,cmap='gray',interpolation='none') 
    plt.title('img_AC')

    plt.subplot(1,4,3)
    plt.imshow(cam_heatmap, 'jet',vmin = 0, vmax = 1, interpolation='none')
    plt.colorbar()
    plt.title('Attention map')

    plt.subplot(1,4,4)
    plt.imshow(img_NAC, 'gray', interpolation='none')
    plt.imshow(cam_heatmap, 'jet',vmin = 0, vmax = 1, interpolation='none', alpha=0.5)
    plt.colorbar()
    plt.title('Attention map and img_NAC')
    # plt.show()
    fig.savefig(path_out, bbox_inches='tight')

def store_cam(tensor_img_AC,tensor_img_NAC,output_dir,time_stamp,model_name,slice_id):
    grads_val = grad_block[0].cpu().data.numpy().squeeze(0)
    fmap = fmap_block[0].cpu().data.numpy().squeeze(0)
    cam = gen_cam(fmap, grads_val)
    cam = np.float32(resize_sitk_2D(cam, (344, 344)))
    tensor_img_NAC = tensor_img_NAC.cpu().data.numpy().squeeze()
    tensor_img_NAC = np.float32(resize_sitk_2D(tensor_img_NAC, (344, 344)))
    tensor_img_AC = tensor_img_AC.cpu().data.numpy().squeeze()
    tensor_img_AC = np.float32(resize_sitk_2D(tensor_img_AC, (344, 344)))
    show_cam_on_image(tensor_img_AC,tensor_img_NAC, cam, output_dir,time_stamp,model_name,slice_id)
    print('Attention Map of',str(model_name),'with slice_id',str(slice_id),'is saved!')

if __name__ == '__main__':

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = '/home/WIN-UNI-DUE/sotiling/WS20/activation_maps'
    patient_list = ['Patient15'] 
    data_path ='/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl'
    slice_list = [140]

    model_classifier = torch.load('/home/WIN-UNI-DUE/sotiling/WS20/pmb_ws20_dl-pet-ac-attention/pre_trained_grad_cam_classifier/grad_cam_new/classification_rec_complete_neu3', map_location=torch.device('cuda'))
    model_classifier = model_classifier.float()
    
    for i, slice_id in enumerate(slice_list):
        path_NAC = os.path.join(data_path,patient_list[0],patient_list[0]+"_NAC.gipl")
        tensor_img_NAC = load_data(path_NAC,slice_id)

        path_AC = os.path.join(data_path,patient_list[0],patient_list[0]+"_AC.gipl")
        tensor_img_AC = load_data(path_AC,slice_id)

        grad_block = []
        fmap_block = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_function = nn.MSELoss()

        model_id =1
        if model_id == 1: # all AGs
            model = modified_att_unet_1()
            model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-02-02_20:54:23.pth'
            load_model(model,model_path,device)
            model.Up_conv2d_relu2.register_forward_hook(forward_hook)
            model.Up_conv2d_relu2.register_backward_hook(backward_hook)
            output = forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id)
            backward(loss_function,model,output,tensor_img_AC,device)
            store_cam(tensor_img_AC,tensor_img_NAC,output_dir,time_stamp,'modified_att_unet_1',slice_id)
        elif model_id == 2: # AG4
            model = modified_att_unet_2()
            model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-02-03_11:03:39.pth'
            load_model(model,model_path,device)
            model.Up_conv2d_relu2.register_forward_hook(forward_hook)
            model.Up_conv2d_relu2.register_backward_hook(backward_hook)
            output = forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id)
            backward(loss_function,model,output,tensor_img_AC,device)
            store_cam(tensor_img_AC,tensor_img_NAC,output_dir,time_stamp,'modified_att_unet_2',slice_id)
        elif model_id == 3: # AG1
            model = modified_att_unet_3()
            model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-02-09_01:20:22.pth'
            load_model(model,model_path,device) 
            model.Up_conv2d_relu2.register_forward_hook(forward_hook)
            model.Up_conv2d_relu2.register_backward_hook(backward_hook)
            output = forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id)
            backward(loss_function,model,output,tensor_img_AC,device)
            store_cam(tensor_img_AC,tensor_img_NAC,output_dir,time_stamp,'modified_att_unet_3',slice_id)
        elif model_id == 4: # AG1,2
            model = modified_att_unet_4()
            model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-02-09_01:21:09.pth'
            load_model(model,model_path,device) 
            model.Up_conv2d_relu2.register_forward_hook(forward_hook)
            model.Up_conv2d_relu2.register_backward_hook(backward_hook)
            output = forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id)
            backward(loss_function,model,output,tensor_img_AC,device)
            store_cam(tensor_img_AC,tensor_img_NAC,output_dir,time_stamp,'modified_att_unet_4',slice_id)
        elif model_id == 5: # AG1,2,3
            model = modified_att_unet_5()
            model_path = '/home/WIN-UNI-DUE/sotiling/WS20/checkpoint/best_model_2021-02-09_01:24:07.pth'
            load_model(model,model_path,device) 
            model.Up_conv2d_relu2.register_forward_hook(forward_hook)
            model.Up_conv2d_relu2.register_backward_hook(backward_hook)
            output = forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id)
            backward(loss_function,model,output,tensor_img_AC,device)
            store_cam(tensor_img_AC,tensor_img_NAC,output_dir,time_stamp,'modified_att_unet_5',slice_id)
        # elif model_id == 6:
        #     model = modified_att_unet_6()
        #     #TODO:
        #     model_path = ''
        #     load_model(model,model_path,device) 
        #     model.Up_conv2d_relu2.register_forward_hook(forward_hook)
        #     model.Up_conv2d_relu2.register_backward_hook(backward_hook)
        #     output = forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id)
        #     backward(loss_function,model,output,tensor_img_AC,device)
        #     store_cam(tensor_img_NAC,output_dir,time_stamp,'modified_att_unet_6',slice_id)
        # elif model_id == 7:
        #     model = modified_att_unet_7()
        #     #TODO:
        #     model_path = ''
        #     load_model(model,model_path,device) 
        #     model.Up_conv2d_relu.register_forward_hook(forward_hook)
        #     model.Up_conv2d_relu.register_backward_hook(backward_hook)
        #     output = forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id)
        #     backward(loss_function,model,output,tensor_img_AC,device)
        #     store_cam(tensor_img_NAC,output_dir,time_stamp,'modified_att_unet_7',slice_id)
        # elif model_id == 8:
        #     model = modified_att_unet_8()
        #     #TODO:
        #     model_path = ''
        #     load_model(model,model_path,device) 
        #     model.Up_conv2d_relu.register_forward_hook(forward_hook)
        #     model.Up_conv2d_relu.register_backward_hook(backward_hook)
        #     output = forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id)
        #     backward(loss_function,model,output,tensor_img_AC,device)
        #     store_cam(tensor_img_NAC,output_dir,time_stamp,'modified_att_unet_8',slice_id)
        # elif model_id == 9:
        #     model = modified_att_unet_6()
        #     #TODO:
        #     model_path = ''
        #     load_model(model,model_path,device) 
        #     model.Up_conv2d_relu.register_forward_hook(forward_hook)
        #     model.Up_conv2d_relu.register_backward_hook(backward_hook)
        #     output = forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id)
        #     backward(loss_function,model,output,tensor_img_AC,device)
        #     store_cam(tensor_img_NAC,output_dir,time_stamp,'modified_att_unet_9',slice_id)
        # elif model_id == 10:
        #     model = modified_att_unet_10()
        #     #TODO:
        #     model_path = ''
        #     load_model(model,model_path,device) 
        #     model.Up_conv2d_relu.register_forward_hook(forward_hook)
        #     model.Up_conv2d_relu.register_backward_hook(backward_hook)
        #     output = forward(tensor_img_NAC,tensor_img_AC,model,device,model_classifier,model_id)
        #     backward(loss_function,model,output,tensor_img_AC,device)
        #     store_cam(tensor_img_NAC,output_dir,time_stamp,'modified_att_unet_10',slice_id)