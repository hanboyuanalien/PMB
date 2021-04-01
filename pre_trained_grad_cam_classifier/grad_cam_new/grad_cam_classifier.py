# In[0]
from SimpleITK.SimpleITK import ShapeDetectionLevelSet
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
# import cv2
import numpy as np
import os
from Mydataset import MyDataset
# from Mydataset import resize_sitk_2D
# import SimpleITK as sitk
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torchsummary import summary
# In[model class]


class NET(nn.Module):
    def __init__(self, net):
        super(NET, self).__init__()
        
        # get the pretrained model
        self.net = net
        
        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(self.net.conv1,
                                           self.net.batch1,
                                           self.net.relu1,
                                           self.net.pool1,
                                           self.net.conv2,
                                           self.net.batch2,
                                           self.net.relu2,
                                           self.net.pool2,
                                           self.net.conv3,
                                           self.net.batch3,
                                           self.net.relu3,
                                           self.net.pool3,
                                           self.net.conv4,
                                           self.net.batch4,
                                           self.net.relu4,
                                           self.net.pool4,
                                           self.net.conv5)
                                           
                                                          
       # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the model
        self.classifier = nn.Sequential(self.net.relu5,
                                        self.net.fc1)
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

# In[Grad-CAM]

def get_grad_cam(model,img):
    
    net = NET(model)
    net.eval()
    
    # get the most likely prediction of the model
    pred = net(img)
    # get the gradient of the output with respect to the parameters of the model
    pred[:,0].backward() # 0 is for nac
        
    # pull the gradients out of the model
    gradients = net.get_activations_gradient()
    
    # pool the gradients across the channels -> mean intensity of the gradient over a specific feature-map channel
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    
    # get the activations of the last convolutional layer
    activations = net.get_activations(img).detach()
    
    # weight the channels by corresponding gradients (this model has 128 out channels in conv5 )
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    
    # relu on top of the heatmap
    heatmap = np.maximum(heatmap, 0)
    
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    
    return heatmap

# In[initialize]

model = torch.load('/home/WIN-UNI-DUE/sotiling/WS20/pmb_ws20_dl-pet-ac-attention/pmb_ws20_dl-pet-ac-attention-Marie-Padberg-grad_cam_new/grad_cam_new/classification_rec_complete_neu3', map_location=torch.device('cpu'))
model = model.double()

data_path = '/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl'
patients_list_train = [15]
train_set = MyDataset(data_path,
                      patientList=patients_list_train,
                      transform=transforms.ToTensor())


print('start dataloader2')
dataloader2= []
loader = DataLoader(train_set, shuffle=False, batch_size=1)

for NAC_img, AC_img,_ in loader:
    print('NAC_img',NAC_img.shape)
    # summary(model, input_size=(1,1,384,384))
    # break
    cam = get_grad_cam(model,NAC_img)
    plt.imshow(cam, 'jet')
    plt.show()
    print('a')


    


# In[sources]
#http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf p.172ff
#https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

# %%
