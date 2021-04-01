# In[0]
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt

import numpy as np
import os
from Mydataset import MyDataset
from Mydataset import resize_sitk_2D
from my_model import CNN
# import SimpleITK as sitk
from torch.utils.data import DataLoader
from torchsummary import summary


# In[model]

model = torch.load('/home/WIN-UNI-DUE/sotiling/WS20/pmb_ws20_dl-pet-ac-attention/pmb_ws20_dl-pet-ac-attention-Marie-Padberg-grad_cam_new/grad_cam_new/classification_rec_complete_neu3', map_location=torch.device('cuda'))
model = model.float()
# summary(model, input_size=(1,384,384))

# In[Grad-CAM]

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        
        # get the pretrained model
        self.net = model
        
        # dissect the network to access its last convolutional layer
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

def grad_cam(img):
    
    net = NET()
    net.eval()
    
    # get the most likely prediction of the model
    pred = net(img)
    
    # get the gradient of the output with respect to the parameters of the model
    most_likely_class = np.argmax(np.array([float(pred[0][0]), float(pred[0][1])]))
    if most_likely_class == 0: 
        print('most likely class: NAC')
        pred[:,0].backward()
    else: 
        print('most likely class: AC')
        pred[:,1].backward()
    # 0 für nac and 1 for ac
    
    
    # pull the gradients out of the model
    gradients = net.get_activations_gradient()
    
    # pool the gradients across the channels -> mean intensity of the gradient over a specific feature-map channel
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    
    # get the activations of the last convolutional layer
    activations = net.get_activations(img).detach()
    #print('act1',activations)
    # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    # detach () creates a new view such that these operations are no more tracked i.e gradient is no longer being 
    # computed and subgraph is not going to be recorded
    
    # weight the channels by corresponding gradients (this model has 128 out channels in conv5 )
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    
    # relu on top of the heatmap
    heatmap = np.maximum(heatmap, 0)
    
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    
    # draw the heatmap (size is dictated by the spatial informations of the last conv layer)
    plt.imshow(heatmap.squeeze(), cmap='jet')
    plt.colorbar()
    return heatmap

# In[initialize]
data_path = '/media/MedIP-Praxisprojekt/WS20_PMB/NAC_AC_gipl'
patients_list_train = [15]
train_set = MyDataset(data_path,
                      patientList=patients_list_train,
                      transform=transforms.ToTensor())

print('start dataloader2')
dataloader2= []
a = DataLoader(train_set, shuffle=False, batch_size=1)
for batch,(nac, ac, _) in enumerate(a, 1):
    print(batch)
    dataloader2 += [[nac, [0]]] # 0 für nac
    dataloader2 += [[ac, [1]]] # 1 für ac

for image,_ in dataloader2[:1]:
    img = image

heatmap = grad_cam(img)

 # draw the heatmap (size is dictated by the spatial informations of the last conv layer)
#plt.imshow(heatmap.squeeze(), cmap='jet')
#plt.colorbar()

img = img[0, 0]
heatmap = np.float32(resize_sitk_2D(heatmap, (344, 344)))
img = np.float32(resize_sitk_2D(img, (344, 344)))

fig = plt.figure(figsize=(21, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, 'gray', interpolation='none')
#plt.colorbar()
plt.title('original AC image (slice 300)')
plt.subplot(1, 3, 2)
plt.imshow(heatmap, 'jet', vmin=0, vmax=1, interpolation='none')
plt.colorbar()
plt.title('Grad-CAM')
plt.subplot(1, 3, 3)
plt.imshow(img, 'gray', interpolation='none')
plt.imshow(heatmap, 'jet', vmin=0, vmax=1, interpolation='none', alpha=0.5)
plt.colorbar()
plt.title('Grad-CAM and original image')
plt.show()
fig.savefig('grad-cam_ac_300.tif', bbox_inches='tight')

# In[sources]
#http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf p.172ff
#https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
