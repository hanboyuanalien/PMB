import torch
from torch import nn
from torchsummary import summary
import cv2

# modified from the attention unet model 
# gating signals of the 1st and 2nd attention gates were replaced

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()           
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2), # for memory burnout
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
            )
    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
        
class modified_att_unet_4(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(modified_att_unet_4,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=1,F_l=512,F_int=256) # replace gating signal with lowest resolution
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=1,F_l=256,F_int=128) # replace gating signal with 2 lowest resolution
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32) 

        self.Up_conv2d_1 = nn.Conv2d(128, 1, kernel_size=3,stride=1,padding=1,bias=True)
        self.Up_conv2d_batch = nn.BatchNorm2d(1)
        self.Up_conv2d_relu = nn.ReLU(inplace=True)

        self.Up_conv2d_2 = nn.Conv2d(1, 1, kernel_size=3,stride=1,padding=1,bias=True)
        self.Up_conv2d_batch2 = nn.BatchNorm2d(1)
        self.Up_conv2d_relu2 = nn.ReLU(inplace=True)

    def forward(self,x,cam1,cam2):
        # encoding path
        
        x1 = self.Conv1(x)

        x1_p = self.Maxpool(x1)
        x2 = self.Conv2(x1_p)

        x2_p = self.Maxpool(x2)
        x3 = self.Conv3(x2_p)

        x3_p = self.Maxpool(x3)
        x4 = self.Conv4(x3_p)

        x4_p = self.Maxpool(x4)
        x5 = self.Conv5(x4_p) # feature vectors

        # decoding + concat path
        d5 = self.Up5(x5) # d5 torch.Size([2, 512, 48, 48])
        x4_att = self.Att5(cam1,x4) # cam1 inserted hier
        d5 = torch.cat((x4_att,d5),dim=1) 
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5) # d4 torch.Size([2, 256, 96, 96])
        x3_att = self.Att4(cam2,x3) # cam2 inserted hier
        d4 = torch.cat((x3_att,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4) 
        x2_att = self.Att3(d3,x2)
        d3 = torch.cat((x2_att,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3) 
        x1_att = self.Att2(d2,x1) 
        d2 = torch.cat((x1_att,d2),dim=1)

        d2 = self.Up_conv2d_1(d2)
        d2 = self.Up_conv2d_batch(d2)
        d2 = self.Up_conv2d_relu(d2)
        d1 = self.Up_conv2d_2(d2)
        d1 = self.Up_conv2d_batch2(d1)
        d1 = self.Up_conv2d_relu2(d1)

        return d1