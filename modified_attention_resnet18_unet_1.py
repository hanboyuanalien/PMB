import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torchsummary import summary

# modified from the attention resnet18_unet model 
# gating signals of All attention gates were replaced by Grad-CAM 

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

# The structure of the entire model
class modified_att_resnet18_unet_1(nn.Module): 
    def __init__(self):
        super().__init__()

        base_model = torchvision.models.resnet18(pretrained =True)
        # pretrained = True, returns the model trained on ImageNet.
        base_layers = list(base_model.children())
        
        # Encoder
        self.layer1 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(1,1),stride=1, bias=False),
                                    base_layers[0],
                                    base_layers[1],
                                    base_layers[2]
                                    )
                
        self.layer2 = nn.Sequential(*base_layers[3:5])
        self.layer3 = base_layers[5]
        self.layer4 = base_layers[6]
        self.layer5 = base_layers[7]

        # Decoder
        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Att5 = Attention_block(F_g=1,F_l=256,F_int=128) # F_g comes from CAM and has only one channel
        self.Up_conv5 = conv_block(ch_in=512,ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Att4 = Attention_block(F_g=1,F_l=128,F_int=64) # F_g comes from CAM and has only one channel
        self.Up_conv4 = conv_block(ch_in=256,ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Att3 = Attention_block(F_g=1,F_l=64,F_int=32) # F_g comes from CAM and has only one channel
        self.Up_conv3 = conv_block(ch_in=128,ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=64)
        self.Att2 = Attention_block(F_g=1,F_l=64,F_int=32) # F_g comes from CAM and has only one channel
        self.Up_conv2 = conv_block(ch_in=128,ch_out=32)

        self.export = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
                                    nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False)
                                    )

    def forward(self,input,cam1,cam2,cam3,cam4):
        e1 = self.layer1(input) 
        e2 = self.layer2(e1) 
        e3 = self.layer3(e2) 
        e4 = self.layer4(e3) 
        f = self.layer5(e4) 

        d5 = self.Up5(f) # d5 torch.Size([2, 256, 24, 24])
        x4_att = self.Att5(cam1,e4)
        d5 = torch.cat((x4_att,d5),dim=1) 
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5) # d4 torch.Size([2, 128, 48, 48])
        x3_att = self.Att4(cam2,e3)
        d4 = torch.cat((x3_att,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4) # d3 torch.Size([2, 64, 96, 96])
        x2_att = self.Att3(cam3,e2)
        d3 = torch.cat((x2_att,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3) # d2 torch.Size([2, 64, 192, 192])
        x1_att = self.Att2(cam4,e1)
        d2 = torch.cat((x1_att,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.export(d2)
        
        return d1

# if __name__ == "__main__":

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = modified_att_resnet18_unet_1().to(device)
#     print(model)
#     summary(model, input_size=(1,384,384))
