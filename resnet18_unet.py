import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torchsummary import summary

class Decoder(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)  # upconvolution
        self.conv_relu = nn.Sequential(nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True))


    # defines the logic of forward propagation
    def forward(self, x1, x2):
        x1 = self.up(x1)  # do upconvolution on feature map x1 in decoder
        x1 = torch.cat((x1, x2), dim=1)  # concatenate x1 and x2 which is coresponding feature map in encoder
        x1 = self.conv_relu(x1)  # do conv_relu on the combined feature maps
        return x1


# The structure of the entire model
class resnet18_unet(nn.Module):
    def __init__(self):
        super(resnet18_unet, self).__init__()
        base_model = torchvision.models.resnet18(pretrained=True)  # pretrained = True, returns the model trained on ImageNet.
        base_layers = list(base_model.children())

        # Encoder
        self.layer1 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(1,1), stride=1, bias=False),
                                    base_layers[0],
                                    base_layers[1],
                                    base_layers[2]
                                    )

        self.layer2 = nn.Sequential(*base_layers[3:5])

        self.layer3 = base_layers[5]
        self.layer4 = base_layers[6]
        self.layer5 = base_layers[7]

        # Decoder
        self.decode4 = Decoder(512, 256 + 256, 256)
        self.decode3 = Decoder(256, 256 + 128, 256)
        self.decode2 = Decoder(256, 128 + 64, 128)
        self.decode1 = Decoder(128, 64 + 64, 64)
        self.decode0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                                    nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)
                                    )

    def forward(self, input):
        e1 = self.layer1(input)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        f = self.layer5(e4)

        d4 = self.decode4(f, e4)
        d3 = self.decode3(d4, e3)
        d2 = self.decode2(d3, e2)
        d1 = self.decode1(d2, e1)
        d0 = self.decode0(d1)

        return d0

# if __name__ == "__main__":

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = resnet18_unet().to(device)
#     print(model)
#     summary(model, input_size=(1,384,384))
