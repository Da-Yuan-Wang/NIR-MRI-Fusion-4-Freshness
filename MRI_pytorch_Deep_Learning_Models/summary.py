# This part of code is only used to view the network structure, not for testing code
#--------------------------------------------#
from torchsummary import summary

from nets.mobilenet import mobilenet_v2
from nets.resnet50 import resnet50
from nets.vgg16 import vgg16
from nets.vit import vit

if __name__ == "__main__":
    model = mobilenet_v2(num_classes=1000, pretrained=False).train().cuda()
    summary(model,(3, 224, 224))

'''https://github.com/sksq96/pytorch-summary'''