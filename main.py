import torch
from torchsummary import summary

from model import ResNet18

resnet18 = ResNet18(3, outputs=1000)
resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(resnet18, (3, 224, 224))
