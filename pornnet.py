import torch.nn as nn
import torch
from torchsummary import summary

from resnet import res_18, res_34
from littlenet import LitNet


class PornNet(nn.Module):
    def __init__(self, arch):
        super(PornNet, self).__init__()
        backbone_factory = {'res_18': res_18(output_all_layer=True),
                            'res_34': res_34(output_all_layer=True),
                            'litnet': LitNet()}
        self.backbone = backbone_factory[arch]  # [N,C,H,W]:[N,512,5,5]
        self.out_conv = OutConv()

    def forward(self, x):
        feature = self.backbone(x)
        n, c, _, _ = x.size()
        out_feature = feature
        out_label = self.out_conv(out_feature)
        out_label = out_label.view(n, 5)
        # print(out_label.size())

        return out_label.softmax(dim=1)


class OutConv(nn.Module):
    def __init__(self):
        super(OutConv, self).__init__()
        conv_layer1 = [nn.Conv2d(256, 512, 3, stride=1, padding=1),
                       nn.BatchNorm2d(512),
                       nn.ReLU()
                       ]

        conv_layer2 = [nn.Conv2d(512, 64, 1),
                       nn.BatchNorm2d(64),
                       nn.ReLU()
                       ]
        conv_layer3 = [nn.Conv2d(64, 5, 1)
                       ]
        self.layer1 = nn.Sequential(*conv_layer1)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.layer2 = nn.Sequential(*conv_layer2)
        self.layer3 = nn.Sequential(*conv_layer3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool2(self.pool1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = PornNet('litnet')
    summary(model, (3, 128, 178), device='cpu')
    print(model(torch.rand([8, 3, 128, 178])))
