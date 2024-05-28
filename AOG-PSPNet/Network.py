import torch
from torch import nn
from torch.nn import functional as F
from AogBlock import *

import extractors


class PSPModule(nn.Module):
    def __init__(
        self,
        features,
        out_features=1024,
        sizes=(1, 2, 3, 6),
        Ttype=T_Normal_Block,
        sub_nums=4,
    ):
        super(PSPModule, self).__init__()
        self.stages = []
        self.Ttype = Ttype
        self.sub_nums = sub_nums
        self.stages = nn.ModuleList(
            [self._make_stage(features, size) for size in sizes]
        )
        self.bottleneck = nn.Conv1d(
            features * (len(sizes) + 1), out_features, kernel_size=1
        )
        self.Lrelu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool1d(output_size=size)
        conv = nn.Conv1d(features, features, kernel_size=1, bias=False)
        # conv = AOG_Building_Block(in_channels=features, out_channels=features,
        #                           stride=1, Ttype=self.Ttype, sub_nums=self.sub_nums)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        l = feats.size(2)
        priors = [
            F.interpolate(input=stage(feats), size=l, mode="nearest")
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.Lrelu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        l = 2 * x.size(2)
        p = F.interpolate(input=x, size=l, mode="nearest")
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(
        self,
        n_classes=1,
        sizes=(1, 2, 3, 6),
        psp_size=2048,
        deep_features_size=1024,
        resblock_size=[3, 4, 23, 3],
        Ttype=T_Normal_Block,
        sub_nums=4,
    ):
        super(PSPNet, self).__init__()
        self.Ttype = Ttype
        self.sub_nums = sub_nums
        self.feats = extractors.ResNet(extractors.Bottleneck, resblock_size)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout(p=0.15)
        self.final = nn.Sequential(nn.Conv1d(64, n_classes, kernel_size=1))
        #         self.final = nn.Sequential(
        #             AOG_Building_Block(in_channels=64, out_channels=n_classes,
        #                                stride=1, Ttype=self.Ttype, sub_nums=self.sub_nums)
        #         )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool1d(input=class_f, output_size=1).view(
            -1, class_f.size(1)
        )
        return self.final(p), self.classifier(auxiliary)


# from torchsummary import summary
# import torch
# from torchvision.models import resnet18

# input_shape = [4096]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = PSPNet().to(device)

# summary(model, input_size=(1, 4096))
