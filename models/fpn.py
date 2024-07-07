import torch.nn as nn
import torch.nn.functional as F


class LightFPN(nn.Module):
    def __init__(self, stage_in_channels, stage_out_channels):
        super(LightFPN, self).__init__()
        assert len(stage_in_channels) == 4
        assert len(stage_out_channels) == 3
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(stage_in_channels[3], stage_out_channels[2], 1, 1, 0, bias = False),
            nn.BatchNorm2d(stage_out_channels[2]),
            nn.ReLU6(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(stage_in_channels[2], stage_out_channels[2], 1, 1, 0, bias = False),
            nn.BatchNorm2d(stage_out_channels[2]),
            nn.ReLU6(inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(stage_in_channels[2], stage_out_channels[2], 3, 1, 1,
                      groups=stage_out_channels[2], bias = False),
            nn.BatchNorm2d(stage_out_channels[2]),
            nn.ReLU6(inplace=True),
            nn.Conv2d(stage_out_channels[2], stage_out_channels[1], 1, 1, 0, bias = False),
            nn.BatchNorm2d(stage_out_channels[1]),
            nn.ReLU6(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(stage_in_channels[1], stage_out_channels[1], 1, 1, 0, bias = False),
            nn.BatchNorm2d(stage_out_channels[1]),
            nn.ReLU6(inplace=True),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(stage_in_channels[1], stage_out_channels[1], 3, 1, 1,
                      groups=stage_out_channels[1], bias = False),
            nn.BatchNorm2d(stage_out_channels[1]),
            nn.ReLU6(inplace=True),
            nn.Conv2d(stage_out_channels[1], stage_out_channels[0], 1, 1, 0, bias = False),
            nn.BatchNorm2d(stage_out_channels[0]),
            nn.ReLU6(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(stage_in_channels[0], stage_out_channels[0], 1, 1, 0, bias = False),
            nn.BatchNorm2d(stage_out_channels[0]),
            nn.ReLU6(inplace=True),
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(stage_out_channels[0], stage_out_channels[0], 3, 1, 1,
                      groups=stage_out_channels[0], bias = False),
            nn.BatchNorm2d(stage_out_channels[0]),
            nn.ReLU6(inplace=True),
            nn.Conv2d(stage_out_channels[0], stage_out_channels[0], 1, 1, 0, bias = False),
            nn.BatchNorm2d(stage_out_channels[0]),
            nn.ReLU6(inplace=True),
        )

        # weight initialization
        self._initialize_weights()
        
    def forward(self, C1, C2, C3, C4):
        P4 = self.conv1(C4)
        x = F.interpolate(P4, scale_factor=2, mode="bilinear", align_corners=False)
        x += self.conv2(C3)
        
        P3 = self.conv3(x)
        x = F.interpolate(P3, scale_factor=2, mode="bilinear", align_corners=False)
        x += self.conv4(C2)
        
        P2 = self.conv5(x)
        x = F.interpolate(P2, scale_factor=2, mode="bilinear", align_corners=False)
        x += self.conv6(C1)
        
        P1 = self.conv7(x)
        
        return P1, P2, P3, P4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)