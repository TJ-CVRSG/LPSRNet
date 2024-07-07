import torch
import torch.nn as nn
import torch.nn.functional as F

# P1 P2 P3 P4
# (128, 32) -> (16, 4)
# (128, 32) P1
# ( 64, 16) P2
# ( 32,  8) P3
# ( 16,  4) P4

class LightPAN(nn.Module):
    def __init__(self, stage_in_channels):
        super(LightPAN, self).__init__()
        assert len(stage_in_channels) == 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(stage_in_channels[0], stage_in_channels[0], 3, 2, 1, bias = False),
            nn.BatchNorm2d(stage_in_channels[0]),
            nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(stage_in_channels[0] + stage_in_channels[0], stage_in_channels[0] + stage_in_channels[0], 3, 2, 1, bias = False),
            nn.BatchNorm2d(stage_in_channels[0] + stage_in_channels[0]),
            nn.ReLU(inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(stage_in_channels[0] + stage_in_channels[0] + stage_in_channels[1], stage_in_channels[0] + stage_in_channels[0] + stage_in_channels[1], 3, 2, 1, bias = False),
            nn.BatchNorm2d(stage_in_channels[0] + stage_in_channels[0] + stage_in_channels[1]),
            nn.ReLU(inplace=True),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(stage_in_channels[0] + stage_in_channels[0] + stage_in_channels[1] + stage_in_channels[2], stage_in_channels[0] + stage_in_channels[0] + stage_in_channels[1] + stage_in_channels[2], 3, 1, 1, bias = False),
            nn.BatchNorm2d(stage_in_channels[0] + stage_in_channels[0] + stage_in_channels[1] + stage_in_channels[2]),
            nn.ReLU(inplace=True),
        )
        
        # weight initialization
        self._initialize_weights()
        
    def forward(self, P1, P2, P3, P4):
        x = self.conv1(P1)
        x = torch.cat((x, P2), dim=1)
        
        x = self.conv2(x)
        x = torch.cat((x, P3), dim=1)
        
        x = self.conv3(x)
        x = torch.cat((x, P4), dim=1)
        
        x = self.conv4(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)