import torch
import torch.nn as nn

def dw_conv3(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        # nn.ReLU6(),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )

class Header(nn.Module):
    def __init__(self, in_channel, out_channel, expand=4, mode='train'):
        super(Header, self).__init__()

        self.mode = mode

        hidden_channel = in_channel * expand
        
        self.header = nn.Sequential(*[
            dw_conv3(in_channel,
                    hidden_channel),
            nn.Conv2d(hidden_channel,
                    out_channel, 1, 1, 0, bias=True),
            nn.Sigmoid()
        ])

        # weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.header(x)

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
    
class OcrHeader(nn.Module):
    def __init__(self, in_channel):
        super(OcrHeader, self).__init__()
        
        # (16, 4) [24+40+80+160] 304 / [16+24+40+96] 176
        # ( 8, 2)
        # ( 8, 1)

        # hidden_channel = in_channel * expand
        
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(in_channel, in_channel, 3, (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        ])
        
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(in_channel, in_channel, 3, (2, 1), (1, 1), bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        ])
        
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(in_channel, 120, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
        ])
        
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(120, 66, 1, 1, bias=True),
            # nn.Sigmoid(),
        ])

        # weight initialization
        self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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


class OcrCTCHeader(nn.Module):
    def __init__(self, in_channel):
        super(OcrCTCHeader, self).__init__()
        
        # (16, 4) [24+40+80+160] 304
        # (16, 2)
        # (16, 1)

        # hidden_channel = in_channel * expand
        
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(in_channel, in_channel, 3, (2, 1), (1, 1), bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        ])
        
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(in_channel, in_channel, 3, (2, 1), (1, 1), bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        ])
        
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(in_channel, 120, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
        ])
        
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(120, 66, 1, 1, bias=True),
        ])

        # weight initialization
        self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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