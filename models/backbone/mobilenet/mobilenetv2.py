from torch import nn
from torchvision.ops.misc import Conv2dNormActivation

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=nn.BatchNorm2d,
                                             activation_layer=nn.ReLU6))
        layers.extend([
            # dw
            Conv2dNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0):

        super(MobileNetV2, self).__init__()

        input_channel = 32
        last_channel = 1280

        # building first layer
        input_channel = _make_divisible(
            input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult),
                                            8)

        # building features
        self.features1 = nn.Sequential(*[
            Conv2dNormActivation(3, input_channel, kernel_size=3, stride=2,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6),
            InvertedResidual(input_channel, _make_divisible(
                16 * width_mult, 8), 1, expand_ratio=1),
            InvertedResidual(_make_divisible(16 * width_mult, 8),
                             _make_divisible(24 * width_mult, 8), 2, expand_ratio=6),
            InvertedResidual(_make_divisible(24 * width_mult, 8),
                             _make_divisible(24 * width_mult, 8), 1, expand_ratio=6)
        ])

        self.features2 = nn.Sequential(*[
            InvertedResidual(_make_divisible(24 * width_mult, 8),
                             _make_divisible(32 * width_mult, 8), 2, expand_ratio=6),
            InvertedResidual(_make_divisible(32 * width_mult, 8),
                             _make_divisible(32 * width_mult, 8), 1, expand_ratio=6),
            InvertedResidual(_make_divisible(32 * width_mult, 8),
                             _make_divisible(32 * width_mult, 8), 1, expand_ratio=6),

        ])

        self.features3 = nn.Sequential(*[
            InvertedResidual(_make_divisible(32 * width_mult, 8),
                             _make_divisible(64 * width_mult, 8), 2, expand_ratio=6),
            InvertedResidual(_make_divisible(64 * width_mult, 8),
                             _make_divisible(64 * width_mult, 8), 1, expand_ratio=6),
            InvertedResidual(_make_divisible(64 * width_mult, 8),
                             _make_divisible(64 * width_mult, 8), 1, expand_ratio=6),
            InvertedResidual(_make_divisible(64 * width_mult, 8),
                             _make_divisible(64 * width_mult, 8), 1, expand_ratio=6),
        ])

        self.features4 = nn.Sequential(*[
            InvertedResidual(_make_divisible(64 * width_mult, 8),
                             _make_divisible(96 * width_mult, 8), 1, expand_ratio=6),
            InvertedResidual(_make_divisible(96 * width_mult, 8),
                             _make_divisible(96 * width_mult, 8), 1, expand_ratio=6),
            InvertedResidual(_make_divisible(96 * width_mult, 8),
                             _make_divisible(96 * width_mult, 8), 1, expand_ratio=6),
            InvertedResidual(_make_divisible(96 * width_mult, 8),
                             _make_divisible(160 * width_mult, 8), 2, expand_ratio=6),
            InvertedResidual(_make_divisible(160 * width_mult, 8),
                             _make_divisible(160 * width_mult, 8), 1, expand_ratio=6),
            InvertedResidual(_make_divisible(160 * width_mult, 8),
                             _make_divisible(160 * width_mult, 8), 1, expand_ratio=6),
            InvertedResidual(_make_divisible(160 * width_mult, 8),
                             _make_divisible(320 * width_mult, 8), 1, expand_ratio=6),
            Conv2dNormActivation(_make_divisible(320 * width_mult, 8),
                               self.last_channel, kernel_size=1,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6),
        ])

        # weight initialization
        self._initialize_weights()
        
    def forward(self, x):

        f1 = self.features1(x)
        f2 = self.features2(f1)
        f3 = self.features3(f2)
        f4 = self.features4(f3)

        return f1, f2, f3, f4
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
