from models.backbone.mobilenet.mobilenetv2 import MobileNetV2
from models.backbone.mobilenet.mobilenetv3 import create_mobilenetv3

def create_mobilenet(cfg):
    if cfg.name == 'mobilenetv2':
        return MobileNetV2(cfg.width_mult)
    elif cfg.name.startswith('mobilenetv3'):
        return create_mobilenetv3(cfg.name, cfg.width_mult, cfg.feature_layers, cfg.reduced_tail, cfg.dilated, cfg.pretrained)