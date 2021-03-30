from ret_benchmark.modeling.registry import BACKBONES

from .bninception import BNInception
from .resnet import ResNet18, ResNet50
from .googlenet import GoogLeNet

def build_backbone(cfg):
    assert (
        cfg.MODEL.BACKBONE.NAME in BACKBONES
    ), f"backbone {cfg.MODEL.BACKBONE} is not defined"
    model = BACKBONES[cfg.MODEL.BACKBONE.NAME](
        last_stride=cfg.MODEL.BACKBONE.LAST_STRIDE
    )
    if cfg.MODEL.BACKBONE.FREEZE_BN:
        from torch import nn
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    return model
