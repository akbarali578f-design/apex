import torch
import torch.nn as nn
from torchvision.models import vgg11_bn as tv_vgg11_bn
from torchvision.models import vgg13_bn as tv_vgg13_bn
from torchvision.models import vgg16_bn as tv_vgg16_bn
from torchvision.models import vgg19_bn as tv_vgg19_bn
from torchvision.models import (
    VGG11_BN_Weights,
    VGG13_BN_Weights,
    VGG16_BN_Weights,
    VGG19_BN_Weights,
)

__all__ = [
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
]


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True, **kwargs):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    @torch.no_grad()
    def get_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels: int = 3):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
        512, 512, 512, "M", 512, 512, 512, "M",
    ],
    "E": [
        64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
        512, 512, 512, 512, "M", 512, 512, 512, 512, "M",
    ],
}


def _tv_builder_and_weights(arch: str):
    if arch == "vgg11_bn":
        return tv_vgg11_bn, VGG11_BN_Weights.IMAGENET1K_V1
    if arch == "vgg13_bn":
        return tv_vgg13_bn, VGG13_BN_Weights.IMAGENET1K_V1
    if arch == "vgg16_bn":
        return tv_vgg16_bn, VGG16_BN_Weights.IMAGENET1K_V1
    if arch == "vgg19_bn":
        return tv_vgg19_bn, VGG19_BN_Weights.IMAGENET1K_V1
    raise ValueError(f"Unsupported arch for torchvision pretrained load: {arch}")


def _load_imagenet_pretrained_partial(model: nn.Module, arch: str):
    builder, weights = _tv_builder_and_weights(arch)
    tv_model = builder(weights=weights)
    src_state = tv_model.state_dict()
    dst_state = model.state_dict()

    matched = {}
    for k, v in src_state.items():
        if k in dst_state and dst_state[k].shape == v.shape:
            matched[k] = v

    dst_state.update(matched)
    model.load_state_dict(dst_state, strict=False)
    print(f"=> loaded ImageNet-pretrained compatible params for {arch}: {len(matched)} tensors")


def _vgg(arch, cfg, batch_norm, pretrained, progress, device, **kwargs):
    use_imagenet = False
    if isinstance(pretrained, str):
        use_imagenet = pretrained.strip().lower() in ["imagenet", "true"]
    elif isinstance(pretrained, bool):
        use_imagenet = pretrained

    if use_imagenet:
        kwargs["init_weights"] = False

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, in_channels=3), **kwargs)

    if use_imagenet:
        _load_imagenet_pretrained_partial(model, arch)

    return model


def vgg11_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    return _vgg("vgg11_bn", "A", True, pretrained, progress, device, **kwargs)


def vgg13_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    return _vgg("vgg13_bn", "B", True, pretrained, progress, device, **kwargs)


def vgg16_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    return _vgg("vgg16_bn", "D", True, pretrained, progress, device, **kwargs)


def vgg19_bn(pretrained=False, progress=True, device="cpu", **kwargs):
    return _vgg("vgg19_bn", "E", True, pretrained, progress, device, **kwargs)
