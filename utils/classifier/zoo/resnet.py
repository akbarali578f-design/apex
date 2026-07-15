import os
import torch
import torch.nn as nn

__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
]


# -----------------------------------------------------------------------------
# Basic layers
# -----------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


# -----------------------------------------------------------------------------
# Pretrained / stem helpers
# -----------------------------------------------------------------------------

def _wants_pretrained(pretrained):
    """
    Accept bool or string style pretrained flags.

    Examples:
      pretrained=True        -> True
      pretrained="imagenet"  -> True
      pretrained="none"      -> False
      pretrained=None        -> False
    """
    if isinstance(pretrained, str):
        return pretrained.lower() not in ["none", "false", "0", "no", ""]
    return bool(pretrained)


def _infer_imagenet_stem(arch, pretrained, imagenet_stem, kwargs):
    """
    Decide whether to use the standard ImageNet ResNet stem.

    This is the important compatibility fix for your H-group experiments:
    - Caltech256 ResNet50 checkpoint has conv1.weight shape [64, 3, 7, 7]
    - CUBS200 ResNet50 checkpoint also uses the same 7x7 ImageNet stem
    - During utility loading, pretrained may be None, so relying only on
      pretrained is not enough.

    Rule:
      1. If caller explicitly passes imagenet_stem, respect it.
      2. If pretrained is requested, use ImageNet stem.
      3. If ResNet50 has num_classes 200 or 256, use ImageNet stem.
      4. Otherwise keep original QUEEN/CIFAR-style 3x3 stem.
    """
    if imagenet_stem is not None:
        return bool(imagenet_stem)

    if _wants_pretrained(pretrained):
        return True

    num_classes = kwargs.get("num_classes", None)
    if arch == "resnet50" and num_classes in [200, 256]:
        return True

    return False


# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# -----------------------------------------------------------------------------
# ResNet
# -----------------------------------------------------------------------------

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        imagenet_stem=False,
    ):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.imagenet_stem = bool(imagenet_stem)

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple, "
                "got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group

        if self.imagenet_stem:
            # Standard ImageNet ResNet stem.
            # Required for your Caltech256 / CUBS200 ResNet50 victim checkpoints.
            self.conv1 = nn.Conv2d(
                3,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        else:
            # Original QUEEN/CIFAR-style stem.
            self.conv1 = nn.Conv2d(
                3,
                self.inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # Keep maxpool to preserve project compatibility.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )

        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )

        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights(zero_init_residual)

    def _init_weights(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer

        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.fc(x)
        return x

    @torch.no_grad()
    def get_features(self, x):
        return self._forward_features(x)


# -----------------------------------------------------------------------------
# Weight loading
# -----------------------------------------------------------------------------

def _strip_module_prefix(state_dict):
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        new_state_dict[k] = v

    return new_state_dict


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _load_torchvision_state_dict(arch, progress=True):
    import torchvision.models as tv_models

    if arch == "resnet18":
        if hasattr(tv_models, "ResNet18_Weights"):
            tv_model = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.IMAGENET1K_V1,
                progress=progress,
            )
        else:
            tv_model = tv_models.resnet18(pretrained=True, progress=progress)

    elif arch == "resnet34":
        if hasattr(tv_models, "ResNet34_Weights"):
            tv_model = tv_models.resnet34(
                weights=tv_models.ResNet34_Weights.IMAGENET1K_V1,
                progress=progress,
            )
        else:
            tv_model = tv_models.resnet34(pretrained=True, progress=progress)

    elif arch == "resnet50":
        if hasattr(tv_models, "ResNet50_Weights"):
            tv_model = tv_models.resnet50(
                weights=tv_models.ResNet50_Weights.IMAGENET1K_V1,
                progress=progress,
            )
        else:
            tv_model = tv_models.resnet50(pretrained=True, progress=progress)

    else:
        raise ValueError("Unsupported torchvision ResNet arch: {}".format(arch))

    return tv_model.state_dict()


def _load_pretrained_weights(model, arch, progress=True, device="cpu"):
    script_dir = os.path.dirname(__file__)
    local_path = os.path.join(script_dir, "state_dicts", arch + ".pt")

    if os.path.exists(local_path):
        print("=> loading local pretrained weights from {}".format(local_path))
        loaded_obj = torch.load(local_path, map_location=device)
        pretrained_state = _extract_state_dict(loaded_obj)
    else:
        print(
            "=> WARNING: requested pretrained=imagenet for {}, "
            "but local state_dict was not found.".format(arch)
        )
        print("=> trying torchvision ImageNet weights as fallback...")
        pretrained_state = _load_torchvision_state_dict(arch, progress=progress)

    pretrained_state = _strip_module_prefix(pretrained_state)
    model_state = model.state_dict()

    matched_state = {}
    skipped = []
    unexpected = []

    for k, v in pretrained_state.items():
        if k not in model_state:
            unexpected.append(k)
            continue

        if tuple(v.shape) == tuple(model_state[k].shape):
            matched_state[k] = v
        else:
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))

    load_result = model.load_state_dict(matched_state, strict=False)

    print("=> loaded {} parameters into {}".format(len(matched_state), arch))

    if skipped:
        print("=> skipped {} parameters due to shape mismatch".format(len(skipped)))
        for item in skipped[:20]:
            print("   shape mismatch: {}".format(item))
        if len(skipped) > 20:
            print("   ... {} more skipped".format(len(skipped) - 20))

    if unexpected:
        print("=> ignored {} unexpected parameters".format(len(unexpected)))

    if load_result.missing_keys:
        print("=> missing keys after partial load: {}".format(len(load_result.missing_keys)))
        for k in load_result.missing_keys[:20]:
            print("   missing: {}".format(k))
        if len(load_result.missing_keys) > 20:
            print("   ... {} more missing".format(len(load_result.missing_keys) - 20))

    return model


# -----------------------------------------------------------------------------
# Constructors
# -----------------------------------------------------------------------------

def _resnet(
    arch,
    block,
    layers,
    pretrained=False,
    progress=True,
    device="cpu",
    imagenet_stem=None,
    **kwargs
):
    use_pretrained = _wants_pretrained(pretrained)
    use_imagenet_stem = _infer_imagenet_stem(
        arch=arch,
        pretrained=pretrained,
        imagenet_stem=imagenet_stem,
        kwargs=kwargs,
    )

    model = ResNet(
        block,
        layers,
        imagenet_stem=use_imagenet_stem,
        **kwargs,
    )

    print(
        "=> build {} with imagenet_stem={}, pretrained={}, num_classes={}".format(
            arch,
            use_imagenet_stem,
            pretrained,
            kwargs.get("num_classes", None),
        )
    )

    if use_pretrained:
        model = _load_pretrained_weights(
            model,
            arch=arch,
            progress=progress,
            device=device,
        )

    return model


def resnet18(pretrained=False, progress=True, device="cpu", imagenet_stem=None, **kwargs):
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained=pretrained,
        progress=progress,
        device=device,
        imagenet_stem=imagenet_stem,
        **kwargs,
    )


def resnet34(pretrained=False, progress=True, device="cpu", imagenet_stem=None, **kwargs):
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        pretrained=pretrained,
        progress=progress,
        device=device,
        imagenet_stem=imagenet_stem,
        **kwargs,
    )


def resnet50(pretrained=False, progress=True, device="cpu", imagenet_stem=None, **kwargs):
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained=pretrained,
        progress=progress,
        device=device,
        imagenet_stem=imagenet_stem,
        **kwargs,
    )
