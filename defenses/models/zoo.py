import os
import os.path as osp
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.classifier.zoo import get_cls


class FeatureWrappedModel(nn.Module):
    """
    Wrap classifier models from utils.classifier.zoo and provide:
    1) forward(x) -> logits
    2) get_feats(x) -> penultimate feature
    3) feat_to_sm(feat) -> logits from feature
    """

    def __init__(self, base_model: nn.Module, arch_name: str):
        super().__init__()
        self.base_model = base_model
        self.arch_name = arch_name.lower()

    def forward(self, x):
        return self.base_model(x)

    @torch.no_grad()
    def get_feats(self, x):
        name = self.arch_name

        if "vgg" in name:
            x = self.base_model.features(x)
            x = self.base_model.avgpool(x)
            x = x.view(x.size(0), -1)
            for layer in list(self.base_model.classifier)[:-1]:
                x = layer(x)
            return x

        if "resnet" in name:
            m = self.base_model
            x = m.conv1(x)
            x = m.bn1(x)
            x = m.relu(x)
            x = m.maxpool(x)

            x = m.layer1(x)
            x = m.layer2(x)
            x = m.layer3(x)
            x = m.layer4(x)

            x = m.avgpool(x)
            x = x.reshape(x.size(0), -1)
            return x

        if "densenet" in name:
            m = self.base_model
            x = m.features(x)
            x = F.relu(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return x

        if hasattr(self.base_model, "get_features"):
            feat = self.base_model.get_features(x)
            if feat.dim() > 2:
                feat = F.adaptive_avg_pool2d(feat, (1, 1))
                feat = torch.flatten(feat, 1)
            return feat

        raise NotImplementedError(
            f"get_feats() is not implemented for architecture: {self.arch_name}"
        )

    def feat_to_sm(self, feat):
        name = self.arch_name

        if "vgg" in name:
            return self.base_model.classifier[-1](feat)

        if "alexnet" in name:
            return self.base_model.classifier[-1](feat)

        if "resnet" in name:
            return self.base_model.fc(feat)

        if "densenet" in name:
            return self.base_model.classifier(feat)

        if hasattr(self.base_model, "fc") and isinstance(self.base_model.fc, nn.Module):
            return self.base_model.fc(feat)

        if hasattr(self.base_model, "classifier"):
            clf = self.base_model.classifier
            if isinstance(clf, nn.Linear):
                return clf(feat)
            if isinstance(clf, nn.Sequential):
                return clf(feat)

        raise NotImplementedError(
            f"feat_to_sm() is not implemented for architecture: {self.arch_name}"
        )



    # I12_FEATUREWRAPPED_ROT_FORWARD_COMPAT_START
    def rot_forward(self, x):
        """Compatibility fallback for S4L rotation loss.
    
        Some original QUEEN/S4L code expects the model to expose a
        rotation-prediction head named rot_forward. The current wrapped
        CIFAR10 vgg16_bn model does not expose it, so this method first
        delegates to any wrapped model that has rot_forward; otherwise it
        returns the first four logits/features of the normal forward pass.
        """
        for attr in ("model", "base_model", "_model", "net", "classifier", "module"):
            obj = getattr(self, attr, None)
            if obj is None or obj is self:
                continue
            if hasattr(obj, "rot_forward"):
                return obj.rot_forward(x)
        out = self.forward(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if hasattr(out, "logits"):
            out = out.logits
        if out.dim() > 2:
            out = out.view(out.size(0), -1)
        if out.size(1) < 4:
            pad = out.new_zeros(out.size(0), 4 - out.size(1))
            out = torch.cat([out, pad], dim=1)
        return out[:, :4]
    # I12_FEATUREWRAPPED_ROT_FORWARD_COMPAT_END
def _normalize_arch_name(model_name: str) -> str:
    mapping = {
        "vgg11_bn": "VGG11-BN",
        "vgg13_bn": "VGG13-BN",
        "vgg16_bn": "VGG16-BN",
        "vgg19_bn": "VGG19-BN",
        "resnet18": "ResNet18",
        "resnet34": "ResNet34",
        "resnet50": "ResNet50",
        "densenet121": "DenseNet121",
        "densenet161": "DenseNet161",
        "densenet169": "DenseNet169",
        "mobilenet_v2": "MobileNet",
        "mobilenet": "MobileNet",
        "googlenet": "GoogleNet",
        "inception_v3": "InceptionNet",
        "alexnet": "AlexNet",
        "lenet": "LeNet",
        "convnet": "ConvNet",
        "fullyconnected": "FullyConnected",
    }

    key = model_name.strip()
    if key in mapping:
        return mapping[key]

    key_lower = key.lower()
    if key_lower in mapping:
        return mapping[key_lower]

    return key


def _load_pretrained_or_checkpoint(model: nn.Module, pretrained: Optional[str], device: str):
    """
    Only load explicit checkpoint paths here.

    Notes:
    - 'imagenet' / 'true' should be handled inside get_cls(..., pretrained=...)
    - 'none' / 'false' / '' mean no extra loading
    """
    if pretrained is None or pretrained is False:
        return model

    if isinstance(pretrained, str):
        p = pretrained.strip()

        if p.lower() in ["imagenet", "true", "none", "false", ""]:
            return model

        if osp.exists(p):
            # HGROUP ZOO CHECKPOINT DIR PATCH START
            if isinstance(p, (str, os.PathLike)) and os.path.isdir(str(p)):
                _hgroup_ckpt = os.path.join(str(p), "checkpoint.pth.tar")
                if os.path.exists(_hgroup_ckpt):
                    p = _hgroup_ckpt
                    print(f"=> HGROUP ZOO PATCH: using checkpoint file {p}")
            # HGROUP ZOO CHECKPOINT DIR PATCH END
            ckpt = torch.load(p, map_location=device, weights_only=False)
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    model.load_state_dict(ckpt["state_dict"], strict=False)
                elif "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)

    return model


def get_net(
    model_name,
    modelfamily=None,
    pretrained=None,
    num_classes=10,
    device="cpu",
    rot_semi=False,
    **kwargs,
):
    arch = _normalize_arch_name(model_name)

    extra_kwargs = dict(kwargs)
    extra_kwargs.pop("rot_semi", None)

    # Critical fix:
    # pass pretrained into get_cls so lower-level model builders can
    # actually construct/load ImageNet-pretrained backbones when supported.
    # HGROUP ZOO NUM_CLASSES PATCH START
    if num_classes is None:
        _hgroup_mf_raw = str(modelfamily) if modelfamily is not None else ""
        _hgroup_mf = _hgroup_mf_raw.strip().lower().replace("-", "").replace("_", "")
        if _hgroup_mf in ("caltech256", "caltech", "caltech256dataset"):
            num_classes = 256
        elif _hgroup_mf in ("cubs200", "cub200", "cub2002011", "cub", "cub2011"):
            num_classes = 200
        elif _hgroup_mf in ("cifar10",):
            num_classes = 10
        elif _hgroup_mf in ("cifar100",):
            num_classes = 100
        elif _hgroup_mf in ("imagenet", "ilsvrc2012", "imagenet1k"):
            num_classes = 1000
        if num_classes is not None:
            print(f"=> HGROUP ZOO PATCH: modelfamily={_hgroup_mf_raw} -> num_classes={num_classes}")
    # HGROUP ZOO NUM_CLASSES PATCH END
    base_model = get_cls(
        arch,
        num_classes=num_classes,
        modelfamily=modelfamily,
        device=device,
        pretrained=pretrained,
        **extra_kwargs,
    )

    # Only explicit checkpoint paths are loaded here.
    base_model = _load_pretrained_or_checkpoint(base_model, pretrained, device=device)

    wrapped = FeatureWrappedModel(base_model, arch)
    wrapped = wrapped.to(device)
    return wrapped
