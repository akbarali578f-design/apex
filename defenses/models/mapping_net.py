import torch
import torch.nn as nn


class FeatureMappingNet(nn.Module):
    """
    Map high-dimensional backbone features to 2D for QUEEN sensitivity analysis.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return self.net(x)


def _infer_feature_dim(host_network: str) -> int:
    name = str(host_network).lower()

    if "vgg" in name:
        return 4096
    if "resnet18" in name or "resnet34" in name:
        return 512
    if "resnet50" in name:
        return 2048
    if "densenet121" in name:
        return 1024
    if "densenet161" in name:
        return 2208
    if "densenet169" in name:
        return 1664
    if "mobilenet" in name:
        return 1280
    if "googlenet" in name:
        return 1024
    if "inception" in name:
        return 2048
    if "alexnet" in name:
        return 4096
    if "lenet" in name:
        return 84
    if "convnet" in name:
        return 256

    # fallback
    return 512


def feature_mapping_net(host_network: str):
    in_dim = _infer_feature_dim(host_network)
    return FeatureMappingNet(in_dim=in_dim, hidden_dim=512, out_dim=2)