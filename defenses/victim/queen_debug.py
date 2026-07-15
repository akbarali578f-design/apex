import sys
import shutil
import time
import torch
from torch import Tensor
import tqdm
import random
from defenses.victim.queen import Queen
from defenses.models import zoo
import utils.operator as opt
import os

def debug(num_queries=10, num_shadows=20):
    print("===== Starting Enhanced Queen Debug =====")

    # Create the backbone model and pass it to Queen
    model = zoo.get_net('resnet34', num_classes=10, device='cuda:0')

    # Ensure log directory exists before initializing Queen
    log_dir = 'exp/20230904'
    os.makedirs(log_dir, exist_ok=True)

    defender_queen = Queen(
        model=model,
        r=0.02,
        threshold=0.5,
        k=5,
        num_shadows=num_shadows,
        in_dim=512,
        out_dim=2,
        num_layers=4,
        step_down=4,
        shadow_arch='res18',
        alpha=0.5,
        beta=1.0,
        num_classes=10,
        host_network='res34',
        out_path=log_dir,
        model_dir=log_dir,  # Explicitly pass model_dir to avoid NoneType
        pth_protectee_net_ckpt='exp/20230904/cls_ckpt/res34_mnist.pt',
        pth_mapping_net_ckpt='exp/20230904/map_net_ckpt/map_net_res34_mnist.pt',
        dir_shadow='exp/20230904/shadow_ckpt',
        pth_sensitive_analysis='exp/20230904/sensitivity_analysis/sa_res_res34_mnist.pt',
        pth_feature_centers='exp/20230904/training_feats/feats_res34_mnist.pt.feat_centers.pt'
    )

    print("===== Queen initialized =====")
    print(f"Number of shadow models: {len(defender_queen.shadow_models)}")
    print(f"Feature dimension: {defender_queen.feature_dataset.tensors[0].shape[1]}")

    # Generate random queries
    X = torch.rand((num_queries, 3, 32, 32)).to(defender_queen.device)
    softmax_falsified, softmax_protectee = defender_queen(X, return_origin=True)

    # Loop through each query and print detailed debug info
    for i in range(num_queries):
        feat2d = defender_queen.mapping_net(defender_queen.model.get_feats(X[i].unsqueeze(0)))
        label = torch.argmax(softmax_protectee[i]).item()
        fc_dist = defender_queen.get_fc_dist(feat2d, defender_queen.centers_2d[label].to(defender_queen.device))
        in_region = defender_queen.in_sensitive_region(fc_dist, defender_queen.avgdist[label].to(defender_queen.device))
        is_ood = defender_queen.is_ood_query(fc_dist, label)
        sqs = defender_queen.get_sqs(fc_dist, defender_queen.avgdist[label].to(defender_queen.device), defender_queen.alpha)
        cqs_value = defender_queen.cqs[label].item()

        print(f"\n--- Query {i} ---")
        print(f"Softmax Protectee: {softmax_protectee[i]}")
        print(f"Softmax Falsified: {softmax_falsified[i]}")
        print(f"Top1 label: {label}, ID/OOD: {'OOD' if is_ood else 'ID'}, In Sensitive Region: {in_region}")
        print(f"SQS: {sqs.item():.4f}, CQS: {cqs_value:.4f}")
        print(f"Attack mode: {defender_queen.attack_mode}")
        print(f"Counters -> Query: {defender_queen.counter_query}, ID: {defender_queen.counter_id}, OOD: {defender_queen.counter_ood}, AttackWrong: {defender_queen.counter_attack_wrong}, Poisoned: {defender_queen.counter_poison}, CleanTop1: {defender_queen.counter_clean}")

    print("===== Enhanced Queen debug finished =====")
    return softmax_falsified, softmax_protectee


if __name__ == '__main__':
    debug(num_queries=10, num_shadows=20)