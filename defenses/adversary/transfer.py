#!/usr/bin/python
"""Construct transfer set for model extraction attacks."""
import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from defenses import datasets
import defenses.utils.utils as knockoff_utils
import defenses.config as cfg
from defenses.victim import *
from defenses.adversary.wb_recover import Table_Recover
from defenses.utils.utils import samples_to_transferset


def summarize_tensor_probs(Y: torch.Tensor, name: str, out_path: str = None):
    with torch.no_grad():
        row_sum = Y.sum(dim=1)
        Y_safe = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum_safe = torch.nan_to_num(row_sum, nan=0.0, posinf=0.0, neginf=0.0)

        stats = {
            'name': name,
            'shape': list(Y.shape),
            'has_nan': bool(torch.isnan(Y).any().item()),
            'has_inf': bool(torch.isinf(Y).any().item()),
            'min': float(Y_safe.min().item()),
            'max': float(Y_safe.max().item()),
            'neg_ratio': float((Y < 0).float().mean().item()),
            'row_sum_min': float(row_sum_safe.min().item()),
            'row_sum_max': float(row_sum_safe.max().item()),
            'row_sum_mean': float(row_sum_safe.mean().item()),
            'argmax_hist': torch.bincount(torch.argmax(Y_safe, dim=1), minlength=Y.shape[1]).tolist(),
        }

        print(f'===== Transfer label summary: {name} =====')
        for k, v in stats.items():
            if k != 'argmax_hist':
                print(f'{k}: {v}')
        print(f'argmax_hist(first 20): {stats["argmax_hist"][:20]}')

        if out_path is not None:
            with open(out_path, 'w') as f:
                json.dump(stats, f, indent=2)

    return stats


class RandomAdversaryIters(object):
    def __init__(self, blackbox, label_recover, queryset, batch_size=8, hard_label=False):
        self.blackbox = blackbox
        self.label_recover = label_recover
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.hard_label = hard_label
        self.idx_set = set()
        self.budget_idxs = set()

        self.transferset = []
        self.call_times = []

        self._restart()

    def _get_idxset(self, budget):
        prev_state = np.random.get_state()
        np.random.seed(cfg.DEFAULT_SEED)
        self.idx_set = np.arange(len(self.queryset))
        np.random.shuffle(self.idx_set)
        self.idx_set = set(self.idx_set[:budget])
        np.random.set_state(prev_state)

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.all_idxs = np.arange(len(self.queryset))
        np.random.shuffle(self.all_idxs)

        self.transferset = []
        self.call_times = []

    def get_transferset(self, budget, niters=None, queries_per_image=1, only_recovery=False):
        if niters is None:
            niters = budget
        start_B = 0
        end_B = niters
        self.idx_set = self.all_idxs[:budget]

        stat = (self.label_recover is not None or queries_per_image >= 2) and (self.blackbox.log_path is not None)
        if stat:
            idx_file = self.blackbox.log_path.replace('distancetransfer.log.tsv', 'idx.pt')
            log_path = self.blackbox.log_path.replace('distance', 'gtdistance')
            if not osp.exists(log_path):
                with open(log_path, 'w') as wf:
                    columns = ['transferset size', 'l1_max', 'l1_mean', 'l1_std', 'l2_mean', 'l2_std', 'kl_mean', 'kl_std']
                    wf.write('\t'.join(columns) + '\n')

        print('Constructing transferset using: # unique images = {}, # queries = {}'.format(len(self.idx_set), niters))
        assert niters <= budget, "Query number cannot be larger than budget!"

        if queries_per_image > 1:
            print('=> Obtaining mean posteriors over {} predictions per image'.format(queries_per_image))

        y_t_list = []
        IMG, Y_true = [], []
        if self.label_recover is not None:
            X = []
            if only_recovery:
                queries_per_image = 1

        for q in range(queries_per_image):
            num_queries = 0
            Y_t = []
            with tqdm(total=niters) as pbar:
                for B in range(start_B, end_B, self.batch_size):
                    idxs = self.idx_set[np.arange(B, min(B + self.batch_size, end_B))]
                    num_queries += len(idxs)

                    x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.blackbox.device)

                    t_start = time.time()
                    if q == 0:
                        if only_recovery:
                            y_t = torch.stack([self.queryset[i][1] for i in idxs]).to(self.blackbox.device)
                            y_t_true = F.softmax(self.blackbox.model(x_t), dim=1).detach()
                        else:
                            y_t, y_t_true = self.blackbox(x_t, return_origin=True)
                        t_end = time.time()
                        self.call_times.append(t_end - t_start)
                        if hasattr(self.queryset, 'samples'):
                            img_t = [self.queryset.samples[i][0] for i in idxs]
                        else:
                            img_t = [self.queryset.data[i] for i in idxs]
                            if isinstance(self.queryset.data[0], torch.Tensor):
                                img_t = [x.numpy() for x in img_t]

                        IMG = IMG + img_t
                        Y_true.append(y_t_true.detach().cpu())
                        if self.label_recover is not None:
                            X.append(x_t.detach().cpu())
                    else:
                        y_t = self.blackbox(x_t)
                        t_end = time.time()
                        self.call_times.append(t_end - t_start)

                    Y_t.append(y_t)
                    pbar.update(x_t.size(0))
            y_t_list.append(torch.cat(Y_t, dim=0))

        Y = torch.stack(y_t_list).mean(dim=0)
        Y_true = torch.cat(Y_true, dim=0)
        if self.label_recover is not None:
            X = torch.cat(X, dim=0)

        summary_path_pre = osp.join(self.blackbox.out_path, 'transfer_label_summary_pre_recover.json')
        summarize_tensor_probs(
            Y,
            'Y_before_recover' if self.label_recover is not None else 'Y_final_no_recover',
            summary_path_pre
        )

        if queries_per_image >= 2:
            dist = torch.norm(y_t_list[0] - Y, p=2, dim=1)
            if osp.exists(idx_file):
                change_idx = torch.load(idx_file)
            else:
                change_idx = torch.arange(len(dist), device=dist.device)[dist > 1e-3]
            Y_mean = torch.stack([y_t_list[0][change_idx], y_t_list[1][change_idx]]).mean(dim=0)
            if stat:
                torch.save(change_idx, idx_file)
                l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.blackbox.calc_query_distances(
                    [[Y_true[change_idx.to(device=Y_true.device)], y_t_list[0].to(Y_true)[change_idx.to(device=Y_true.device)]]]
                )
                with open(log_path, 'a') as af:
                    test_cols = ["[orig]:%d" % len(change_idx), l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')
                l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.blackbox.calc_query_distances(
                    [[Y_true[change_idx.to(device=Y_true.device)], Y_mean.to(Y_true)]]
                )
                with open(log_path, 'a') as af:
                    test_cols = ["[mean]:%d" % len(change_idx), l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        if self.label_recover is not None:
            print("=> Start to recover the clean labels!")
            self.label_recover.generate_lookup_table(
                load_path=osp.join(self.blackbox.out_path, 'recover_table.pickle'),
                estimation_set=[X, Y_true]
            )
            with tqdm(total=len(Y)) as pbar:
                Y = self.label_recover(Y, pbar)
                Y = Y.to(Y_true)

            summary_path_post = osp.join(self.blackbox.out_path, 'transfer_label_summary_post_recover.json')
            summarize_tensor_probs(Y, 'Y_after_recover', summary_path_post)

            if stat:
                l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = self.blackbox.calc_query_distances([[Y_true, Y]])
                with open(log_path, 'a') as af:
                    test_cols = [num_queries, l1_max, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        if self.hard_label:
            topk_vals, indices = torch.topk(Y, 1)
            Y = torch.zeros_like(Y)
            Y = Y.scatter(1, indices, torch.ones_like(topk_vals))

        debug_samples = {
            'Y_first20': Y[:20].detach().cpu(),
            'Ytrue_first20': Y_true[:20].detach().cpu(),
        }
        torch.save(debug_samples, osp.join(self.blackbox.out_path, 'transfer_label_debug_first20.pt'))

        for i in range(len(IMG)):
            img_i = IMG[i].squeeze() if isinstance(IMG[i], np.ndarray) else IMG[i]
            self.transferset.append((img_i, Y[i].cpu().squeeze()))

        return self.transferset


def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training', choices=['random', 'adaptive'])
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('defense', metavar='TYPE', type=str, help='Type of defense to use',
                        choices=knockoff_utils.BBOX_CHOICES, default='none')
    parser.add_argument('defense_args', metavar='STR', type=str, help='Blackbox arguments in format "k1:v1;k2:v2;..."')

    parser.add_argument('--defense_aware', type=int, help="Whether using defense-aware attack", default=0)
    parser.add_argument('--recover_args', type=str, help='Recover arguments in format "k1:v1;k2:v2;..."')
    parser.add_argument('--hardlabel', type=int, help="Whether only use hard label for extraction", default=0)

    parser.add_argument('--quantize', type=int, help="Whether using quantized defense", default=0)
    parser.add_argument('--quantize_args', type=str, help='Quantization arguments in format "k1:v1;k2:v2;..."')

    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budget', metavar='N', type=int, help='# images', default=None)
    parser.add_argument('--nqueries', metavar='N', type=int, help='# queries to blackbox using budget images', default=None)
    parser.add_argument('--qpi', metavar='N', type=int, help='# queries per image', default=1)
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary dataset', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=1)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_transform', type=int, help='Whether to perform data augmentation while querying', default=0)
    parser.add_argument('--train_transform_blur', type=int, help='Whether to perform data Gaussian Blur augmentation', default=0)
    parser.add_argument('--only_recovery', action='store_true', help='Perform prediction recovery only', default=False)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    queryset_name = params['queryset']
    valid_datasets = datasets.__dict__.keys()
    if queryset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[queryset_name]
    if params['train_transform']:
        print('=> Using data augmentation while querying')
        transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    elif params['train_transform_blur']:
        print('=> Using data Guassian Blur augmentation while querying')
        transform = datasets.modelfamily_to_transforms_blur[modelfamily]['train']
    else:
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']

    only_recovery = params['only_recovery'] and osp.exists(osp.join(out_path, 'transferset.pickle'))
    if only_recovery:
        print("=> only recovery begins!")
        transferset_path = osp.join(out_path, 'transferset.pickle')
        try:
            with open(transferset_path, 'rb') as rf:
                transferset_samples = torch.load(rf)
        except RuntimeError:
            with open(transferset_path, 'rb') as rf:
                transferset_samples = pickle.load(rf)
        num_classes = transferset_samples[0][1].size(0)
        queryset = samples_to_transferset(transferset_samples, budget=len(transferset_samples), transform=transform)
        print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))
    else:
        queryset = datasets.__dict__[queryset_name](train=True, transform=transform)

    if params['budget'] is None:
        params['budget'] = len(queryset)

    blackbox_dir = params['victim_model_dir']
    defense_type = params['defense']
    if defense_type == 'rand_noise':
        BB = RandomNoise
    elif defense_type == 'rand_noise_wb':
        BB = RandomNoise_WB
    elif defense_type == 'mad':
        BB = MAD
    elif defense_type == 'mad_wb':
        BB = MAD_WB
    elif defense_type == 'mld':
        BB = MLD
    elif defense_type == "am":
        BB = AM
    elif defense_type == 'reverse_sigmoid':
        BB = ReverseSigmoid
    elif defense_type == 'reverse_sigmoid_wb':
        BB = ReverseSigmoid_WB
    elif defense_type in ['none', 'topk', 'rounding']:
        BB = Blackbox
    elif defense_type == 'queen':
        BB = Queen
    elif defense_type == 'dp':
        BB = DP
    else:
        raise ValueError('Unrecognized blackbox type')

    defense_kwargs = knockoff_utils.parse_defense_kwargs(params['defense_args'])
    defense_kwargs['log_prefix'] = 'transfer'
    print('=> Initializing BBox with defense {} and arguments: {}'.format(defense_type, defense_kwargs))
    blackbox = BB.from_modeldir(blackbox_dir, device, **defense_kwargs)

    quantize_blackbox = None
    if params['quantize']:
        quantize_kwargs = knockoff_utils.parse_defense_kwargs(params['quantize_args'])
        if quantize_kwargs['epsilon'] > 0.0:
            print('=> Initializing Quantizer with arguments: {}'.format(quantize_kwargs))
            quantize_blackbox = incremental_kmeans(blackbox, **quantize_kwargs)

    if params['defense_aware']:
        recover_kwargs = knockoff_utils.parse_defense_kwargs(params['recover_args'])
        print('=> Initializing Label Recovery with arguments: {}'.format(recover_kwargs))
        recover = Table_Recover(
            quantize_blackbox if quantize_blackbox is not None else blackbox,
            batch_size=params['batch_size'],
            epsilon=quantize_kwargs['epsilon'] if (params['quantize'] and quantize_kwargs['epsilon'] > 0.0) else None,
            recover_mean=True,
            **recover_kwargs
        )
    else:
        recover = None

    for k, v in defense_kwargs.items():
        params[k] = v

    batch_size = params['batch_size']
    if params['policy'] == 'random':
        adversary = RandomAdversaryIters(
            quantize_blackbox if quantize_blackbox is not None else blackbox,
            recover,
            queryset,
            batch_size=batch_size,
            hard_label=params['hardlabel']
        )
    elif params['policy'] == 'adaptive':
        raise NotImplementedError()
    else:
        raise ValueError("Unrecognized policy")

    print('=> constructing transfer set...')
    transferset = adversary.get_transferset(params['budget'], params['nqueries'],
                                            queries_per_image=params['qpi'], only_recovery=only_recovery)
    transfer_out_path = osp.join(out_path, 'transferset.pickle')
    with open(transfer_out_path, 'wb') as wf:
        torch.save(transferset, wf)
    print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

    rt_mean, rt_std = np.mean(adversary.call_times), np.std(adversary.call_times)
    rt_dict = {'batch_size': batch_size, 'mean': rt_mean, 'std': rt_std}
    rt_out_path = osp.join(out_path, 'run_times_summary.json')
    with open(rt_out_path, 'w') as wf:
        json.dump(rt_dict, wf, indent=True)
    rt_out_path = osp.join(out_path, 'run_times.pickle')
    with open(rt_out_path, 'wb') as wf:
        pickle.dump(adversary.call_times, wf)
    print('=> run time (Batch size = {}): mean = {}\t std = {}'.format(batch_size, rt_mean, rt_std))

    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params_transfer.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()