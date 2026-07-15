"""
@Desc: Query Unlearning.
"""
import sys
import os
import shutil
import time
from typing import Any
from torch.nn import Module
from torch.optim import Optimizer
import torch
from torch import Tensor
import tqdm
import random
import numpy as np
from scipy.optimize import minimize
from defenses.victim.blackbox import Blackbox
import utils.operator as opt
from defenses import datasets
from torch.utils.data import DataLoader, TensorDataset
from defenses.models.mapping_net import feature_mapping_net
from utils.lossfunc import SupervisedContrastiveLoss
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils.classifier import ClassifierTrainer
from utils.data import DataFetcher
from defenses import config as CFG
from torch.autograd import detect_anomaly
from utils.stats import Timer


class Queen(Blackbox):
    def __init__(self, r: float, threshold: float, k: int,
                 in_dim: int, out_dim: int, num_layers: int, step_down: int,
                 shadow_arch: str, alpha: float = 1.0, beta: float = 1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        '''Set device'''
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        '''Hyperparameters'''
        self.alpha = alpha
        self.beta = beta
        self.r = r
        self.threshold = threshold
        self.k = k
        self.num_shadows = kwargs['num_shadows']
        assert self.k <= self.num_shadows, 'k must be less or equal to the number of shadow models!'
        self.host_network = kwargs['host_network']
        self.out_dir = kwargs['out_path']
        self.counter_within = 0
        self.counter_reverse = 0
        self.counter_record = 0
        self.counter_query = 0
        self.num_classes = kwargs['num_classes']
        self.cqs = torch.zeros(self.num_classes).to(self.device)
        self.record = {}
        for i in range(self.num_classes):
            self.record.update({i: None})

        # external shadow loading controls
        self.external_shadow_path = kwargs.get('shadow_path', None)
        self.external_shadow_arch = kwargs.get('external_shadow_arch', self.host_network)

        print(
            f'=> QUEEN hyperparams: alpha={self.alpha}, beta={self.beta}, '
            f'r={self.r}, threshold={self.threshold}, k={self.k}, '
            f'num_shadows={self.num_shadows}, '
            f'external_shadow_arch={self.external_shadow_arch}'
        )

        '''Check training features'''
        self.pth_feature_dataset = opt.os.join(self.model_dir, 'training_feats/training_feats.pt')
        if opt.os.pth_exist(self.pth_feature_dataset):
            print('=> Loading the existing training features...')
            to_load = torch.load(self.pth_feature_dataset, map_location='cpu')
            features, labels = to_load['features'], to_load['labels']
            self.feature_dataset = TensorDataset(features, labels)
            del to_load
        else:
            print('=> No training features. Extracting training features...')
            opt.os.mkdir(opt.os.get_dir(self.pth_feature_dataset))
            modelfamily = datasets.dataset_to_modelfamily[self.dataset_name]
            transform = datasets.modelfamily_to_transforms[modelfamily]['train']
            train_set = datasets.__dict__[self.dataset_name](train=True, transform=transform)
            bs = 256
            train_loader = DataLoader(train_set, bs, True)

            features, labels = None, None
            for x, y in tqdm.tqdm(train_loader, desc='Extracting training features...'):
                x = x.to(self.device)
                x = self.model.get_feats(x)
                x = x.detach().cpu()
                y = y.detach().cpu()
                features = opt.tensor.cat_tensors(features, x)
                labels = opt.tensor.cat_tensors(labels, y)

            self.feature_dataset = TensorDataset(features, labels)
            to_save = {'features': features, 'labels': labels}
            torch.save(to_save, self.pth_feature_dataset)
            print('=> Training features extracted and saved! Shape of the training features:', features.shape)
            del bs, modelfamily, transform, train_set, train_loader, to_save

        '''Get feature centers'''
        self.pth_feature_centers = opt.os.join(self.model_dir, 'feature_centers/feature_centers.pt')
        if opt.os.pth_exist(self.pth_feature_centers):
            print('=> Loading feature centers...')
            self.centers = torch.load(self.pth_feature_centers, map_location='cpu')
        else:
            self.centers = self.get_feat_centers(features, labels, self.num_classes)
            opt.os.mkdir(opt.os.get_dir(self.pth_feature_centers))
            torch.save(self.centers.detach().cpu(), self.pth_feature_centers)

        '''Check mapping network'''
        self.mapping_net = feature_mapping_net(self.host_network).to(self.device)
        self.pth_mapping_net = opt.os.join(self.model_dir, 'mapping_net/mapping_net.pt')
        if opt.os.pth_exist(self.pth_mapping_net):
            print('=> Loading the mapping net...')
            self.mapping_net.load_state_dict(torch.load(self.pth_mapping_net, map_location=self.device))
        else:
            print('=> No mapping net. Training a mapping net from scratches...')
            opt.os.mkdir(opt.os.get_dir(self.pth_mapping_net))
            self.mapping_net.train()
            print(self.mapping_net)

            num_ep = 100
            bs = 500
            lr = 0.0001
            step_size, gamma = 100, 1

            optimizer = torch.optim.SGD(
                self.mapping_net.parameters(), lr, momentum=0.9, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
            criteria = SupervisedContrastiveLoss()
            feat_loader = DataLoader(self.feature_dataset, bs, True)

            for ep in tqdm.trange(num_ep, desc='Training the mapping network...'):
                pbar = tqdm.tqdm(feat_loader)
                pbar.set_description('Ep: %d, Loss: nan' % ep)
                for xs, ys in pbar:
                    xs = xs.to(self.device)
                    ys = ys.to(self.device)

                    self.mapping_net.zero_grad()
                    optimizer.zero_grad()
                    xs = self.mapping_net(xs)
                    xs = xs.unsqueeze(1)
                    loss = criteria(xs, ys)

                    loss.backward()
                    optimizer.step()
                    pbar.set_description('Ep: %d, Loss: %.4f' % (ep, loss.item()))

                scheduler.step()

            '''Plot the 2D features'''
            pth_fig = opt.os.join(opt.os.get_dir(self.pth_mapping_net), 'feature_2d_map.png')
            with torch.no_grad():
                self.mapping_net.eval()
                feats_2d, labels = None, None
                feat_loader = DataLoader(self.feature_dataset, 1000, False)
                for xs, ys in feat_loader:
                    xs = xs.to(self.device)
                    xs = self.mapping_net(xs).detach().cpu()
                    ys = ys.detach().cpu()
                    feats_2d = opt.tensor.cat_tensors(feats_2d, xs)
                    labels = opt.tensor.cat_tensors(labels, ys)

            plt.figure(figsize=(8, 8))
            plt.scatter(feats_2d[:, 0], feats_2d[:, 1], s=1, c=labels)
            plt.savefig('%s' % pth_fig)
            plt.close()

            torch.save(self.mapping_net.state_dict(), self.pth_mapping_net)
            print('=> Mapping network trained and saved!')

            print('=> Saving the 2D features...')
            to_save = {'features': feats_2d, 'labels': labels}
            self.pth_features_2d = opt.os.join(self.model_dir, 'features_2d/features_2d.pt')
            opt.os.mkdir(opt.os.get_dir(self.pth_features_2d))
            torch.save(to_save, self.pth_features_2d)

        '''Sensitivity Analysis'''
        self.pth_sa_res = opt.os.join(self.model_dir, 'sensitivity_analysis/sensitivity_analysis.pt')
        if opt.os.pth_exist(self.pth_sa_res):
            print('=> Loading the sensitivity analysis results...')
            sa_res = torch.load(self.pth_sa_res, map_location='cpu')
            self.centers_2d = sa_res['centers']
            self.avgdist = sa_res['avgdist']
            del sa_res
        else:
            print('=> No sensitivity analysis results. Performing sensitivity analysis...')
            self.pth_features_2d = opt.os.join(self.model_dir, 'features_2d/features_2d.pt')
            assert opt.os.pth_exist(self.pth_features_2d), '2D features not exist!'
            this_dict = torch.load(self.pth_features_2d, map_location='cpu')
            feats_2d, labels = this_dict['features'], this_dict['labels']
            self.centers_2d, self.avgdist = self.sensitivity_analysis(feats_2d, labels, kwargs['num_classes'])
            sa_res_to_save = {'centers': self.centers_2d, 'avgdist': self.avgdist}
            opt.os.mkdir(opt.os.get_dir(self.pth_sa_res))
            torch.save(sa_res_to_save, self.pth_sa_res)

        '''Shadow models'''
        self.dir_shadow = opt.os.join(self.model_dir, 'shadow_models')
        if not opt.os.pth_exist(self.dir_shadow):
            opt.os.mkdir(self.dir_shadow)

        self.shadow_models = []
        self._init_shadow_models(shadow_arch)

        print('=> Queen initialization complete!')

    # =========================
    # shadow loading helpers
    # =========================
    def _load_single_shadow_state_dict(self, model_arch: str, state_dict):
        trainer = ClassifierTrainer(
            model_arch, self.num_classes, 'sgd', 'steplr', 0.01,
            'crossentropy', '', device=self.device
        )
        trainer.classifier.load_state_dict(state_dict)
        self.shadow_models.append(trainer.classifier)

    def _load_single_shadow(self, model_arch: str, pth: str):
        state = torch.load(pth, map_location=self.device, weights_only=False)

        # 兼容：
        # 1) checkpoint dict with state_dict / model_state_dict
        # 2) 直接保存的 nn.Module
        # 3) 直接保存的纯 state_dict
        if isinstance(state, dict) and ('state_dict' in state or 'model_state_dict' in state):
            state_dict = state.get('state_dict', state.get('model_state_dict'))
        elif hasattr(state, 'state_dict'):
            state_dict = state.state_dict()
        else:
            state_dict = state

        # 去掉 FeatureWrappedModel 前缀
        if isinstance(state_dict, dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('base_model.'):
                    new_k = k[len('base_model.'):]
                else:
                    new_k = k
                new_state_dict[new_k] = v
            state_dict = new_state_dict

        self._load_single_shadow_state_dict(model_arch, state_dict)

    def _discover_external_shadow_ckpts(self):
        """
        支持读取如下结构：
          shadow_path/shadow_0/checkpoint.pth.tar
          shadow_path/shadow_1/checkpoint.pth.tar
          ...
        """
        ckpts = []
        if self.external_shadow_path is None:
            return ckpts
        if not os.path.isdir(self.external_shadow_path):
            print(f'=> external shadow_path not found: {self.external_shadow_path}')
            return ckpts

        entries = sorted(os.listdir(self.external_shadow_path))
        for name in entries:
            subdir = os.path.join(self.external_shadow_path, name)
            ckpt = os.path.join(subdir, 'checkpoint.pth.tar')
            if os.path.isfile(ckpt):
                ckpts.append(ckpt)

        print(f'=> Discovered {len(ckpts)} external shadow checkpoint(s) from {self.external_shadow_path}')
        return ckpts

    def _train_single_shadow(self, shadow_arch: str, train_loader, shadow_idx: int):
        dir_temp_i = opt.os.join(self.dir_shadow, f'temp_{shadow_idx}')
        if opt.os.pth_exist(dir_temp_i):
            shutil.rmtree(dir_temp_i, ignore_errors=True)

        trainer = ClassifierTrainer(
            shadow_arch, self.num_classes, 'sgd', 'steplr', 0.01,
            'crossentropy', dir_temp_i, device=self.device
        )
        trainer.train_classifier(1, train_loader, None)

        pth_shadow = opt.os.join(self.dir_shadow, f'{shadow_arch}_{shadow_idx}_.pt')
        torch.save(trainer.classifier.state_dict(), pth_shadow)
        self.shadow_models.append(trainer.classifier)

        if opt.os.pth_exist(dir_temp_i):
            shutil.rmtree(dir_temp_i, ignore_errors=True)

    def _init_shadow_models(self, shadow_arch: str):
        loaded_paths = set()

        # 1) external shadows: use external_shadow_arch
        ext_ckpts = self._discover_external_shadow_ckpts()
        if len(ext_ckpts) > 0:
            num_to_load = min(len(ext_ckpts), self.num_shadows)
            print(f'=> Loading {num_to_load} external shadow model(s) with arch={self.external_shadow_arch}...')
            for i in range(num_to_load):
                self._load_single_shadow(self.external_shadow_arch, ext_ckpts[i])
                loaded_paths.add(ext_ckpts[i])

        # 2) local flat shadows: use shadow_arch
        if len(self.shadow_models) < self.num_shadows:
            flat_ckpts = sorted(opt.os.get_files(self.dir_shadow, '.pt'))
            flat_ckpts = [p for p in flat_ckpts if p not in loaded_paths]

            if len(flat_ckpts) > 0:
                num_can_load = min(len(flat_ckpts), self.num_shadows - len(self.shadow_models))
                print(f'=> Loading {num_can_load} local flat shadow model(s) from {self.dir_shadow} with arch={shadow_arch}...')
                for i in range(num_can_load):
                    self._load_single_shadow(shadow_arch, flat_ckpts[i])

        num_loaded = len(self.shadow_models)
        if num_loaded < self.num_shadows:
            num_to_train = self.num_shadows - num_loaded
            print(f'=> Need {self.num_shadows} shadow model(s), but only {num_loaded} loaded. '
                  f'Training {num_to_train} additional shadow model(s)...')

            t0 = time.time()
            modelfamily = datasets.dataset_to_modelfamily[self.dataset_name]
            transform = datasets.modelfamily_to_transforms[modelfamily]['train']
            train_set = datasets.__dict__[self.dataset_name](train=True, transform=transform)
            train_loader = DataLoader(train_set, 256, True)

            for i in range(num_loaded, self.num_shadows):
                self._train_single_shadow(shadow_arch, train_loader, i)

            runtime = time.time() - t0
            print('=> Time of training %d additional shadow model(s): %.2f seconds' % (num_to_train, runtime))

        if len(self.shadow_models) < self.k:
            raise RuntimeError(
                f'Not enough shadow models after initialization: loaded/trained {len(self.shadow_models)}, '
                f'but k={self.k}. Please check external shadow_path / shadow_models cache / num_shadows settings.'
            )

        print(f'=> Shadow models ready: {len(self.shadow_models)} loaded/trained, k={self.k}')

    @staticmethod
    def get_eu_dist(x: Tensor, y: Tensor) -> float:
        assert x.shape == y.shape
        if len(x.shape) == 2:
            eu_dist = torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=1))
        else:
            eu_dist = torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=0))
        return eu_dist

    def get_amct_avgdist(self, feats_2D: Tensor, labels: Tensor, label: int):
        feats_2D = feats_2D[labels == label]
        amct = feats_2D.mean(dim=0).unsqueeze(0)
        amct_ = amct.repeat(feats_2D.shape[0], 1)
        dist = torch.sort(self.get_eu_dist(feats_2D, amct_))[0]
        avg_dist = torch.mean(dist)
        return amct, avg_dist

    def sensitivity_analysis(self, feats_2D: Tensor, labels: Tensor, num_classes: int):
        centers, avgdist = {}, {}
        for label in range(num_classes):
            amct, avg_dist = self.get_amct_avgdist(feats_2D, labels, label)
            centers.update({label: amct})
            avgdist.update({label: avg_dist})
        return centers, avgdist

    @classmethod
    def get_fc_dist(self, feat2d: Tensor, center: Tensor) -> float:
        fc_dist = self.get_eu_dist(feat2d, center)
        return fc_dist

    @classmethod
    def in_sensitive_region(self, fc_dist: float, rs: float) -> bool:
        return fc_dist < rs

    @classmethod
    def no_previous_record(self, feat2d: Tensor, record: Tensor, r: float) -> bool:
        not_recorded = True
        if record is None:
            return not_recorded
        feat2d = feat2d.repeat(record.shape[0], 1)
        dist = self.get_eu_dist(feat2d, record)
        if not torch.all(dist > r):
            not_recorded = False
        return not_recorded

    @classmethod
    def get_sqs(self, dist: float, avg_dist: float, alpha: float) -> float:
        sqs = 0.5 * torch.erfc((alpha * (dist - avg_dist)) / avg_dist)
        return sqs

    @classmethod
    def update_cqs(self, sqs: float, cqs: Tensor, label: int, r: float, rs: float) -> Tensor:
        cqs[label] += sqs.item() ** 2 * (r / rs) ** 2
        return cqs

    @classmethod
    def cqs_over_threshold(self, cqs, label, threshold) -> bool:
        return cqs[label] > threshold

    @classmethod
    def get_pir_softmax(self, lst_shadows: list, q: Tensor, k: int) -> Tensor:
        if len(lst_shadows) < k:
            raise RuntimeError(
                f'get_pir_softmax received only {len(lst_shadows)} shadow model(s), but k={k}.'
            )

        lst_idx = list(range(len(lst_shadows)))
        lst_idx = random.sample(lst_idx, k)

        with torch.no_grad():
            y_soft_pir = None
            for idx in lst_idx:
                lst_shadows[idx].eval()
                y_soft = lst_shadows[idx](q)
                if y_soft_pir is None:
                    y_soft_pir = y_soft
                else:
                    y_soft_pir += y_soft
            y_soft_pir /= k

        return y_soft_pir

    @staticmethod
    def cosine_similarity_objective(x, y) -> float:
        cosine_similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return -cosine_similarity

    @staticmethod
    def constraint_function(x) -> float:
        return np.sum(x) - 1

    @staticmethod
    def get_feat_centers(feats: Tensor, labels: Tensor, n_classes: int) -> Tensor:
        centers = []
        for y in range(n_classes):
            cur_feats = feats[labels == y]
            cur_center = torch.mean(cur_feats, 0)
            centers.append(cur_center.detach().cpu().numpy())
        centers = Tensor(np.array(centers))
        return centers

    @staticmethod
    def make_hard_wrong_softmax(num_classes: int, wrong_label: int, true_label: int = None,
                                main_prob: float = 0.84, device: str = 'cpu') -> Tensor:
        probs = torch.full((num_classes,), (1.0 - main_prob) / (num_classes - 1), device=device)
        probs[wrong_label] = main_prob

        if true_label is not None and true_label != wrong_label:
            spill = probs[true_label].item()
            probs[true_label] = 1e-6
            redistribute_idx = [i for i in range(num_classes) if i not in [wrong_label, true_label]]
            if len(redistribute_idx) > 0:
                probs[redistribute_idx] += spill / len(redistribute_idx)

        probs = probs / probs.sum()
        return probs

    @torch.no_grad()
    def get_farthest_label(self, feat: Tensor, feat_centers: Tensor) -> int:
        feat = feat.to(self.device)
        feat_centers = feat_centers.to(self.device)
        feat_rep = feat.unsqueeze(0).repeat(feat_centers.shape[0], 1)
        dist = self.get_eu_dist(feat_rep, feat_centers)
        idx = torch.argmax(dist).item()
        return idx

    def gen_falsified_softmax(self, y_target: Tensor) -> Tensor:
        y_target = y_target.detach().cpu().numpy()
        y_fal = np.random.rand(len(y_target))
        bounds = [(0, 1) for _ in y_fal]
        constraint = {'type': 'eq', 'fun': self.constraint_function}
        result = minimize(
            self.cosine_similarity_objective, y_fal, args=(y_target,),
            method='SLSQP', constraints=constraint, bounds=bounds
        )
        y_fal = Tensor(result.x)
        return y_fal

    def falsify_gradient(self, y_soft_pro: Tensor, y_soft_pir: Tensor) -> Tensor:
        if y_soft_pro.dim() == 1:
            y_soft_pro = y_soft_pro.unsqueeze(0)
        if y_soft_pir.dim() == 1:
            y_soft_pir = y_soft_pir.unsqueeze(0)

        out = torch.zeros_like(y_soft_pro)

        for i in range(y_soft_pro.shape[0]):
            true_label = torch.argmax(y_soft_pro[i]).item()

            pirate_rank = torch.argsort(y_soft_pir[i], descending=True).tolist()
            wrong_label = None
            for c in pirate_rank:
                if c != true_label:
                    wrong_label = c
                    break

            if wrong_label is None:
                wrong_label = (true_label + 1) % y_soft_pro.shape[1]

            out[i] = self.make_hard_wrong_softmax(
                num_classes=y_soft_pro.shape[1],
                wrong_label=wrong_label,
                true_label=true_label,
                main_prob=0.84,
                device=y_soft_pro.device
            )

        return out

    def find_farthest_center(self, feat: Tensor, feat_centers: Tensor) -> Tensor:
        feat, feat_centers = feat.to(self.device), feat_centers.to(self.device)
        feat = feat.unsqueeze(0).repeat(feat_centers.shape[0], 1)
        dist = self.get_eu_dist(feat, feat_centers)
        idx = torch.argmax(dist)
        return feat_centers[idx]

    @torch.no_grad()
    def perturb_sm(self, feat: Tensor, center: Tensor, net: Module, step_size: float = 0.05) -> Tensor:
        cur_sm = net.feat_to_sm(feat)
        cur_sm = F.softmax(cur_sm, dim=1)
        true_label = torch.argmax(cur_sm, dim=1).item()

        target_label = self.get_farthest_label(feat.squeeze(0), self.centers)
        if target_label == true_label:
            target_label = (true_label + 1) % self.num_classes

        wrong_sm = self.make_hard_wrong_softmax(
            num_classes=self.num_classes,
            wrong_label=target_label,
            true_label=true_label,
            main_prob=0.80,
            device=feat.device
        ).unsqueeze(0)

        mix_lambda = 0.87
        mixed_sm = mix_lambda * wrong_sm + (1.0 - mix_lambda) * cur_sm
        mixed_sm = mixed_sm / mixed_sm.sum(dim=1, keepdim=True)

        return mixed_sm

    @torch.no_grad()
    def __call__(self, query_input: Tensor, stat: bool = True, return_origin: bool = False):
        self.model.eval()
        self.mapping_net.eval()
        self.counter_query += query_input.shape[0]

        query_input = query_input.to(self.device)
        feats = self.model.get_feats(query_input)
        feats2d = self.mapping_net(feats)
        softmax_protectee = F.softmax(self.model(query_input), 1)
        labels_protectee = torch.argmax(softmax_protectee, 1)

        softmax_falsified = torch.zeros_like(softmax_protectee)

        for i in tqdm.trange(feats2d.shape[0], desc='Queen Processing...'):
            feat2d = feats2d[i].unsqueeze(0)
            label = labels_protectee[i].item()
            center, avgdist = self.centers_2d[label].to(self.device), self.avgdist[label].to(self.device)
            cur_record = self.record[label]
            rs = avgdist * self.beta

            sensitive = False
            fc_dist = self.get_fc_dist(feat2d, center)
            if self.in_sensitive_region(fc_dist, rs):
                self.counter_within += 1
                if self.no_previous_record(feat2d, cur_record, self.r):
                    sensitive = True

            if sensitive:
                cur_softmax_protectee = softmax_protectee[i]
                if self.cqs_over_threshold(self.cqs, label, self.threshold):
                    query = query_input[i].unsqueeze(0)
                    cur_softmax_pirate = self.get_pir_softmax(self.shadow_models, query, self.k)
                    cur_softmax_falsified = self.falsify_gradient(cur_softmax_protectee, cur_softmax_pirate)
                    softmax_falsified[i] = cur_softmax_falsified.squeeze(0)
                    self.counter_reverse += 1
                else:
                    self.record[label] = opt.tensor.cat_tensors(self.record[label], feat2d)
                    self.counter_record += 1
                    sqs = self.get_sqs(fc_dist, rs, self.alpha)
                    self.cqs = self.update_cqs(sqs, self.cqs, label, self.r, rs)
                    feat = feats[i]
                    farthest_center = self.find_farthest_center(feat, self.centers)
                    cur_softmax_falsified = self.perturb_sm(feat.unsqueeze(0), farthest_center, self.model)
                    softmax_falsified[i] = cur_softmax_falsified.squeeze(0)
            else:
                feat = feats[i]
                farthest_center = self.find_farthest_center(feat, self.centers)
                cur_softmax_falsified = self.perturb_sm(feat.unsqueeze(0), farthest_center, self.model)
                softmax_falsified[i] = cur_softmax_falsified.squeeze(0)

        self.call_count += query_input.shape[0]

        std_out = '====== Cumulative Report: Query: %d, Within: %d, Recorded: %d, Reversed: %d ======\n' % (
            self.counter_query, self.counter_within, self.counter_record, self.counter_reverse
        )
        with open(opt.os.join(self.out_dir, 'queen_log.txt'), 'a') as f:
            f.write(std_out)

        if return_origin:
            return softmax_falsified, softmax_protectee
        return softmax_falsified


def debug():
    defender_queen = Queen(
        protectee_arch='res34',
        n_classes=10,
        alpha=0.5,
        beta=1.0,
        r=0.02,
        threshold=0.5,
        k=5,
        pth_protectee_net_ckpt='exp/20230904/cls_ckpt/res34_mnist.pt',
        in_dim=512,
        out_dim=2,
        num_layers=4,
        step_down=4,
        pth_mapping_net_ckpt='exp/20230904/map_net_ckpt/map_net_res34_mnist.pt',
        dir_shadow='exp/20230904/shadow_ckpt',
        shadow_arch='res18',
        pth_sensitive_analysis='exp/20230904/sensitivity_analysis/sa_res_res34_mnist.pt',
        pth_feature_centers='exp/20230904/training_feats/feats_res34_mnist.pt.feat_centers.pt',
    )

    X = torch.rand((100, 3, 32, 32)).cuda()
    return


if __name__ == '__main__':
    debug()