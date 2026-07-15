import csv
import json
import os
import os.path as osp
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
from torch.distributions import Dirichlet
from torch.multiprocessing import Manager, Process
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from defenses import datasets
import defenses.models.zoo as zoo


numclasses_to_nn = {
    10: [128, 64],
    43: [512, 256],
    100: [1024, 512],
    200: [2048, 1024],
    256: [2048, 1024],
}


def _hidden_layers_for_num_classes(num_classes):
    num_classes = int(num_classes)
    if num_classes in numclasses_to_nn:
        return numclasses_to_nn[num_classes]
    # Safe fallback for future datasets.
    h1 = max(128, min(4096, num_classes * 16))
    h2 = max(64, min(2048, num_classes * 8))
    return [h1, h2]


class Recover_NN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = int(num_classes)
        hidden_layer = _hidden_layers_for_num_classes(self.num_classes)
        self.fc1 = nn.Linear(self.num_classes, hidden_layer[0])
        self.fc2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.fc3 = nn.Linear(hidden_layer[1], self.num_classes)

        self.bn1 = nn.BatchNorm1d(hidden_layer[0])
        self.bn2 = nn.BatchNorm1d(hidden_layer[1])
        self.bn3 = nn.BatchNorm1d(self.num_classes)

    def forward(self, x):
        if x.dim() != 2:
            raise RuntimeError("Recover_NN expects 2D input, got shape={}".format(tuple(x.shape)))
        expected_dim = int(self.fc1.in_features)
        if int(x.size(1)) != expected_dim:
            raise RuntimeError(
                "Recover_NN input width mismatch: got {}, expected {}. "
                "This usually means a stale CIFAR10 recover table is being reused in a CIFAR100 run. "
                "Delete recover_table.pickle/recover_nn.pt/transferset.pickle and rerun.".format(
                    int(x.size(1)), expected_dim
                )
            )
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        return F.softmax(x, dim=1)


class Table_Recover:
    max_sample_size = 5000000

    def __init__(
        self,
        blackbox,
        table_size=10000,
        batch_size=1,
        epsilon=None,
        perturb_norm=1,
        recover_mean=True,
        recover_norm=2,
        tolerance=1e-4,
        concentration_factor=4.0,
        shadow_path=None,
        recover_nn=False,
        recover_proc=1,
    ):
        self.table_size = int(table_size)
        self.blackbox = blackbox
        self.num_classes = self._infer_num_classes(blackbox)
        self.device = self.blackbox.device
        self.batch_size = int(batch_size)
        self.epsilon = epsilon
        self.perturb_norm = perturb_norm
        self.top1_recover = bool(getattr(self.blackbox, "top1_preserve", False))
        self.recover_mean = recover_mean
        self.recover_norm = recover_norm
        self.tolerance = tolerance
        self.concentration_factor = concentration_factor
        self.shadow_generate = bool(shadow_path is not None and osp.exists(shadow_path))
        self.shadow_path = shadow_path if self.shadow_generate else None
        self.recover_nn = bool(recover_nn)
        self.num_proc = max(1, int(recover_proc))
        self.true_label_sample = None
        self.perturbed_label_sample = None

        os.makedirs(self.blackbox.out_path, exist_ok=True)
        if not self.recover_nn:
            self.log_path = osp.join(self.blackbox.out_path, "recover_distance{}.log.tsv".format(self.blackbox.log_prefix))
            self.logger = csv.writer(open(self.log_path, "a"), delimiter="\t")
            self.logger.writerow(["call count", "recover distance mean", "recover distance std"])
            self.call_count = 0
        else:
            self.log_path = osp.join(self.blackbox.out_path, "recover_nn_training.log.tsv")
            self.logger = csv.writer(open(self.log_path, "a"), delimiter="\t")
            self.logger.writerow(["Epoch", "Loss", "L2 Distance"])

    @staticmethod
    def _infer_num_classes(blackbox):
        """Infer class count robustly for CIFAR10/CIFAR100 recovery.

        J-group CIFAR100 D-DAE can fail if a legacy CIFAR10 default leaks into
        the recovery table. Prefer explicit job environment variables, then
        dataset/model-family hints, then blackbox/model attributes.
        """
        env_keys = ["NUM_CLASSES", "NUM_CLASS", "N_CLASSES"]
        for key in env_keys:
            value = os.environ.get(key)
            if value is not None:
                try:
                    value = int(value)
                    if value > 1:
                        return value
                except Exception:
                    pass

        dataset_hint = " ".join(
            str(os.environ.get(k, ""))
            for k in ["DATASET", "MODEL_FAMILY", "MODELFAMILY", "MODelfamily"]
        ).lower()
        if "cifar100" in dataset_hint:
            return 100
        if "cifar10" in dataset_hint:
            return 10
        if "gtsrb" in dataset_hint:
            return 43
        if "cubs" in dataset_hint or "cub" in dataset_hint:
            return 200
        if "caltech256" in dataset_hint:
            return 256

        candidates = [
            getattr(blackbox, "num_classes", None),
            getattr(getattr(blackbox, "model", None), "num_classes", None),
            getattr(getattr(blackbox, "model", None), "n_classes", None),
        ]

        model = getattr(blackbox, "model", None)
        for attr in ["fc", "linear", "classifier"]:
            layer = getattr(model, attr, None)
            if hasattr(layer, "out_features"):
                candidates.append(layer.out_features)
            elif isinstance(layer, nn.Sequential) and len(layer) > 0:
                last = layer[-1]
                if hasattr(last, "out_features"):
                    candidates.append(last.out_features)

        for value in candidates:
            if value is not None:
                try:
                    value = int(value)
                    if value > 1:
                        return value
                except Exception:
                    pass
        raise RuntimeError(
            "Cannot infer num_classes from environment or blackbox. "
            "Please export NUM_CLASSES=100 for CIFAR100."
        )

    def _validate_label_matrix(self, name, tensor, allow_none=False):
        if tensor is None:
            if allow_none:
                return None
            raise RuntimeError("{} is None".format(name))
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor)
        if tensor.dim() != 2:
            raise RuntimeError("{} must be 2D, got shape={}".format(name, tuple(tensor.shape)))
        if int(tensor.size(1)) != int(self.num_classes):
            raise RuntimeError(
                "{} width mismatch: expected {}, got {}. "
                "For CIFAR100 D-DAE this must be [N,100], not [N,10].".format(
                    name, int(self.num_classes), int(tensor.size(1))
                )
            )
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise RuntimeError("{} contains NaN or Inf".format(name))
        return tensor.float()

    def _load_existing_table_if_valid(self, load_path, table_size):
        if load_path is None or not osp.exists(load_path):
            return table_size
        try:
            with open(load_path, "rb") as wf:
                true_sample, pert_sample = pickle.load(wf)
            true_sample = self._validate_label_matrix("loaded true_label_sample", true_sample)
            pert_sample = self._validate_label_matrix("loaded perturbed_label_sample", pert_sample)
        except Exception as exc:
            backup = load_path + ".stale_or_invalid_{}".format(int(time.time()))
            try:
                os.rename(load_path, backup)
                print("[WARN] Ignored invalid recover table and moved it to: {}".format(backup))
            except OSError:
                print("[WARN] Ignored invalid recover table: {}".format(load_path))
            print("[WARN] Reason: {}".format(exc))
            self.true_label_sample, self.perturbed_label_sample = None, None
            return table_size

        print("Loaded Existing Table with table length {}!".format(len(true_sample)))
        if len(true_sample) >= table_size:
            self.true_label_sample = true_sample[:table_size, :]
            self.perturbed_label_sample = pert_sample[:table_size, :]
            return 0

        self.true_label_sample = true_sample
        self.perturbed_label_sample = pert_sample
        print("Supplementing existing table with {} samples...".format(table_size - len(true_sample)))
        return table_size - len(true_sample)

    def generate_lookup_table(self, load_path=None, estimation_set=None, table_size=None, load_nn=False):
        if table_size is None:
            table_size = self.table_size
        table_size = int(table_size)

        print("[INFO] Table_Recover num_classes={}".format(int(self.num_classes)))
        table_size = self._load_existing_table_if_valid(load_path, table_size)

        if table_size > 0:
            print("Building Recover Table! Total Samples Number={}!".format(table_size))
            true_label_sample = []
            x_info_idxs = []
            estimation_input = None

            if self.shadow_generate:
                print("Use shadow models for generation!")
                assert estimation_set is not None, (
                    "The estimation set cannot be None when using shadow models for true prediction generation!"
                )
                estimation_input, _ = self.estimate_dir(estimation_set)
                shadow_true, shadow_xidx = self._generate_shadow_true_labels(estimation_input, max_samples=table_size)
                if shadow_true is not None and len(shadow_true) > 0:
                    true_label_sample.append(shadow_true)
                    x_info_idxs += shadow_xidx
                    table_size -= len(shadow_true)

            if table_size > 0:
                alpha = None
                if estimation_set is not None:
                    estimation_input, estimation_label = self.estimate_dir(estimation_set)
                    estimation_label = self._sanitize_estimation_label(estimation_label)
                    if estimation_label is not None:
                        concentration = self.num_classes * self.concentration_factor
                        alpha = torch.clamp(estimation_label * concentration, min=1e-6)

                true_label_sample_dir, x_info_idxs_dir = self.get_dirichlet_samples(alpha, table_size)
                true_label_sample.append(true_label_sample_dir)
                x_info_idxs += x_info_idxs_dir

            if len(true_label_sample) == 0:
                raise RuntimeError("Failed to generate any true-label samples for recover table.")

            true_label_sample = torch.cat(true_label_sample, dim=0).float()
            true_label_sample = self._validate_label_matrix("true_label_sample", true_label_sample)
            if len(true_label_sample) > int(self.table_size):
                true_label_sample = true_label_sample[: int(self.table_size)]
                x_info_idxs = x_info_idxs[: int(self.table_size)]

            if not getattr(self.blackbox, "require_xinfo", False):
                estimation_input = None
                x_info_idxs = None

            if self.num_proc == 1:
                perturbed_label_sample = self.get_perturbed_label_sample(
                    self.blackbox, true_label_sample, estimation_input, x_info_idxs, self.batch_size
                )
            else:
                perturbed_label_sample = self.get_perturbed_label_sample_parallel(
                    self.blackbox, true_label_sample, estimation_input, x_info_idxs, self.num_proc
                )

            perturbed_label_sample = self._validate_label_matrix("perturbed_label_sample", perturbed_label_sample)

            if self.epsilon is not None:
                pert_norm = torch.norm(true_label_sample - perturbed_label_sample, p=self.perturb_norm, dim=1)
                keep = pert_norm <= self.epsilon
                true_label_sample = true_label_sample[keep]
                perturbed_label_sample = perturbed_label_sample[keep]

            if self.true_label_sample is None or self.perturbed_label_sample is None:
                self.true_label_sample = true_label_sample
                self.perturbed_label_sample = perturbed_label_sample
            else:
                self.true_label_sample = torch.cat(
                    [self.true_label_sample, true_label_sample.to(self.true_label_sample)], dim=0
                )
                self.perturbed_label_sample = torch.cat(
                    [self.perturbed_label_sample, perturbed_label_sample.to(self.perturbed_label_sample)], dim=0
                )

            self.true_label_sample = self._validate_label_matrix("final true_label_sample", self.true_label_sample)
            self.perturbed_label_sample = self._validate_label_matrix(
                "final perturbed_label_sample", self.perturbed_label_sample
            )

            print("Recover Table Completed!")
            with open(osp.join(self.blackbox.out_path, "recover_table.pickle"), "wb") as wf:
                pickle.dump([self.true_label_sample, self.perturbed_label_sample], wf)

        try:
            self.true_label_sample = self.true_label_sample.to(self.device)
            self.perturbed_label_sample = self.perturbed_label_sample.to(self.device)
        except Exception:
            print("[Warning]: Not enough GPU memory for storing the lookup table, will use cpu instead!")
            self.true_label_sample = self.true_label_sample.cpu()
            self.perturbed_label_sample = self.perturbed_label_sample.cpu()

        if not self.recover_nn:
            if self.top1_recover:
                self.true_top1 = torch.argmax(self.true_label_sample, dim=1).to(self.device)
        else:
            self.nn = Recover_NN(self.num_classes)
            print("Generative Model:")
            print(self.nn)
            self.nn.to(self.device)
            model_out_path = osp.join(self.blackbox.out_path, "recover_nn.pt")
            if osp.exists(model_out_path) and load_nn:
                print("Load existing generative model at " + model_out_path)
                state = torch.load(model_out_path, map_location=self.device)
                self.nn.load_state_dict(state)
            else:
                print("Training NN for Recovering!")
                self.nn = self.train_recover_nn(
                    self.nn,
                    self.perturbed_label_sample,
                    self.true_label_sample,
                    epoch=200,
                    batch_size=1024,
                    lr=1e-2,
                )
                torch.save(self.nn.state_dict(), model_out_path)

    def _generate_shadow_true_labels(self, estimation_input, max_samples):
        true_label_sample = []
        x_info_idxs = []
        max_samples = int(max_samples)
        if max_samples <= 0:
            return None, []

        for d in sorted(os.listdir(self.shadow_path)):
            shadow_dir = osp.join(self.shadow_path, d)
            ckpt_path = osp.join(shadow_dir, "checkpoint.pth.tar")
            params_dir = osp.join(shadow_dir, "params.json")
            if not ("shadow" in d and osp.isdir(shadow_dir) and osp.exists(ckpt_path) and osp.exists(params_dir)):
                continue
            with open(params_dir) as f:
                params = json.load(f)
            shadow_dataset = params.get("dataset")
            shadow_arch = params.get("model_arch")
            shadow_num_classes = int(params.get("num_classes", self.num_classes))
            if shadow_num_classes != self.num_classes:
                print(
                    "[WARN] Skip shadow model {} because num_classes={} but current num_classes={}".format(
                        shadow_dir, shadow_num_classes, self.num_classes
                    )
                )
                continue
            modelfamily = datasets.dataset_to_modelfamily[shadow_dataset]
            shadow_model = zoo.get_net(shadow_arch, modelfamily, ckpt_path, num_classes=shadow_num_classes)
            shadow_model.to(self.device)
            shadow_model.eval()
            with torch.no_grad():
                for i in range(0, len(estimation_input), self.batch_size):
                    x = estimation_input[i : min(i + self.batch_size, len(estimation_input))].to(self.device)
                    y = F.softmax(shadow_model(x), dim=1).detach().cpu()
                    true_label_sample.append(y)
                    x_info_idxs += list(range(i, min(i + self.batch_size, len(estimation_input))))
                    if sum(len(t) for t in true_label_sample) >= max_samples:
                        break
            del shadow_model
            if sum(len(t) for t in true_label_sample) >= max_samples:
                break

        if len(true_label_sample) == 0:
            print("[WARN] No compatible shadow model generated labels; fallback to Dirichlet samples.")
            return None, []

        true_label_sample = torch.cat(true_label_sample, dim=0)[:max_samples]
        x_info_idxs = x_info_idxs[: len(true_label_sample)]
        return true_label_sample, x_info_idxs

    def _sanitize_estimation_label(self, estimation_label):
        if estimation_label is None:
            return None
        if not torch.is_tensor(estimation_label):
            estimation_label = torch.tensor(estimation_label)
        if estimation_label.dim() != 2 or int(estimation_label.size(1)) != self.num_classes:
            print(
                "[WARN] Ignore estimation labels with shape {}; expected width {}. "
                "Dirichlet fallback will be uniform.".format(tuple(estimation_label.shape), self.num_classes)
            )
            return None
        estimation_label = estimation_label.float().to(self.device)
        estimation_label = torch.clamp(estimation_label, min=1e-8)
        estimation_label = estimation_label / estimation_label.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return estimation_label

    def estimate_dir(self, estimation_set):
        if isinstance(estimation_set, str) and osp.exists(estimation_set):
            print("Estimating Dirichlet Distribution via Labels in '{}'".format(estimation_set))
            with open(estimation_set, "rb") as wf:
                estimation_data = pickle.load(wf)
            estimation_input = torch.cat(
                [torch.tensor(estimation_data[i][0]).reshape([1, -1]) for i in range(len(estimation_data))], dim=0
            )
            estimation_label = torch.cat(
                [torch.tensor(estimation_data[i][1]).reshape([1, -1]) for i in range(len(estimation_data))], dim=0
            )
        else:
            try:
                estimation_input = estimation_set[0]
                estimation_label = estimation_set[1].clone().detach()
            except Exception as exc:
                raise RuntimeError("Not a valid estimation set form (must be a path or a list of tensors)") from exc
        return estimation_input, estimation_label.to(self.device)

    def get_dirichlet_samples(self, alpha=None, table_size=1000000):
        table_size = int(table_size)
        if table_size <= 0:
            return torch.empty(0, self.num_classes), []

        sample_list = []
        alpha_idxs = []
        if alpha is None:
            alpha = torch.ones(1, self.num_classes, device=self.device)
        elif not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        else:
            alpha = alpha.float().to(self.device)

        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        if alpha.dim() != 2 or int(alpha.size(1)) != int(self.num_classes):
            print(
                "[WARN] Invalid Dirichlet alpha shape {}; expected [N, {}]. Use uniform alpha.".format(
                    tuple(alpha.shape), self.num_classes
                )
            )
            alpha = torch.ones(1, self.num_classes, device=self.device)

        alpha = torch.clamp(alpha, min=1e-6)
        n_alpha = int(alpha.size(0))
        base = table_size // n_alpha
        remainder = table_size % n_alpha

        for n, a in enumerate(alpha):
            s = base + (1 if n < remainder else 0)
            if s <= 0:
                continue
            alpha_idxs += [n] * s
            distribution = Dirichlet(a)
            group_num = s // self.max_sample_size
            final_group = s % self.max_sample_size
            for _ in range(group_num):
                sample_list.append(distribution.sample((self.max_sample_size,)).cpu())
            if final_group > 0:
                sample_list.append(distribution.sample((final_group,)).cpu())

        if len(sample_list) == 0:
            return torch.empty(0, self.num_classes), []
        return torch.cat(sample_list, dim=0), alpha_idxs

    def get_uniform_samples(self, table_size=1000000):
        raise NotImplementedError("Not implemented uniform sampling")

    @staticmethod
    def _normalize_yprime_output(out, ref_tensor):
        if not torch.is_tensor(out):
            out = torch.tensor(out)
        out = out.detach()
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return out.to(ref_tensor)

    @staticmethod
    def _safe_get_yprime(blackbox, y_batch, x_info=None):
        """Call blackbox.get_yprime robustly.

        Some defenses, especially original MAD code paths, do not support all batched
        forms during recover-table generation.  First try a normal batched call.  If
        that fails, fall back to per-sample calls while preserving a 2D [N,K] result.
        """
        try:
            if x_info is None:
                out = blackbox.get_yprime(y_batch)
            else:
                out = blackbox.get_yprime(y_batch, x_info=x_info)
            return Table_Recover._normalize_yprime_output(out, y_batch)
        except Exception as batch_exc:
            outs = []
            for j in range(len(y_batch)):
                y_one = y_batch[j : j + 1]
                try:
                    if x_info is None:
                        out = blackbox.get_yprime(y_one)
                    else:
                        out = blackbox.get_yprime(y_one, x_info=x_info)
                except Exception:
                    # Some legacy wrappers expect a single vector [K] rather than [1,K].
                    y_vec = y_batch[j]
                    if x_info is None:
                        out = blackbox.get_yprime(y_vec)
                    else:
                        out = blackbox.get_yprime(y_vec, x_info=x_info)
                outs.append(Table_Recover._normalize_yprime_output(out, y_batch))
            if len(outs) == 0:
                raise batch_exc
            return torch.cat(outs, dim=0)

    @staticmethod
    def get_perturbed_label_sample(
        blackbox,
        true_label_sample,
        xs=None,
        x_info_idxs=None,
        batch_size=32,
        output=None,
        count=None,
        proc_idx=None,
    ):
        if count is None:
            pbar = tqdm(total=len(true_label_sample))
        true_label_sample = true_label_sample.float()
        if true_label_sample.dim() != 2:
            raise RuntimeError(
                "get_perturbed_label_sample expects 2D labels, got shape={}".format(
                    tuple(true_label_sample.shape)
                )
            )
        # Use the generated table width as the source of truth.  Do not trust
        # blackbox.num_classes here, because some legacy wrappers keep a CIFAR10
        # default even in CIFAR100 jobs.
        num_classes = int(true_label_sample.size(1))
        blackbox_num_classes = getattr(blackbox, "num_classes", None)
        if blackbox_num_classes is not None and int(blackbox_num_classes) != num_classes:
            print(
                "[WARN] blackbox.num_classes={} but recover-table width={}; "
                "using table width as expected class count.".format(
                    int(blackbox_num_classes), num_classes
                )
            )

        if xs is None or x_info_idxs is None:
            perturbed_label_sample = []
            for start_idx in range(0, len(true_label_sample), batch_size):
                end_idx = min(start_idx + batch_size, len(true_label_sample))
                y_batch = true_label_sample[start_idx:end_idx, :].to(blackbox.device)
                perturbed_label = Table_Recover._safe_get_yprime(blackbox, y_batch)
                perturbed_label_sample.append(perturbed_label.detach().to(true_label_sample))
                if int(perturbed_label.size(1)) != num_classes:
                    raise RuntimeError(
                        "blackbox.get_yprime returned width {}, expected {}".format(
                            int(perturbed_label.size(1)), num_classes
                        )
                    )
                if count is not None:
                    count.value += len(perturbed_label)
                else:
                    pbar.update(len(perturbed_label))
            result = torch.cat(perturbed_label_sample, dim=0)
        else:
            assert len(x_info_idxs) == len(true_label_sample), (
                "The length of x_info_idxs must be equal to the length of true_label_sample!"
            )
            x_info_idxs_set = set(x_info_idxs)
            x_info_idxs = np.array(x_info_idxs)
            perturbed_label_sample = torch.zeros_like(true_label_sample)
            for i in x_info_idxs_set:
                x_i = xs[int(i)].unsqueeze(0)
                index_i = np.arange(len(true_label_sample))[x_info_idxs == i]
                true_label_i = true_label_sample[index_i].to(blackbox.device)
                x_info = blackbox.get_xinfo(x_i)
                for start_idx in range(0, len(true_label_i), batch_size):
                    end_idx = min(start_idx + batch_size, len(true_label_i))
                    y_batch = true_label_i[start_idx:end_idx, :]
                    perturbed_label = Table_Recover._safe_get_yprime(blackbox, y_batch, x_info=x_info)
                    if int(perturbed_label.size(1)) != num_classes:
                        raise RuntimeError(
                            "blackbox.get_yprime returned width {}, expected {}".format(
                                int(perturbed_label.size(1)), num_classes
                            )
                        )
                    perturbed_label_sample[index_i[start_idx:end_idx], :] = perturbed_label.detach().to(
                        true_label_sample
                    )
                    if count is not None:
                        count.value += len(perturbed_label)
                    else:
                        pbar.update(len(perturbed_label))
            result = perturbed_label_sample

        if output is not None and proc_idx is not None:
            output[proc_idx] = result
            return
        return result

    def get_perturbed_label_sample_parallel(self, blackbox, true_label_sample, xs=None, x_info_idxs=None, num_proc=10):
        print("Generating recover table with %d processes..." % num_proc)
        torch.multiprocessing.set_start_method("spawn", force=True)
        with Manager() as manager:
            proc_data = np.array_split(true_label_sample.cpu(), num_proc)
            if xs is not None and x_info_idxs is not None:
                assert len(x_info_idxs) == len(true_label_sample), "x_info_idxs must have the same length as true_label_sample!"
                x_info_idxs_proc = np.array_split(np.array(x_info_idxs), num_proc)
                shared_xs = manager.list([x.cpu() if torch.is_tensor(x) else x for x in xs])
            else:
                x_info_idxs_proc = [None] * num_proc
                shared_xs = None
            count = manager.Value("i", 0)
            perturbed_label_output = manager.list([None] * num_proc)
            proc = []
            for i in range(num_proc):
                p = Process(
                    target=Table_Recover.get_perturbed_label_sample,
                    args=(
                        blackbox,
                        proc_data[i],
                        shared_xs,
                        x_info_idxs_proc[i],
                        self.batch_size,
                        perturbed_label_output,
                        count,
                        i,
                    ),
                )
                proc.append(p)
            for p in proc:
                p.start()
            with tqdm(total=len(true_label_sample)) as pbar:
                prev_count = 0
                while None in perturbed_label_output:
                    current_count = count.value
                    if current_count > prev_count:
                        pbar.update(current_count - prev_count)
                        prev_count = current_count
                    dead_failed = [idx for idx, p in enumerate(proc) if (not p.is_alive()) and p.exitcode not in (None, 0)]
                    if dead_failed:
                        raise RuntimeError(
                            "Recover-table subprocess failed: {}. See traceback above.".format(dead_failed)
                        )
                    time.sleep(0.2)
                current_count = sum(len(perturbed_label_output[n]) for n in range(num_proc))
                if current_count > prev_count:
                    pbar.update(current_count - prev_count)
            for p in proc:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError("Recover-table subprocess exited with code {}".format(p.exitcode))
            perturbed_label_sample = list(perturbed_label_output)
        res = torch.cat(perturbed_label_sample, dim=0)
        print("Ended multiprocessing with total number of samples = %d" % len(res))
        return res

    def train_recover_nn(self, model, pert_label, true_label, epoch=20, batch_size=128, lr=1e-3):
        """Train the D-DAE recovery network with strict class-dimension checking.

        For CIFAR100, both pert_label and true_label must be [N, 100].
        If a [N, 10] tensor appears here, that means an old CIFAR10 recovery
        table/model is being reused or the table generation code is producing
        labels with the wrong class count. Do not pad/truncate silently, because
        that can make the run finish with invalid results.
        """
        expected_dim = int(self.num_classes)
        pert_label = self._validate_label_matrix("pert_label", pert_label)
        true_label = self._validate_label_matrix("true_label", true_label)

        if int(pert_label.size(1)) != expected_dim:
            raise RuntimeError(
                "pert_label width mismatch: got {}, expected {}".format(
                    int(pert_label.size(1)), expected_dim
                )
            )
        if int(true_label.size(1)) != expected_dim:
            raise RuntimeError(
                "true_label width mismatch: got {}, expected {}".format(
                    int(true_label.size(1)), expected_dim
                )
            )
        if hasattr(model, "fc1") and int(model.fc1.in_features) != expected_dim:
            raise RuntimeError(
                "Recover_NN input dim mismatch: model.fc1.in_features={} expected={}".format(
                    int(model.fc1.in_features), expected_dim
                )
            )
        if hasattr(model, "fc3") and int(model.fc3.out_features) != expected_dim:
            raise RuntimeError(
                "Recover_NN output dim mismatch: model.fc3.out_features={} expected={}".format(
                    int(model.fc3.out_features), expected_dim
                )
            )

        dataset = TensorDataset(pert_label.cpu(), true_label.cpu())
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        model.train()

        for e in tqdm(range(epoch)):
            total_loss = 0.0
            total_dist = 0.0
            total_iter = 0

            for pl, tl in trainloader:
                total_iter += 1
                pl = pl.to(self.device)
                tl = tl.to(self.device)

                if pl.dim() != 2 or int(pl.size(1)) != expected_dim:
                    raise RuntimeError(
                        "Batch pert_label width mismatch: got shape={}, expected width={}".format(
                            tuple(pl.shape), expected_dim
                        )
                    )
                if tl.dim() != 2 or int(tl.size(1)) != expected_dim:
                    raise RuntimeError(
                        "Batch true_label width mismatch: got shape={}, expected width={}".format(
                            tuple(tl.shape), expected_dim
                        )
                    )

                output = model(pl)
                loss = F.l1_loss(output, tl) * self.num_classes
                dist = torch.mean(torch.norm(output.detach() - tl, p=2, dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_dist += dist.item()

            scheduler.step()
            avg_loss = total_loss / max(total_iter, 1)
            avg_dist = total_dist / max(total_iter, 1)
            print("Epoch: {}\tLoss: {:.4f}\tL2 Distance: {:.4f}".format(e + 1, avg_loss, avg_dist))
            self.logger.writerow([e + 1, avg_loss, avg_dist])

        return model

    def _normalize_probability_rows(self, value, fallback, context=""):
        """Return a finite row-stochastic tensor, falling back row-wise when needed."""
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=self.device, dtype=fallback.dtype)
        value = value.to(device=fallback.device, dtype=fallback.dtype)
        fallback = fallback.to(device=value.device, dtype=value.dtype)

        if value.dim() != 2:
            raise RuntimeError(
                "{} must be 2D, got shape={}".format(context or "probability tensor", tuple(value.shape))
            )
        if int(value.size(1)) != int(self.num_classes):
            raise RuntimeError(
                "{} width mismatch: got {}, expected {}".format(
                    context or "probability tensor", int(value.size(1)), int(self.num_classes)
                )
            )

        value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
        value = torch.clamp(value, min=0.0)

        row_sum = value.sum(dim=1, keepdim=True)
        valid = torch.isfinite(row_sum) & (row_sum > 1e-12)
        normalized = value / row_sum.clamp_min(1e-12)

        if fallback.dim() == 2 and int(fallback.size(0)) == int(value.size(0)):
            fallback_norm = torch.clamp(fallback, min=0.0)
            fallback_sum = fallback_norm.sum(dim=1, keepdim=True)
            fallback_norm = fallback_norm / fallback_sum.clamp_min(1e-12)
            normalized = torch.where(valid, normalized, fallback_norm)
        return normalized

    def _apply_recovery_quality_guard(self, y_before, y_after, context=""):
        """Avoid using collapsed D-DAE recovery outputs.

        In several CIFAR10/CIFAR100 D-DAE runs, both recover_nn=1 and recover_nn=0
        can produce near-uniform labels.  Training on these labels makes the
        surrogate loss stay close to log(K).  This guard keeps the recovered label
        only when it is meaningfully confident; otherwise it falls back to the
        original queried label for that row.
        """
        y_before = self._normalize_probability_rows(y_before, y_before, context="{}_before".format(context))
        y_after = self._normalize_probability_rows(y_after, y_before, context="{}_after".format(context))

        num_classes = float(self.num_classes)
        uniform_conf = 1.0 / num_classes

        # Defaults:
        #   CIFAR10: uniform+0.05 = 0.15 catches the observed 0.085~0.125 collapse.
        #   CIFAR100: uniform+0.02 = 0.03 catches near-uniform 100-way collapse
        #             without rejecting normal non-collapsed outputs too aggressively.
        default_margin = max(0.02, 0.5 / num_classes)
        try:
            uniform_margin = float(os.environ.get("DDAE_UNIFORM_MARGIN", default_margin))
        except Exception:
            uniform_margin = default_margin
        try:
            conf_ratio = float(os.environ.get("DDAE_CONF_RATIO", 0.5))
        except Exception:
            conf_ratio = 0.5

        before_conf = torch.max(y_before, dim=1).values
        after_conf = torch.max(y_after, dim=1).values

        before_useful = before_conf > (uniform_conf + uniform_margin)
        after_near_uniform = after_conf <= (uniform_conf + uniform_margin)
        after_too_weak = after_conf < (before_conf * conf_ratio)

        bad = before_useful & (after_near_uniform | after_too_weak)
        if torch.any(bad):
            num_bad = int(torch.sum(bad).item())
            print(
                "[WARN] D-DAE recovery quality guard fallback in {}: "
                "{}/{} rows. before_conf_mean={:.6f}, after_conf_mean={:.6f}, "
                "uniform_conf={:.6f}, margin={:.6f}, ratio={:.3f}".format(
                    context or "Table_Recover",
                    num_bad,
                    int(y_before.size(0)),
                    float(before_conf.detach().mean().cpu().item()),
                    float(after_conf.detach().mean().cpu().item()),
                    float(uniform_conf),
                    float(uniform_margin),
                    float(conf_ratio),
                )
            )
            y_after = y_after.clone()
            y_after[bad] = y_before[bad]

        return self._normalize_probability_rows(y_after, y_before, context="{}_guarded".format(context))

    def __call__(self, yprime, pbar=None):
        assert yprime.dim() == 2, "yprime must be a batch with dim=2"
        if int(yprime.size(1)) != int(self.num_classes):
            raise RuntimeError(
                "Table_Recover input width mismatch: got {}, expected {}".format(
                    int(yprime.size(1)), int(self.num_classes)
                )
            )

        yprime = yprime.to(self.device).float()
        yprime = self._normalize_probability_rows(yprime, yprime, context="input_yprime")

        if self.recover_nn:
            with torch.no_grad():
                raw_res = self.nn(yprime).detach()
            res = self._apply_recovery_quality_guard(yprime, raw_res, context="recover_nn")
            return res

        res_rows = []
        rec_dis = []

        perturbed_all = self._validate_label_matrix("perturbed_label_sample", self.perturbed_label_sample).to(self.device)
        true_all = self._validate_label_matrix("true_label_sample", self.true_label_sample).to(self.device)

        if int(len(perturbed_all)) == 0 or int(len(true_all)) == 0:
            print("[WARN] Empty recovery table; fallback to original labels.")
            return yprime

        top1_label = torch.argmax(yprime, dim=1)

        for row_idx in range(len(yprime)):
            c = int(top1_label[row_idx].item())
            y_one = yprime[row_idx : row_idx + 1]

            if self.top1_recover:
                if not hasattr(self, "true_top1"):
                    self.true_top1 = torch.argmax(true_all, dim=1).to(self.device)
                mask_c = self.true_top1.to(self.device) == c
                perturbed_label_filtered = perturbed_all[mask_c, :]
                true_label_filtered = true_all[mask_c, :]
            else:
                perturbed_label_filtered = perturbed_all
                true_label_filtered = true_all

            # This is the direct fix for:
            # RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0
            if int(perturbed_label_filtered.size(0)) == 0 or int(true_label_filtered.size(0)) == 0:
                print(
                    "[WARN] Empty D-DAE candidate set for class {}; fallback one row to original label.".format(c)
                )
                res_rows.append(y_one)
                if pbar is not None:
                    pbar.update(1)
                continue

            distances = torch.norm(y_one - perturbed_label_filtered, p=self.recover_norm, dim=1)

            if distances.numel() == 0:
                print(
                    "[WARN] Empty D-DAE distance tensor for class {}; fallback one row to original label.".format(c)
                )
                res_rows.append(y_one)
                if pbar is not None:
                    pbar.update(1)
                continue

            if not torch.isfinite(distances).all():
                finite_mask = torch.isfinite(distances)
                if torch.sum(finite_mask).item() == 0:
                    print(
                        "[WARN] All D-DAE distances are non-finite for class {}; fallback one row.".format(c)
                    )
                    res_rows.append(y_one)
                    if pbar is not None:
                        pbar.update(1)
                    continue
                distances = distances[finite_mask]
                true_label_filtered = true_label_filtered[finite_mask, :]

            if not self.recover_mean:
                min_idx = torch.argmin(distances)
                recovered_one = true_label_filtered[min_idx : min_idx + 1, :]
                rec_dis.append(distances[min_idx].detach())
            else:
                min_distance = torch.min(distances)
                tolerance = max(float(self.tolerance), float(min_distance.detach().cpu().item()))
                mask = distances <= tolerance

                if torch.sum(mask).item() == 0:
                    min_idx = torch.argmin(distances)
                    recovered_one = true_label_filtered[min_idx : min_idx + 1, :]
                    rec_dis.append(distances[min_idx].detach())
                else:
                    recovered_one = torch.mean(true_label_filtered[mask, :], dim=0, keepdim=True)
                    rec_dis.append(torch.mean(distances[mask]).detach())

            recovered_one = self._normalize_probability_rows(
                recovered_one, y_one, context="table_recover_row"
            )
            res_rows.append(recovered_one)

            if pbar is not None:
                pbar.update(1)

        if len(res_rows) == 0:
            print("[WARN] No D-DAE rows recovered; fallback to original labels.")
            res = yprime
        else:
            res = torch.cat(res_rows, dim=0)

        if int(res.size(0)) != int(yprime.size(0)):
            raise RuntimeError(
                "Recovered batch length mismatch: got {}, expected {}".format(
                    int(res.size(0)), int(yprime.size(0))
                )
            )

        res = self._apply_recovery_quality_guard(yprime, res, context="table_recover")

        self.call_count += len(yprime)
        if len(rec_dis) > 0:
            rec_dis_t = torch.stack([d.detach().cpu() for d in rec_dis])
            mean_rec_dis = torch.mean(rec_dis_t).item()
            std_rec_dis = torch.std(rec_dis_t).item() if len(rec_dis_t) > 1 else 0.0
        else:
            mean_rec_dis, std_rec_dis = 0.0, 0.0
        self.logger.writerow([self.call_count, mean_rec_dis, std_rec_dis])

        return res
