"""Balanced language-family batch sampler for distributed training."""

import math
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from promptingnemo.tokenizer.config import LANG_TO_FAMILY


class BalancedLanguageBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, temperature=0.2, seed=42, lang_to_family_map: Dict[str, str] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0
        self.lang_to_family_map = {k.upper(): v for k, v in (lang_to_family_map or {}).items()}

        self.num_samples = len(self.dataset)
        self.world_size = 1
        self.rank = 0
        self._refresh_distributed_context()

        family_counts: Dict[str, int] = {}
        self.sample_families: List[str] = []
        for lang_id in self.dataset.language_ids:
            family = self._resolve_family(lang_id)
            self.sample_families.append(family)
            family_counts[family] = family_counts.get(family, 0) + 1

        self.family_counts = family_counts
        self.families = list(self.family_counts.keys())
        print("ALL LANGUAGE FAMILIES")
        print(self.families)
        print("ALL FAMILY COUNTS")
        print(self.family_counts)

        dataset_weights = getattr(self.dataset, 'sample_keyphrase_weights', None)
        if dataset_weights is not None and len(dataset_weights) == len(self.dataset.language_ids):
            self.sample_weights = np.asarray(dataset_weights, dtype=np.float32)
        else:
            self.sample_weights = np.ones(len(self.dataset.language_ids), dtype=np.float32)

        # Calculate sampling probabilities with temperature
        total_samples = len(self.dataset)
        weights = np.array([count / total_samples for count in self.family_counts.values()]) if total_samples > 0 else np.array([])
        if weights.size > 0:
            temp_weights = weights ** (1 / self.temperature)
            self.family_sample_probs = temp_weights / np.sum(temp_weights)
        else:
            self.family_sample_probs = np.array([])

        family_indices_map = {family: [] for family in self.families}
        family_weight_map = {family: [] for family in self.families}
        for idx, family in enumerate(self.sample_families):
            if family in family_indices_map:
                family_indices_map[family].append(idx)
                family_weight_map[family].append(float(self.sample_weights[idx]))
        self.family_indices = {
            family: np.array(indices, dtype=np.uint32) for family, indices in family_indices_map.items()
        }
        self.family_weights = {
            family: np.array(weights if weights else [1.0] * len(family_indices_map[family]), dtype=np.float32)
            for family, weights in family_weight_map.items()
        }

        self.num_batches_per_epoch = self.num_samples // self.batch_size

        # Adjust for distributed training
        self.num_samples_per_rank = self._calculate_samples_per_rank()

    def __iter__(self):
        self._refresh_distributed_context()

        # Seed with epoch to ensure different shuffling each epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)

        # 1. Create the full list of indices for the epoch based on language sampling probabilities
        num_samples_per_family = np.round(self.family_sample_probs * self.num_samples).astype(int)

        # Adjust to ensure the total number of samples is correct
        diff = self.num_samples - np.sum(num_samples_per_family)
        if diff != 0 and len(self.family_sample_probs) > 0:
            num_samples_per_family[np.argmax(self.family_sample_probs)] += diff

        epoch_indices_list = []
        for i, family in enumerate(self.families):
            num_family_samples = num_samples_per_family[i]
            if num_family_samples > 0:
                replace = len(self.family_indices[family]) < num_family_samples
                weights = self.family_weights.get(family)
                if weights is None or len(weights) == 0:
                    probs = None
                else:
                    weights = weights.astype(np.float64)
                    total = np.sum(weights)
                    if total <= 0:
                        probs = None
                    else:
                        probs = weights / total
                indices = np.random.choice(
                    self.family_indices[family],
                    num_family_samples,
                    replace=replace,
                    p=probs,
                )
                epoch_indices_list.append(indices)

        if not epoch_indices_list:
            self.epoch += 1
            return iter([])

        epoch_indices = np.concatenate(epoch_indices_list)

        # 2. Shuffle the indices for the entire epoch
        np.random.shuffle(epoch_indices)

        # 3. Yield batches for the current rank without creating a list of all batches
        num_batches = len(epoch_indices) // self.batch_size

        for batch_idx in range(self.rank, num_batches, self.world_size):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            yield epoch_indices[start_idx:end_idx].tolist()

        self.epoch += 1

    def __len__(self):
        self._refresh_distributed_context()
        num_batches = self.num_samples // self.batch_size
        if num_batches == 0:
            return 0
        return math.ceil(num_batches / max(self.world_size, 1))

    def _calculate_samples_per_rank(self):
        world_size = max(self.world_size, 1)
        if world_size <= 1:
            return self.num_samples
        samples = self.num_samples // world_size
        if self.num_samples % world_size != 0:
            samples += 1
        return samples

    def _refresh_distributed_context(self):
        if not dist.is_available():
            self.world_size = 1
            self.rank = 0
            return

        try:
            if dist.is_initialized():
                self.world_size = max(int(dist.get_world_size()), 1)
                self.rank = int(dist.get_rank())
            else:
                self.world_size = 1
                self.rank = 0
        except (RuntimeError, ValueError):
            # During dataloader workers spawn, the default process group might not be ready yet.
            self.world_size = 1
            self.rank = 0
        self.num_samples_per_rank = self._calculate_samples_per_rank()

    def _resolve_family(self, lang_id: str) -> str:
        if not lang_id:
            return "UNKNOWN"
        lang_upper = str(lang_id).upper()
        if lang_upper in self.lang_to_family_map:
            return self.lang_to_family_map[lang_upper]
        if lang_upper in LANG_TO_FAMILY:
            return LANG_TO_FAMILY[lang_upper]
        return f"Singleton_{lang_upper}"
