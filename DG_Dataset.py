# dataset.py
import os
import glob
from typing import List, Optional, Tuple, Dict, Any, Literal
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def _list_images(root: str) -> List[str]:
    # Faster and robust listing across extensions
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(root, f"*{ext}")))
    return paths


@dataclass
class DGItem:
    path: str
    class_idx: int
    domain_idx: int


class DG_Dataset(Dataset):
    """
    Domain-Generalization style dataset loader.

    Expected folder layout (example: PACS):
        <root_dir>/<dataset_name>/
            Art/
                dog/
                    xxx.jpg
                cat/
                    yyy.jpg
            Photo/
                dog/
                cat/
            Cartoon/
            Sketch/

    Args:
        root_dir: path to datasets root
        dataset_name: folder name under root_dir (e.g., "PACS")
        domains: list of domain names to include (None → all)
        transform: torchvision-like transform applied to PIL image
        return_domain: if True, __getitem__ returns (img, class_idx, domain_idx)
        cache_index: if True, build and keep an in-memory index once
    """
    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        domains: Optional[List[str]] = None,
        transform=None,
        return_domain: bool = True,
        cache_index: bool = True,
    ):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.ds_root = os.path.join(root_dir, dataset_name)
        assert os.path.isdir(self.ds_root), f"Not found: {self.ds_root}"

        # Discover domains
        all_domains = sorted(
            [d for d in os.listdir(self.ds_root) if os.path.isdir(os.path.join(self.ds_root, d))]
        )
        self.domains = sorted(domains) if domains is not None else all_domains
        for d in self.domains:
            assert d in all_domains, f"Domain '{d}' not found in {all_domains}"

        # Collect all class names across selected domains (global label space)
        class_names = set()
        for d in self.domains:
            dpath = os.path.join(self.ds_root, d)
            if not os.path.isdir(dpath):
                continue
            for c in os.listdir(dpath):
                cpath = os.path.join(dpath, c)
                if os.path.isdir(cpath):
                    class_names.add(c)
        self.classes = sorted(class_names)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.domain_to_idx: Dict[str, int] = {d: i for i, d in enumerate(self.domains)}

        # Build index of samples
        items: List[DGItem] = []
        for d in self.domains:
            d_idx = self.domain_to_idx[d]
            dpath = os.path.join(self.ds_root, d)
            for c in os.listdir(dpath):
                cpath = os.path.join(dpath, c)
                if not os.path.isdir(cpath):
                    continue
                if c not in self.class_to_idx:
                    # Skip classes not present in our global set (shouldn't happen)
                    continue
                c_idx = self.class_to_idx[c]
                for img_path in _list_images(cpath):
                    items.append(DGItem(img_path, c_idx, d_idx))

        if len(items) == 0:
            raise RuntimeError(f"No images found under {self.ds_root} for domains {self.domains}")

        self.items: List[DGItem] = items if cache_index else None
        self._items_source = items  # keep a reference even if cache_index=False
        self.transform = transform
        self.return_domain = return_domain
        self.cache_index = cache_index

    def __len__(self) -> int:
        return len(self.items if self.items is not None else self._items_source)

    def __getitem__(self, idx: int):
        it = (self.items if self.items is not None else self._items_source)[idx]
        img = Image.open(it.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.return_domain:
            return img, it.class_idx, it.domain_idx
        else:
            return img, it.class_idx

    # ---------- Weighted sampling helpers ----------
    def compute_sample_weights(
        self,
        mode: Literal["class", "domain", "class_in_domain"] = "class",
        smoothing: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.

        mode:
            "class"           -> inverse class frequency
            "domain"          -> inverse domain frequency
            "class_in_domain" -> inverse frequency of (class within domain)
        smoothing:
            small value added to counts to avoid extreme weights (e.g., 0.1)
        """
        items = self.items if self.items is not None else self._items_source

        if mode == "class":
            # count per class
            from collections import Counter
            counts = Counter([it.class_idx for it in items])
            weights = [1.0 / (counts[it.class_idx] + smoothing) for it in items]

        elif mode == "domain":
            from collections import Counter
            counts = Counter([it.domain_idx for it in items])
            weights = [1.0 / (counts[it.domain_idx] + smoothing) for it in items]

        elif mode == "class_in_domain":
            from collections import Counter
            keys = [(it.domain_idx, it.class_idx) for it in items]
            counts = Counter(keys)
            weights = [1.0 / (counts[(it.domain_idx, it.class_idx)] + smoothing) for it in items]

        else:
            raise ValueError(f"Unknown weighting mode: {mode}")

        t = torch.tensor(weights, dtype=torch.float)
        # Normalize for stability (optional)
        t = t / t.mean()
        return t

    def make_dataloader(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        weighted_sampling: bool = False,
        weight_mode: Literal["class", "domain", "class_in_domain"] = "class",
        replacement: bool = True,
        smoothing: float = 0.0,
    ) -> DataLoader:
        """
        Create a DataLoader. If weighted_sampling=True, uses WeightedRandomSampler.

        Note: when using WeightedRandomSampler, set shuffle=False.
        """
        sampler = None
        if weighted_sampling:
            sample_weights = self.compute_sample_weights(mode=weight_mode, smoothing=smoothing)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),  # epoch size ≈ dataset size
                replacement=replacement,
            )
            shuffle = False  # sampler + shuffle are mutually exclusive

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
