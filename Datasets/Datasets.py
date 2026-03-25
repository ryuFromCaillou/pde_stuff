import torch
from torch.utils.data import Dataset, Subset


class FullPDEDataset(Dataset):
    def __init__(self, cfg, build_fn):
        """
        cfg: config object (attribute access) passed into build_fn.
        build_fn: function like build_dataset_from_burgers(cfg) returning (t, x, y, y_noisy, N).
        """
        t_np, x_np, y_np, y_noisy_np, _ = build_fn(cfg)

        # keep on CPU; move to device in training loop
        self.t = torch.from_numpy(t_np).float()
        self.x = torch.from_numpy(x_np).float()
        self.y = torch.from_numpy(y_np).float()
        self.y_noisy = torch.from_numpy(y_noisy_np).float()

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, idx):
        # return shape (1,) so DataLoader stacks to (B,1)
        return (
            self.t[idx : idx + 1],
            self.x[idx : idx + 1],
            self.y[idx : idx + 1],
            self.y_noisy[idx : idx + 1],
        )

    def full(self):
        # for full-domain operations (plots, diagnostics)
        return self.t, self.x, self.y, self.y_noisy


class PDEDataset(Dataset):
    """
    PDE dataset that can optionally expose a fixed-size subset view (for faster experiments)
    while still retaining the full dataset for plotting/diagnostics.

    If `cfg.subset_size` is set (int), this dataset's `__len__`/`__getitem__` will operate on
    `Subset(full_dataset, idx)` where `idx` is sampled once via a seeded `torch.Generator`.

    - Full tensors are accessible via `.full()`.
    - Subset indices are accessible via `.subset_indices`.
    """

    def __init__(self, cfg, build_fn):
        self.full_dataset = FullPDEDataset(cfg, build_fn)

        subset_size = getattr(cfg, "subset_size", None)
        subset_seed = int(getattr(cfg, "subset_seed", 0))

        self.subset_indices = None
        self.subset = None

        if subset_size is not None:
            subset_size = int(subset_size)
            if subset_size <= 0:
                raise ValueError("cfg.subset_size must be a positive int when provided.")

            m = min(subset_size, len(self.full_dataset))
            g = torch.Generator().manual_seed(subset_seed)
            idx = torch.randperm(len(self.full_dataset), generator=g)[:m]
            self.subset_indices = idx
            self.subset = Subset(self.full_dataset, idx.tolist())

    def __len__(self):
        if self.subset is None:
            return len(self.full_dataset)
        return len(self.subset)

    def __getitem__(self, idx):
        if self.subset is None:
            return self.full_dataset[idx]
        return self.subset[idx]

    def full(self):
        return self.full_dataset.full()
