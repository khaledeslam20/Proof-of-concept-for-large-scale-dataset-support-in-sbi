"""
Problem:

Currently, sbi requires all simulation data to be loaded into memory:
    trainer.append_simulations(theta, x)

This does not scale. For example:
10M samples × 7 floats × 4 bytes ≈ 280 MB (minimum),
and additional copies are created internally.

This makes large-scale inference impractical on typical hardware.

Goal of this Proof of Concept
----------------------------
Demonstrate a minimal working approach for training directly from disk:

    dataset = HDF5Dataset("large_simulations.h5")
    trainer.append_dataset(dataset)
    trainer.train()

This POC focuses on three core components:
1. HDF5Dataset — reads data from disk safely with multiprocessing
2. RunningStats — computes normalization statistics without loading all data
3. append_dataset — connects datasets to the training pipeline

Not included yet (planned for full implementation):
- Chunk-based sampling optimizations
- Internal refactoring (SimulationStore)
- Distributed training support
- Full integration into all sbi trainers
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple




class HDF5Dataset(Dataset):

  # Dataset that reads (theta, x) from an HDF5 file without loading everything into RAM.
    def __init__(self, path: str) -> None:
        self.path = path
        self._file = None

        import h5py
        with h5py.File(path, "r") as f:
            self._len = int(f["theta"].shape[0])

            if f["theta"].shape[0] != f["x"].shape[0]:
                raise ValueError("theta and x must have the same number of samples.")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        import h5py

        if self._file is None:
            self._file = h5py.File(self.path, "r")

        theta = self._file["theta"][idx]
        x = self._file["x"][idx]

        return (
            torch.from_numpy(theta.copy()).float(),
            torch.from_numpy(x.copy()).float(),
        )

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass




class RunningStats:
    """
    1- Computes mean and standard deviation in a single pass.

    2- Uses a numerically stable streaming algorithm (Welford / Chan).

    3- Memory usage is O(D), not O(N × D).

    4- Stats are computed in float64 for stability, then converted to float32.

    Note:
    - Computation is done on CPU
    - Final results can be moved to GPU for training
    """

    def __init__(self) -> None:
        self.n = 0
        self.mean: Optional[Tensor] = None
        self.M2: Optional[Tensor] = None

    def update(self, batch: Tensor) -> None:
        batch = batch.detach().cpu().to(torch.float64)

        if batch.dim() == 1:
            batch = batch.unsqueeze(-1)

        batch = batch.view(-1, batch.shape[-1])

        batch_n = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)

        if self.mean is None:
            self.mean = torch.zeros_like(batch_mean)
            self.M2 = torch.zeros_like(batch_mean)

        delta = batch_mean - self.mean
        total_n = self.n + batch_n

        self.mean += delta * batch_n / total_n
        self.M2 += batch_var * batch_n + delta**2 * self.n * batch_n / total_n
        self.n = total_n

    def finalize(self) -> Tuple[Tensor, Tensor]:
        if self.n < 2:
            raise RuntimeError("Not enough samples to compute statistics.")

        variance = self.M2 / (self.n - 1)
        std = torch.sqrt(variance).clamp(min=1e-5)

        return self.mean.float(), std.float()


def append_dataset(
    dataset: Dataset,
) -> Tuple[Dataset, Dict[str, Tuple[Tensor, Tensor]]]:
    """
    Register a dataset without loading it into memory.

    Steps:
    1. Validate dataset output format
    2. Compute normalization statistics using a streaming pass

    Important:
    - This requires one full pass over the dataset
    - For very large datasets, this adds startup time
    """

    sample = dataset[0]
    if not isinstance(sample, (tuple, list)) or len(sample) not in (2, 3):
        raise ValueError("Dataset must return (theta, x) or (theta, x, prior_mask).")

    stats_loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    theta_stats = RunningStats()
    x_stats = RunningStats()

    for theta_batch, x_batch in stats_loader:
        theta_stats.update(theta_batch)
        x_stats.update(x_batch)

    return dataset, {
        "theta": theta_stats.finalize(),
        "x": x_stats.finalize(),
    }



class SyntheticDataset(Dataset):
    """
    Simple dataset used for testing.

    In practice, this would be replaced by HDF5Dataset.
    """

    def __init__(self, n=50000, d_theta=2, d_x=5):
        torch.manual_seed(42)
        self._theta = torch.randn(n, d_theta) * 2 + 1
        self._x = torch.randn(n, d_x) * 0.5

    def __len__(self):
        return len(self._theta)

    def __getitem__(self, idx):
        return self._theta[idx].clone(), self._x[idx].clone()

    def exact_stats(self):
        return {
            "theta": (self._theta.mean(0), self._theta.std(0)),
            "x": (self._x.mean(0), self._x.std(0)),
        }


def demo():
    print("=" * 50)
    print("Running Proof of Concept Demo")
    print("=" * 50)

    dataset = SyntheticDataset()

    print("\n[1] Computing streaming statistics: ")
    dataset, stats = append_dataset(dataset)

    print("\n[2] Verifying correctness against exact statistics...")
    exact = dataset.exact_stats()

    for key in ["theta", "x"]:
        est_mean, est_std = stats[key]
        true_mean, true_std = exact[key]

        mean_error = (est_mean - true_mean).abs().max().item()
        std_error = (est_std - true_std).abs().max().item()

        print(f"\n  [{key}]")
        print(f"  mean error: {mean_error:.3e}")
        print(f"  std  error: {std_error:.3e}")

        assert mean_error < 1e-3, "Mean error too large"
        assert std_error < 1e-3, "Std error too large"

    print("\n All checks passed")

    print("\n[3] Testing DataLoader:")
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    theta, x = next(iter(loader))

    print(f"theta shape: {theta.shape}")
    print(f"x shape:     {x.shape}")

    


if __name__ == "__main__":
    demo()