from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


def _resolve_input_size(metadata: Dict[str, Any], default: int) -> int:
    """Resolve an optional square input size from metadata."""
    raw = metadata.get("input_size", default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _resolve_dataset_domain(metadata: Dict[str, Any]) -> str:
    raw = str(metadata.get("dataset_domain", "cifar10")).strip().lower()
    if raw in {"", "auto"}:
        return "cifar10"
    return raw


def _build_transform(
    metadata: Dict[str, Any],
    input_size: int,
    to_three_channels: bool = False,
) -> transforms.Compose:
    pipeline: list[Any] = [
        transforms.Resize((input_size, input_size)),
    ]
    if to_three_channels:
        pipeline.append(transforms.Grayscale(num_output_channels=3))
    pipeline.append(transforms.ToTensor())
    normalization_cfg = metadata.get("normalization", "auto")
    if isinstance(normalization_cfg, dict):
        mean = normalization_cfg.get("mean")
        std = normalization_cfg.get("std")
        if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)):
            pipeline.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(pipeline)


def get_dataloader(
    dataset_name: str,
    input_size: int,
    batch_size: int = 128,
    num_workers: int = 2,
    data_root: str = "./data",
    train_shuffle: bool = True,
    test_shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """Factory for built-in datasets with dynamic resize."""
    name = str(dataset_name).strip().lower()

    if name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )
        dataset_num_classes = 10
    elif name == "cifar100":
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )
        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )
        dataset_num_classes = 100
    elif name in {"fashion_mnist", "fashion-mnist"}:
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        train_dataset = datasets.FashionMNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.FashionMNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )
        dataset_num_classes = 10
    else:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. "
            "Supported: cifar10, cifar100, fashion_mnist."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle,
        num_workers=num_workers,
    )
    return train_loader, test_loader, dataset_num_classes


def _build_dataset_from_metadata(metadata: Dict[str, Any]) -> Tuple[Dataset[Any], Dataset[Any], int, Tuple[int, ...]]:
    """Create a torchvision dataset instance from high-level metadata.

    This helper encapsulates dataset selection logic so that the rest of
    the pipeline only needs to deal with generic PyTorch datasets.

    Parameters
    ----------
    metadata:
        A dictionary describing the dataset. Expected keys include:

        - ``dataset_domain``: Name of the dataset domain, e.g., ``"cifar10"`` or ``"mnist"``.
        - ``data_root`` (optional): Root directory where data will be
          downloaded/stored. Defaults to ``"./data"``.

    Returns
    -------
    dataset:
        A PyTorch ``Dataset`` object providing ``(input, label)`` pairs
        for training-like data.

    num_classes:
        The number of classes in the dataset.

    input_shape:
        The expected single-sample input shape, e.g. ``(3, 32, 32)``.

    Notes
    -----
    Only a small subset of common vision datasets are implemented for
    this project (e.g., CIFAR-10 and MNIST). The function is written in
    a way that makes it straightforward to extend with additional
    domains later.
    """
    domain = _resolve_dataset_domain(metadata)
    data_root = str(metadata.get("data_root", "./data"))

    if domain in {"cifar10", "cifar100", "fashion_mnist"}:
        default_size = 32 if domain in {"cifar10", "cifar100"} else 28
        input_size = _resolve_input_size(metadata, default=default_size)
        train_loader, test_loader, num_classes = get_dataloader(
            dataset_name=domain,
            input_size=input_size,
            batch_size=1,
            num_workers=0,
            data_root=data_root,
            train_shuffle=False,
            test_shuffle=False,
        )
        input_shape = (3, input_size, input_size)
        return train_loader.dataset, test_loader.dataset, num_classes, input_shape

    if domain == "mnist":
        # Keep backward compatibility for legacy domain naming.
        input_size = _resolve_input_size(metadata, default=28)
        transform = _build_transform(metadata, input_size, to_three_channels=True)
        train_dataset = datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )
        num_classes = len(getattr(train_dataset, "classes", [])) or 10
        input_shape = (3, input_size, input_size)
        return train_dataset, test_dataset, num_classes, input_shape

    if domain == "custom_imagefolder":
        input_size = _resolve_input_size(metadata, default=224)
        dataset_path = str(metadata.get("dataset_path") or data_root)
        root = Path(dataset_path)
        if not root.exists():
            raise ValueError(
                f"custom_imagefolder path does not exist: {dataset_path}"
            )
        transform = _build_transform(metadata, input_size)
        train_dataset = datasets.ImageFolder(root=str(root), transform=transform)
        test_dataset = train_dataset
        num_classes = len(getattr(train_dataset, "classes", []))
        if num_classes <= 0:
            raise ValueError("custom_imagefolder has no class folders.")
        sample_x, _ = train_dataset[0]
        channels = int(sample_x.shape[0]) if hasattr(sample_x, "shape") else 3
        input_shape = (channels, input_size, input_size)
        return train_dataset, test_dataset, num_classes, input_shape

    raise ValueError(
        f"Unsupported dataset_domain '{domain}'. "
        "Currently supported domains include: 'cifar10', 'cifar100', "
        "'fashion_mnist', 'mnist', 'custom_imagefolder'."
    )


def prepare_shadow_data(
    metadata: Dict[str, Any],
    batch_size: int = 128,
    num_workers: int = 2,
    shuffle: bool = True,
    member_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader, int, Tuple[int, ...]]:
    """Prepare data loaders for membership/non-membership candidate sets.

    This function is a crucial bridge between high-level configuration
    (metadata) and concrete PyTorch data structures used by shadow
    models and attack algorithms.

    Algorithm
    ---------
    1. Use ``metadata['dataset_domain']`` to instantiate and download a
       torchvision training dataset (e.g., CIFAR-10 or MNIST).
    2. Randomly split the dataset into:
       - ``Member Candidates``: ``member_ratio`` of the samples.
       - ``Non-Member Candidates``: the remaining samples.
    3. Wrap each split into a ``DataLoader`` so that subsequent modules
       (shadow trainer, MIA/PIA) can iterate over batches efficiently.

    Parameters
    ----------
    metadata:
        Configuration dictionary that must at least contain
        ``dataset_domain``; see ``_build_dataset_from_metadata`` for
        optional keys.

    batch_size:
        Batch size for the resulting data loaders.

    num_workers:
        Number of worker processes used by the data loaders. Set to 0
        if you experience issues on your platform.

    shuffle:
        Whether to shuffle the member and non-member loaders on each
        epoch. For shadow training and MIA/PIA, enabling shuffling is
        typically beneficial.

    member_ratio:
        Fraction of samples assigned to member candidates. Must be in
        (0, 1). For example, 0.8 means 80% member and 20% non-member.

    Returns
    -------
    member_loader:
        Data loader over the ``Member Candidates`` subset.

    non_member_loader:
        Data loader over the ``Non-Member Candidates`` subset.

    num_classes:
        Number of classes in the dataset. Useful for constructing
        shadow models and meta-classifiers.

    input_shape:
        The expected single-sample input shape. This is used by the
        shadow model definition to configure the first convolutional or
        linear layer correctly.
    """
    train_dataset, test_dataset, num_classes, input_shape = _build_dataset_from_metadata(metadata)

    total_size = len(train_dataset)
    if total_size == 0 or len(test_dataset) == 0:
        raise RuntimeError("Dataset is empty; cannot prepare shadow data.")
    if total_size < 2:
        raise RuntimeError("Training dataset too small for member split (need at least 2 samples).")

    if not 0.0 < member_ratio < 1.0:
        raise ValueError(f"member_ratio must be in (0, 1), got {member_ratio}.")

    member_size = int(member_ratio * total_size)
    member_size = max(1, min(member_size, total_size - 1))
    holdout_size = total_size - member_size

    member_subset, _unused_train_holdout = random_split(
        train_dataset,
        lengths=[member_size, holdout_size],
        generator=torch.Generator().manual_seed(int(metadata.get("split_seed", 42))),
    )
    non_member_subset: Dataset[Any]
    max_non_member = len(test_dataset)
    if max_non_member > member_size:
        rng = torch.Generator().manual_seed(int(metadata.get("split_seed", 42)) + 1)
        indices = torch.randperm(max_non_member, generator=rng)[:member_size].tolist()
        non_member_subset = Subset(test_dataset, indices)
    else:
        non_member_subset = test_dataset

    member_loader = DataLoader(
        member_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    non_member_loader = DataLoader(
        non_member_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return member_loader, non_member_loader, num_classes, input_shape

