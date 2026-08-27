"""Topology-focused patch sampling from generic raw-volume manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from densetopo_unet.config import ExperimentConfig
from densetopo_unet.io import open_raw_volume
from densetopo_unet.manifest import (
    DataManifest,
    SampleRecord,
    Split,
    load_critical_points,
    load_false_cases,
)


NormalizationMode = Literal["max_abs", "positive_max"]


def normalization_scale(
    volume: np.ndarray,
    mode: NormalizationMode,
    epsilon: float,
) -> float:
    """Calculate a zero-anchored per-volume normalization divisor."""

    if epsilon <= 0:
        raise ValueError("normalization epsilon must be positive")
    if mode == "max_abs":
        scale = float(np.max(np.abs(volume)))
    elif mode == "positive_max":
        scale = float(np.max(volume))
    else:
        raise ValueError(f"unsupported normalization mode: {mode}")
    if not np.isfinite(scale) or scale <= epsilon:
        raise ValueError(
            f"normalization scale must be finite and greater than {epsilon}; observed {scale}"
        )
    return scale


@dataclass(frozen=True)
class _TopologyIndex:
    fn: np.ndarray
    fp: np.ndarray
    ft: np.ndarray
    original_extrema: np.ndarray
    coordinates: np.ndarray
    weights: np.ndarray


def _topology_index(record: SampleRecord, shape: tuple[int, int, int]) -> _TopologyIndex:
    if record.false_cases is None or record.critical_points is None:
        raise ValueError(f"sample {record.sample_id} does not provide training topology labels")
    cases = load_false_cases(record.false_cases, shape)
    extrema = load_critical_points(record.critical_points, shape)
    coordinate_weights: dict[tuple[int, int, int], float] = {}
    for case, weight in (("fn", 5.0), ("fp", 3.0), ("ft", 4.0)):
        for coordinate in cases[case]:
            key = tuple(int(value) for value in coordinate)
            coordinate_weights[key] = max(coordinate_weights.get(key, 0.0), weight)
    for coordinate in extrema:
        key = tuple(int(value) for value in coordinate)
        coordinate_weights[key] = max(coordinate_weights.get(key, 0.0), 1.0)
    if coordinate_weights:
        coordinates = np.asarray(list(coordinate_weights), dtype=np.int32)
        weights = np.asarray(list(coordinate_weights.values()), dtype=np.float32)
    else:
        coordinates = np.empty((0, 3), dtype=np.int32)
        weights = np.empty((0,), dtype=np.float32)
    return _TopologyIndex(
        fn=cases["fn"],
        fp=cases["fp"],
        ft=cases["ft"],
        original_extrema=extrema,
        coordinates=coordinates,
        weights=weights,
    )


class TopologyPatchDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Deterministic topology-focused patches with one decompressed input channel."""

    def __init__(
        self,
        manifest: DataManifest,
        config: ExperimentConfig,
        split: Split,
        samples_per_epoch: int,
        seed: int,
        augment: bool,
    ) -> None:
        super().__init__()
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.records = manifest.by_split(split)
        if not self.records:
            raise ValueError(f"manifest contains no {split} samples")
        if any(record.reference is None for record in self.records):
            raise ValueError(f"all {split} samples require a reference volume")
        self.config = config
        self.patch_size = config.model.patch_size
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = 0
        self._indices = [
            _topology_index(record, config.volume.shape) for record in self.records
        ]
        self._memmaps: dict[Path, np.memmap] = {}
        self._scales: dict[Path, float] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def _volume(self, path: Path) -> np.memmap:
        if path not in self._memmaps:
            self._memmaps[path] = open_raw_volume(path, self.config.volume)
        return self._memmaps[path]

    def _scale(self, path: Path) -> float:
        if path not in self._scales:
            self._scales[path] = normalization_scale(
                self._volume(path),
                self.config.normalization.mode,
                self.config.normalization.epsilon,
            )
        return self._scales[path]

    def _choose_start(
        self,
        record_index: int,
        rng: np.random.Generator,
    ) -> tuple[int, int, int]:
        index = self._indices[record_index]
        draw = float(rng.random())
        if draw < 0.35:
            candidates = index.fn
        elif draw < 0.60:
            candidates = index.fp
        elif draw < 0.80:
            candidates = index.original_extrema
        else:
            candidates = np.empty((0, 3), dtype=np.int32)

        if len(candidates):
            center = candidates[int(rng.integers(len(candidates)))].copy()
            jitter = np.asarray(
                [
                    rng.integers(-max(1, patch // 8), max(2, patch // 8 + 1))
                    for patch in self.patch_size
                ],
                dtype=np.int32,
            )
            center += jitter
        else:
            center = np.asarray(
                [rng.integers(0, dimension) for dimension in self.config.volume.shape],
                dtype=np.int32,
            )
        return tuple(
            int(np.clip(value - patch // 2, 0, full - patch))
            for value, patch, full in zip(
                center,
                self.patch_size,
                self.config.volume.shape,
            )
        )

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(
            self.seed + self.epoch * self.samples_per_epoch + int(item)
        )
        record_index = int(rng.integers(len(self.records)))
        record = self.records[record_index]
        start = self._choose_start(record_index, rng)
        slices = tuple(
            slice(origin, origin + size) for origin, size in zip(start, self.patch_size)
        )
        decompressed = np.asarray(self._volume(record.decompressed)[slices], dtype=np.float32).copy()
        assert record.reference is not None
        target = np.asarray(self._volume(record.reference)[slices], dtype=np.float32).copy()

        topo_weight = np.zeros(self.patch_size, dtype=np.float32)
        index = self._indices[record_index]
        lower = np.asarray(start, dtype=np.int32)
        upper = lower + np.asarray(self.patch_size, dtype=np.int32)
        inside = np.all((index.coordinates >= lower) & (index.coordinates < upper), axis=1)
        local = index.coordinates[inside] - lower
        if len(local):
            topo_weight[tuple(local.T)] = index.weights[inside]

        if self.augment:
            for axis in range(3):
                if rng.random() < 0.5:
                    decompressed = np.flip(decompressed, axis=axis)
                    target = np.flip(target, axis=axis)
                    topo_weight = np.flip(topo_weight, axis=axis)
        decompressed = np.ascontiguousarray(decompressed)
        target = np.ascontiguousarray(target)
        topo_weight = np.ascontiguousarray(topo_weight)
        scale = self._scale(record.decompressed)
        return {
            "input": torch.from_numpy(decompressed / scale).unsqueeze(0),
            "decompressed": torch.from_numpy(decompressed).unsqueeze(0),
            "target": torch.from_numpy(target).unsqueeze(0),
            "topo_weight": torch.from_numpy(topo_weight).unsqueeze(0),
            "input_scale": torch.tensor(scale, dtype=torch.float32),
            "sample_id": record.sample_id,
        }
