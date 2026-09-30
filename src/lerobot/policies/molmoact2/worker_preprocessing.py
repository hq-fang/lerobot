"""Overlap native MolmoAct2 input packing with training in CPU data workers."""

import platform
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils._pytree import tree_map
from torch.utils.data import default_collate

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline


class PreparedMolmoAct2Collator:
    """Lazily build the same native processors, including saved resume statistics."""

    def __init__(
        self,
        config: MolmoAct2Config,
        processor_kwargs: dict[str, Any],
        pretrained_path: str | Path | None = None,
        pretrained_revision: str | None = None,
        collate_fn: Callable = default_collate,
        camera_keys: Sequence[str] = (),
    ) -> None:
        self.config = deepcopy(config)
        self.config.device = "cpu"
        self.processor_kwargs = deepcopy(processor_kwargs)
        overrides = self.processor_kwargs.setdefault("preprocessor_overrides", {})
        overrides.setdefault("device_processor", {})["device"] = "cpu"
        self.pretrained_path = pretrained_path
        self.pretrained_revision = pretrained_revision
        self.collate_fn = collate_fn
        self.camera_keys = tuple(camera_keys)
        self._pre: PolicyProcessorPipeline | None = None

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if self._pre is None:
            self._pre, _ = make_pre_post_processors(
                policy_cfg=self.config,
                pretrained_path=self.pretrained_path,
                pretrained_revision=self.pretrained_revision,
                **self.processor_kwargs,
            )
        batch = self.collate_fn(rows)
        # Match the native trainer's handling of datasets exposing uint8 cameras.
        for key in self.camera_keys:
            if key in batch and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].to(dtype=torch.float32) / 255.0
        return self._pre(batch)


class PreparedBatchToDevice:
    def __init__(self, step: DeviceProcessorStep) -> None:
        self.step = step

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        # Re-entering the policy pipeline would discard packed HF keys during
        # batch_to_transition. Transfer the complete dictionary without repacking.
        return tree_map(lambda x: self.step._process_tensor(x) if isinstance(x, torch.Tensor) else x, batch)


def device_only(preprocessor: PolicyProcessorPipeline) -> PreparedBatchToDevice:
    """Transfer packed inputs without rerunning normalization or tokenization."""
    if not preprocessor.steps or not isinstance(preprocessor.steps[-1], DeviceProcessorStep):
        raise ValueError("Expected a final DeviceProcessorStep in the native preprocessing pipeline")
    return PreparedBatchToDevice(preprocessor.steps[-1])


def init_molmoact2_cpu_worker(worker_id: int) -> None:
    """Avoid the oneDNN Arm JIT error in torchvision sharpness convolution."""
    if platform.machine().lower() in {"aarch64", "arm64"}:
        torch.backends.mkldnn.enabled = False
