"""Packed multimodal batches must survive the worker/device boundary intact."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.molmoact2.worker_preprocessing import PreparedMolmoAct2Collator, device_only
from lerobot.processor import DeviceProcessorStep


def test_packed_inputs_and_nested_metadata_survive_device_transfer():
    pre = SimpleNamespace(steps=[DeviceProcessorStep(device="cpu", float_dtype="float32")])
    batch = {
        "input_ids": torch.tensor([[7, 8]], dtype=torch.long),
        "images": torch.ones(2, 3, dtype=torch.bfloat16),
        "token_pooling": torch.tensor([[1, 0]], dtype=torch.long),
        "action": torch.ones(1, 3, 2),
        "metadata": {"views": [torch.tensor([2]), "left"], "task": "fold"},
    }
    result = device_only(pre)(batch)
    assert result.keys() == batch.keys()
    assert torch.equal(result["input_ids"], batch["input_ids"])
    assert result["input_ids"].dtype == torch.long
    assert torch.equal(result["token_pooling"], batch["token_pooling"])
    assert result["images"].dtype == torch.float32
    assert batch["images"].dtype == torch.bfloat16  # No in-place mutation.
    assert torch.equal(result["metadata"]["views"][0], torch.tensor([2]))
    assert result["metadata"]["views"][1] == "left"
    assert result["metadata"]["task"] == "fold"


@pytest.mark.parametrize("steps", [[], [object()]])
def test_device_transfer_requires_a_native_final_device_step(steps):
    with pytest.raises(ValueError, match="final DeviceProcessorStep"):
        device_only(SimpleNamespace(steps=steps))


def test_worker_config_does_not_change_training_device():
    config = SimpleNamespace(device="cuda:3", options={"precision": "bfloat16"})
    collator = PreparedMolmoAct2Collator(config, {})
    assert config.device == "cuda:3"
    assert collator.config.device == "cpu"
    collator.config.options["precision"] = "float32"
    assert config.options["precision"] == "bfloat16"
    assert collator._pre is None  # Initialize the native processor only in its worker.


def test_worker_reuses_saved_processor_stats_and_native_collation(monkeypatch):
    from lerobot.policies.molmoact2 import worker_preprocessing as workers

    calls = []
    kwargs = {"preprocessor_overrides": {"device_processor": {"device": "cuda"}}}
    config = SimpleNamespace(device="cuda")

    def factory(**params):
        calls.append(params)
        return lambda batch: {**batch, "input_ids": torch.ones(2, 3, dtype=torch.long)}, None

    monkeypatch.setattr(workers, "make_pre_post_processors", factory)
    collator = workers.PreparedMolmoAct2Collator(
        config, kwargs, pretrained_path="checkpoint", pretrained_revision="revision"
    )
    rows = [{"action": torch.zeros(3, 2)}, {"action": torch.ones(3, 2)}]
    first = collator(rows)
    second = collator(rows)
    assert len(calls) == 1
    assert calls[0]["pretrained_path"] == "checkpoint"
    assert calls[0]["pretrained_revision"] == "revision"
    assert "dataset_stats" not in calls[0]  # Saved statistics stay authoritative on resume.
    assert calls[0]["preprocessor_overrides"]["device_processor"]["device"] == "cpu"
    assert kwargs["preprocessor_overrides"]["device_processor"]["device"] == "cuda"
    torch.testing.assert_close(first["action"], torch.stack([row["action"] for row in rows]))
    torch.testing.assert_close(second["input_ids"], first["input_ids"])


def test_worker_matches_native_uint8_camera_scaling(monkeypatch):
    from lerobot.policies.molmoact2 import worker_preprocessing as workers

    monkeypatch.setattr(workers, "make_pre_post_processors", lambda **_: (lambda batch: batch, None))
    collator = workers.PreparedMolmoAct2Collator(
        SimpleNamespace(device="cuda"), {}, camera_keys=["observation.images.front"]
    )
    pixels = torch.tensor([0, 127, 255], dtype=torch.uint8).reshape(3, 1, 1)
    batch = collator([{"observation.images.front": pixels}])
    torch.testing.assert_close(batch["observation.images.front"], pixels.unsqueeze(0).float() / 255)
    assert pixels.dtype == torch.uint8
