"""Regression coverage for native normalization inside compiled training graphs."""

# ruff: noqa: E402

import copy

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from lerobot.policies.molmoact2.compile_norms import preserve_native_norms
from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import (
    ActionExpertRMSNorm,
    MolmoAct2RMSNorm,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and Inductor")
@pytest.mark.parametrize("kind", ["layer", "rms", "action_rms", "action_rms_affine"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_compiled_norm_matches_native_autograd(kind: str, dtype: torch.dtype, amp: bool, transposed: bool):
    torch.manual_seed(2026)
    if kind == "layer":
        reference = torch.nn.LayerNorm(128, eps=1e-6)
    elif kind == "rms":
        reference = MolmoAct2RMSNorm(128)
    else:
        reference = ActionExpertRMSNorm(128, elementwise_affine=kind == "action_rms_affine")
    reference = reference.cuda().to(dtype=dtype)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(0.5, 1.5)
    candidate = copy.deepcopy(reference)
    parameter_ids = [id(p) for p in candidate.parameters()]
    state_keys = tuple(candidate.state_dict())
    preserve_native_norms(candidate)
    assert [id(p) for p in candidate.parameters()] == parameter_ids
    assert tuple(candidate.state_dict()) == state_keys
    candidate.compile(fullgraph=True)

    shape = (2, 10, 8, 128) if transposed else (2, 17, 128)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    x = (x.transpose(1, 2) if transposed else x).requires_grad_(True)
    x_candidate = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        expected = reference(x)
    upstream = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (x, *reference.parameters()), upstream)

    # This test intentionally specializes on more dtype/layout/module combinations
    # than the compiler's default per-code-object cache permits.
    with torch._dynamo.config.patch(recompile_limit=64):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            actual = candidate(x_candidate)
        actual_grads = torch.autograd.grad(actual, (x_candidate, *candidate.parameters()), upstream)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        relative_error = (actual_grad.float() - expected_grad.float()).norm() / expected_grad.float().norm()
        assert relative_error < 5e-5
