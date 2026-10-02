"""Preserve eager normalization numerics inside regional training graphs.

Inductor's fused normalization reductions can perturb hybrid BF16 training
much more than repeated eager execution. Opaque operators retain the native
LayerNorm kernels and the model's FP32 RMSNorm arithmetic without graph breaks.
The registered backwards reproduce first-order eager autograd. Explicit output
layouts keep fake metadata correct for transposed attention tensors.
"""

from types import MethodType
from typing import Any

import torch


@torch.library.custom_op("lerobot_molmoact2_norm::rms", mutates_args=())
def rms(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float, weight_left: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    z = x.float()
    inverse = torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)
    out = z * inverse
    if weight is not None:
        out = weight.float() * out if weight_left else out * weight.float()
    return out.to(x.dtype).contiguous(), inverse.contiguous()


@rms.register_fake
def rms_fake(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float, weight_left: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x, memory_format=torch.contiguous_format), x.new_empty(
        (*x.shape[:-1], 1), dtype=torch.float32
    )


@torch.library.custom_op(
    "lerobot_molmoact2_norm::rms_backward",
    mutates_args=(),
    schema="(Tensor x, Tensor? weight, Tensor inverse, Tensor grad) -> (Tensor, Tensor?)",
)
def rms_backward(
    x: torch.Tensor, weight: torch.Tensor | None, inverse: torch.Tensor, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    z = x.float()
    upstream = grad.float()
    weight_grad = None
    if weight is not None:
        weight_grad = (upstream * (z * inverse)).sum_to_size(weight.shape).to(weight.dtype)
        upstream = upstream * weight.float()
    direct = upstream * inverse
    inverse_grad = (upstream * z).sum(-1, keepdim=True)
    variance_grad = (-0.5 * inverse_grad) * inverse.pow(3)
    square_grad = variance_grad.expand_as(z) / z.shape[-1]
    indirect = square_grad * (2 * z)
    return (direct + indirect).to(
        x.dtype
    ).contiguous(), weight_grad.contiguous() if weight_grad is not None else None


@rms_backward.register_fake
def rms_backward_fake(
    x: torch.Tensor, weight: torch.Tensor | None, inverse: torch.Tensor, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return torch.empty_like(x, memory_format=torch.contiguous_format), torch.empty_like(
        weight, memory_format=torch.contiguous_format
    ) if weight is not None else None


def rms_setup(ctx: Any, inputs: tuple, output: tuple) -> None:
    x, weight, _, _ = inputs
    ctx.save_for_backward(x, weight, output[1])
    ctx.mark_non_differentiable(output[1])


def rms_autograd(ctx: Any, grad: torch.Tensor, unused: torch.Tensor | None) -> tuple:
    x, weight, inverse = ctx.saved_tensors
    dx, dw = rms_backward(x, weight, inverse, grad)
    return dx, dw, None, None


rms.register_autograd(rms_autograd, setup_context=rms_setup)


@torch.library.custom_op("lerobot_molmoact2_norm::layer", mutates_args=())
def layer(
    x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, shape: list[int], eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.aten.native_layer_norm.default(x, shape, weight, bias, eps)


@layer.register_fake
def layer_fake(
    x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, shape: list[int], eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stats_shape = (*x.shape[: -len(shape)], *((1,) * len(shape)))
    stats_dtype = torch.float32 if x.dtype in (torch.bfloat16, torch.float16) else x.dtype
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        x.new_empty(stats_shape, dtype=stats_dtype),
        x.new_empty(stats_shape, dtype=stats_dtype),
    )


@torch.library.custom_op(
    "lerobot_molmoact2_norm::layer_backward",
    mutates_args=(),
    schema="(Tensor x, Tensor? weight, Tensor? bias, Tensor mean, Tensor rstd, Tensor grad, SymInt[] shape) -> (Tensor, Tensor?, Tensor?)",
)
def layer_backward(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    grad: torch.Tensor,
    shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    return torch.ops.aten.native_layer_norm_backward.default(
        grad, x, shape, mean, rstd, weight, bias, [True, weight is not None, bias is not None]
    )


@layer_backward.register_fake
def layer_backward_fake(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    grad: torch.Tensor,
    shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(weight, memory_format=torch.contiguous_format) if weight is not None else None,
        torch.empty_like(bias, memory_format=torch.contiguous_format) if bias is not None else None,
    )


def layer_setup(ctx: Any, inputs: tuple, output: tuple) -> None:
    x, weight, bias, shape, _ = inputs
    ctx.save_for_backward(x, weight, bias, output[1], output[2])
    ctx.shape = shape
    ctx.mark_non_differentiable(output[1], output[2])


def layer_autograd(
    ctx: Any, grad: torch.Tensor, unused_mean: torch.Tensor | None, unused_rstd: torch.Tensor | None
) -> tuple:
    x, weight, bias, mean, rstd = ctx.saved_tensors
    dx, dw, db = layer_backward(x, weight, bias, mean, rstd, grad, ctx.shape)
    return dx, dw, db, None, None


layer.register_autograd(layer_autograd, setup_context=layer_setup)


def rms_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    return rms(x, self.weight, self.eps, type(self).__name__ == "MolmoAct2RMSNorm")[0]


def layer_forward(self: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    weight, bias = self.weight, self.bias
    if torch.is_autocast_enabled(x.device.type):
        x = x.float()
        weight = weight.float() if weight is not None else None
        bias = bias.float() if bias is not None else None
    return layer(x, weight, bias, list(self.normalized_shape), self.eps)[0]


def preserve_native_norms(policy: torch.nn.Module) -> tuple[str, ...]:
    """Replace instance forwards only; parameters and state-dict keys are unchanged."""
    names = []
    for name, module in policy.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.forward = MethodType(layer_forward, module)
            names.append(name)
        elif type(module).__name__ in ("MolmoAct2RMSNorm", "ActionExpertRMSNorm"):
            module.forward = MethodType(rms_forward, module)
            names.append(name)
    return tuple(names)
