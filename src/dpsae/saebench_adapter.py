"""Exact adapter from native language SAEs to the pinned SAEBench interface.

The native models operate on centered, scalar-normalized residual activations.
SAEBench supplies raw TransformerLens residual activations, so normalization is
part of ``encode`` and denormalization is part of ``decode``. Decoder vectors
remain the checkpoint vectors; this module never folds or renormalizes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


def one_based_resid_post_hook(one_based_block: int) -> tuple[int, str]:
    """Map the repository's one-based block number to TransformerLens."""

    if not isinstance(one_based_block, int) or one_based_block < 1:
        raise ValueError("one_based_block must be a positive integer")
    hook_layer = one_based_block - 1
    return hook_layer, f"blocks.{hook_layer}.hook_resid_post"


@dataclass
class NativeSAEBenchConfig:
    """Dataclass-compatible subset expected by SAEBench custom SAE loaders."""

    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    context_size: int = 1024
    hook_head_index: int | None = None
    architecture: str = "batch_topk_native_threshold"
    apply_b_dec_to_input: bool = True
    finetuning_scaling_factor: bool = False
    activation_fn_str: str = "relu"
    activation_fn_kwargs: dict[str, Any] | None = None
    prepend_bos: bool = True
    normalize_activations: str = "none"
    dtype: str = "float32"
    device: str = "cpu"
    model_from_pretrained_kwargs: dict[str, Any] | None = None
    dataset_path: str = ""
    dataset_trust_remote_code: bool = True
    seqpos_slice: tuple[None] = (None,)
    training_tokens: int = -1
    sae_lens_training_version: str | None = None
    neuronpedia_id: str | None = None

    def __post_init__(self) -> None:
        if self.activation_fn_kwargs is None:
            self.activation_fn_kwargs = {}
        if self.model_from_pretrained_kwargs is None:
            self.model_from_pretrained_kwargs = {}

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


class NativeBatchTopKSAEBenchAdapter(nn.Module):
    """Read-only SAEBench-compatible view of one native BatchTopK checkpoint."""

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        activation_stats: Mapping[str, Tensor],
        model_name: str,
        one_based_block: int,
        context_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        decoder_norm_atol: float = 1e-5,
        expected_method: str | None = None,
        expected_d_in: int | None = None,
        expected_d_sae: int | None = None,
        expected_k: int | None = None,
    ) -> None:
        super().__init__()
        if dtype != torch.float32:
            raise ValueError("exp10 fixes adapter arithmetic to float32")
        state = payload.get("state_dict")
        spec = payload.get("spec")
        if not isinstance(state, Mapping) or not isinstance(spec, Mapping):
            raise ValueError("native SAE payload must contain spec and state_dict mappings")
        if payload.get("sparsity_mode", "batch_topk") != "batch_topk":
            raise ValueError("exp10 only admits the native BatchTopK pilot checkpoints")
        method = spec.get("method")
        if expected_method is not None and method != expected_method:
            raise ValueError(f"expected {expected_method!r} checkpoint, observed {method!r}")

        required = {
            "encoder_weight",
            "decoder_weight",
            "encoder_bias",
            "decoder_bias",
            "activation_threshold",
            "threshold_updates",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"native SAE state is missing keys: {missing}")

        W_enc = torch.as_tensor(state["encoder_weight"]).detach().float()
        W_dec = torch.as_tensor(state["decoder_weight"]).detach().float()
        b_enc = torch.as_tensor(state["encoder_bias"]).detach().float()
        b_dec = torch.as_tensor(state["decoder_bias"]).detach().float()
        threshold = torch.as_tensor(state["activation_threshold"]).detach().float()
        threshold_updates = torch.as_tensor(state["threshold_updates"]).detach()
        mean = torch.as_tensor(activation_stats.get("mean")).detach().float()
        scale = torch.as_tensor(activation_stats.get("scale")).detach().float()

        if W_enc.ndim != 2 or W_dec.ndim != 2:
            raise ValueError("native encoder and decoder weights must be matrices")
        d_in, d_sae = W_enc.shape
        if W_dec.shape != (d_sae, d_in):
            raise ValueError("native encoder/decoder shapes disagree")
        if b_enc.shape != (d_sae,) or b_dec.shape != (d_in,):
            raise ValueError("native bias shapes disagree with the weights")
        if mean.shape != (d_in,):
            raise ValueError("activation mean must have shape [d_in]")
        if scale.numel() != 1:
            raise ValueError("exp10 requires the stored scalar activation scale")
        if not torch.isfinite(scale).all() or float(scale) <= 0:
            raise ValueError("activation scale must be finite and positive")
        if threshold.numel() != 1 or not torch.isfinite(threshold).all():
            raise ValueError("BatchTopK inference threshold must be one finite scalar")
        if int(threshold_updates) <= 0:
            raise ValueError("BatchTopK checkpoint has no calibrated inference threshold")
        if expected_d_in is not None and d_in != expected_d_in:
            raise ValueError(f"expected d_in={expected_d_in}, observed {d_in}")
        if expected_d_sae is not None and d_sae != expected_d_sae:
            raise ValueError(f"expected d_sae={expected_d_sae}, observed {d_sae}")
        if expected_k is not None and int(spec.get("k", -1)) != expected_k:
            raise ValueError(f"expected k={expected_k}, observed {spec.get('k')}")

        norms = W_dec.norm(dim=1)
        if not torch.allclose(
            norms, torch.ones_like(norms), atol=decoder_norm_atol, rtol=0
        ):
            maximum = float((norms - 1).abs().max())
            raise ValueError(
                "native decoder vectors are not unit norm; adapter-side "
                f"renormalization is forbidden (max deviation {maximum:.3g})"
            )

        hook_layer, hook_name = one_based_resid_post_hook(one_based_block)
        self.W_enc = nn.Parameter(W_enc.to(device), requires_grad=False)
        self.W_dec = nn.Parameter(W_dec.to(device), requires_grad=False)
        self.b_enc = nn.Parameter(b_enc.to(device), requires_grad=False)
        self.b_dec = nn.Parameter(b_dec.to(device), requires_grad=False)
        self.register_buffer("activation_mean", mean.to(device))
        self.register_buffer("activation_scale", scale.reshape(()).to(device))
        self.register_buffer("activation_threshold", threshold.reshape(()).to(device))
        self.register_buffer("threshold_updates", threshold_updates.reshape(()).to(device))
        self.device = device
        self.dtype = dtype
        self.method = str(method)
        self.native_spec = dict(spec)
        self.cfg = NativeSAEBenchConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_layer=hook_layer,
            hook_name=hook_name,
            context_size=context_size,
            dtype="float32",
            device=str(device),
        )
        self.eval()

    def normalize_raw(self, activation: Tensor) -> Tensor:
        return (activation.float() - self.activation_mean) / self.activation_scale

    def denormalize_native(self, activation: Tensor) -> Tensor:
        return activation.float() * self.activation_scale + self.activation_mean

    def encode_normalized(self, normalized: Tensor) -> Tensor:
        scores = torch.relu(
            (normalized.float() - self.b_dec) @ self.W_enc + self.b_enc
        )
        # Native BatchTopKSAE.encode uses >= for its learned global threshold.
        return torch.where(scores >= self.activation_threshold, scores, 0)

    def decode_normalized(self, feature_acts: Tensor) -> Tensor:
        return feature_acts.float() @ self.W_dec + self.b_dec

    def encode(self, raw_activation: Tensor) -> Tensor:
        return self.encode_normalized(self.normalize_raw(raw_activation))

    def decode(self, feature_acts: Tensor) -> Tensor:
        return self.denormalize_native(self.decode_normalized(feature_acts))

    def forward(self, raw_activation: Tensor) -> Tensor:
        return self.decode(self.encode(raw_activation))

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                device = torch.device(first)
            elif isinstance(first, torch.dtype):
                dtype = first
        if dtype is not None and dtype != torch.float32:
            raise ValueError("exp10 adapter cannot be cast away from float32")
        result = super().to(*args, **kwargs)
        if device is not None:
            self.device = torch.device(device)
            self.cfg.device = str(self.device)
        self.dtype = torch.float32
        self.cfg.dtype = "float32"
        return result


def load_native_saebench_adapter(
    *,
    models_path: Path,
    calibration_path: Path,
    payload_name: str,
    model_name: str,
    one_based_block: int,
    context_size: int,
    device: torch.device,
    decoder_norm_atol: float,
    expected_method: str,
    expected_d_in: int,
    expected_d_sae: int,
    expected_k: int,
) -> NativeBatchTopKSAEBenchAdapter:
    """Load an adapter without mutating either source artifact."""

    payloads = torch.load(models_path, map_location="cpu", weights_only=False)
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
    if payload_name not in payloads:
        raise KeyError(f"models artifact has no payload named {payload_name!r}")
    activation_stats = calibration.get("activation_stats")
    if not isinstance(activation_stats, Mapping):
        raise ValueError("calibration artifact has no activation_stats mapping")
    return NativeBatchTopKSAEBenchAdapter(
        payload=payloads[payload_name],
        activation_stats=activation_stats,
        model_name=model_name,
        one_based_block=one_based_block,
        context_size=context_size,
        device=device,
        decoder_norm_atol=decoder_norm_atol,
        expected_method=expected_method,
        expected_d_in=expected_d_in,
        expected_d_sae=expected_d_sae,
        expected_k=expected_k,
    )
