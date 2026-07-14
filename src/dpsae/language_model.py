"""Causal-LM activation capture and SAE insertion utilities."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class ActivationStats:
    """Fixed centering and scalar scaling for one activation site."""

    mean: Tensor
    scale: Tensor

    def normalize(self, activation: Tensor) -> Tensor:
        return (activation.float() - self.mean) / self.scale

    def denormalize(self, activation: Tensor) -> Tensor:
        return activation * self.scale + self.mean

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean.detach().cpu(), "scale": self.scale.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor], device: torch.device) -> "ActivationStats":
        return cls(state["mean"].to(device), state["scale"].to(device))


def estimate_activation_stats(activations: Tensor) -> ActivationStats:
    mean = activations.float().mean(dim=0)
    centered = activations.float() - mean
    scale = centered.square().mean().sqrt().clamp_min(1e-6)
    return ActivationStats(mean=mean, scale=scale)


def _replace_block_output(output, hidden: Tensor):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def _transformer_blocks(model):
    """Return the residual blocks for the supported Hugging Face LM families."""

    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "h"):
        return transformer.h
    gpt_neox = getattr(model, "gpt_neox", None)
    if gpt_neox is not None and hasattr(gpt_neox, "layers"):
        return gpt_neox.layers
    raise TypeError(
        "unsupported causal LM: expected GPT-2 transformer.h or GPT-NeoX "
        "gpt_neox.layers"
    )


class GPT2ActivationModel:
    """Frozen GPT-2/GPT-NeoX LM with one residual capture/insertion site."""

    def __init__(self, model, tokenizer, *, layer: int, device: torch.device) -> None:
        self.blocks = _transformer_blocks(model)
        if layer < 1 or layer > len(self.blocks):
            raise ValueError("layer must use one-based transformer block numbering")
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.device = device
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        layer: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "GPT2ActivationModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(device)
        return cls(model, tokenizer, layer=layer, device=device)

    def autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @torch.inference_mode()
    def activations(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        input_ids = input_ids.to(self.device)
        mask = None if attention_mask is None else attention_mask.to(self.device)
        with self.autocast():
            output = self.model(
                input_ids=input_ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # hidden_states[0] is the embedding stream, so index L is resid_post L.
        return output.hidden_states[self.layer].float()

    @torch.inference_mode()
    def logits(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        replacement: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        input_ids = input_ids.to(self.device)
        mask = None if attention_mask is None else attention_mask.to(self.device)
        handle = None
        if replacement is not None:
            block = self.blocks[self.layer - 1]

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                replaced = replacement(hidden.float()).to(hidden.dtype)
                return _replace_block_output(output, replaced)

            handle = block.register_forward_hook(hook)
        try:
            with self.autocast():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    use_cache=False,
                    return_dict=True,
                )
            return output.logits.float()
        finally:
            if handle is not None:
                handle.remove()


def final_token_logits(logits: Tensor, attention_mask: Tensor) -> Tensor:
    end = attention_mask.to(logits.device).sum(dim=1) - 1
    return logits[torch.arange(logits.shape[0], device=logits.device), end]


def answer_logit_difference(
    logits: Tensor,
    attention_mask: Tensor,
    correct_token_id: Tensor,
    incorrect_token_id: Tensor,
) -> Tensor:
    final = final_token_logits(logits, attention_mask)
    rows = torch.arange(final.shape[0], device=final.device)
    correct = final[rows, correct_token_id.to(final.device)]
    incorrect = final[rows, incorrect_token_id.to(final.device)]
    return correct - incorrect
