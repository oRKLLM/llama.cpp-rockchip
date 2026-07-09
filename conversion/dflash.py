from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .qwen import Qwen3Model


@ModelBase.register("DFlashDraftModel")
class DFlashDraftModel(Qwen3Model):
    """z-lab DFlash block-diffusion speculative-draft head.

    A small Qwen3 body plus an ``fc`` fusion of the concatenated target hidden states and a
    ``hidden_norm`` over the fused context. The draft borrows the target model's token
    embedding / output projection and tokenizer at runtime (ctx_other=target), so its
    safetensors carry neither ``tok_embd``/``lm_head`` nor a tokenizer of their own — those
    come from ``--target-model-dir``. Emits arch ``dflash`` with ``dflash.block_size`` and
    ``dflash.target_layers`` (the target layer indices whose hidden states ``fc`` fuses).
    """
    model_arch = gguf.MODEL_ARCH.DFLASH

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        dcfg = self.hparams.get("dflash_config", {})
        block_size = self.hparams.get("block_size", dcfg.get("block_size", 16))
        target_layers = dcfg.get("target_layer_ids") or self.hparams.get("target_layer_ids")
        if not target_layers:
            raise ValueError("DFlash config is missing dflash_config.target_layer_ids")
        arch = self.gguf_writer.arch
        self.gguf_writer.add_uint32(f"{arch}.block_size", int(block_size))
        self.gguf_writer.add_array(f"{arch}.target_layers", [int(x) for x in target_layers])
        logger.info(f"DFlash: block_size={block_size}, target_layers={list(target_layers)}")

    def set_vocab(self):
        # The draft borrows the target's tokenizer/embeddings; read the vocab from the target
        # model when one is supplied (the draft safetensors carry no tokenizer of their own).
        if self.target_model_dir is not None:
            logger.info(f"DFlash: using tokenizer from target model: {self.target_model_dir}")
            orig = self.dir_model
            self.dir_model = self.target_model_dir
            try:
                super().set_vocab()
            finally:
                self.dir_model = orig
        else:
            logger.warning("DFlash: no --target-model-dir given; reading tokenizer from the draft dir")
            super().set_vocab()

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        # DFlash safetensors name transformer blocks `layers.N.*` (no `model.` prefix); the
        # Qwen3 tensor map expects `model.layers.N.*`.
        tensors = super().index_tensors(remote_hf_model_id)
        renamed: dict[str, Callable[[], Tensor]] = {}
        for name, gen in tensors.items():
            renamed["model." + name if name.startswith("layers.") else name] = gen
        return renamed

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Top-level DFlash tensors the standard Qwen3 map doesn't cover:
        if name == "fc.weight":                       # fuses concatenated target hidden states -> n_embd
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.FC), data_torch)
            return
        if name == "hidden_norm.weight":              # RMSNorm over the fused context
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ENC_OUTPUT_NORM), data_torch)
            return
        if name == "norm.weight":                     # final norm (bare name, not `model.norm`)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch)
            return
        # `tok_embd`/`lm_head` are absent (borrowed from the target at runtime); everything
        # else is a standard Qwen3 block tensor.
        yield from super().modify_tensors(data_torch, name, bid)
