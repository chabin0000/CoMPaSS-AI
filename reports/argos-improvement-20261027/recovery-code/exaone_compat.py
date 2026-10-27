"""Compatibility for the pinned EXAONE 3.5 remote code, without changing weights."""
from __future__ import annotations


def configure_exaone_embeddings(model) -> str | None:
    """Tell Transformers' embedding accessors the vendor's actual layer name.

    The pinned v5 vendor code calls the input layer ``transformer.wte`` but
    omits the accessor hint. PEFT and k-bit input-gradient hooks require it.
    This only sets that hint; it does not replace modules or change tensors.
    Apply identically before training, S0 evaluation and adapter reloading.
    """
    if getattr(model.config, "model_type", None) != "exaone":
        return None
    import torch

    backbone = getattr(model, "transformer", None)
    embedding = getattr(backbone, "wte", None)
    head = getattr(model, "lm_head", None)
    if not isinstance(embedding, torch.nn.Embedding) or not isinstance(head, torch.nn.Linear):
        raise ValueError("EXAONE embedding layout differs from the pinned architecture")
    if embedding.weight.shape != (model.config.vocab_size, model.config.hidden_size):
        raise ValueError("EXAONE embedding dimensions differ from config")
    backbone._input_embed_layer = "wte"
    if model.get_input_embeddings() is not embedding or model.get_output_embeddings() is not head:
        raise ValueError("EXAONE embedding accessors do not resolve the original layers")
    return "exaone-transformer-wte-accessor-v1"
