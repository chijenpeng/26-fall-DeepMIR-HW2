"""Randomly initialized Hugging Face Transformers (trained from scratch)."""

from transformers import GPT2Config, GPT2LMHeadModel, T5Config, T5ForConditionalGeneration

from . import remi


def build_model(config):
    m = config["model"]
    common = {
        "vocab_size": len(remi.VOCAB),
        "pad_token_id": remi.PAD,
        "bos_token_id": remi.BOS,
        "eos_token_id": remi.EOS,
    }
    if config["task"] == "basic":
        return GPT2LMHeadModel(
            GPT2Config(
                **common,
                n_positions=config["max_len"][0],
                n_embd=m["width"],
                n_layer=m["layers"],
                n_head=m["heads"],
                attn_implementation="sdpa",
            )
        )
    return T5ForConditionalGeneration(
        T5Config(
            **common,
            d_model=m["width"],
            d_kv=m["width"] // m["heads"],
            d_ff=m["width"] * 4,
            num_layers=m["encoder_layers"],
            num_decoder_layers=m["decoder_layers"],
            num_heads=m["heads"],
            relative_attention_max_distance=256,
            decoder_start_token_id=remi.BOS,
        )
    )
