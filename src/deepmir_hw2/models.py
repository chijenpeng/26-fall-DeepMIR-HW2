"""Randomly initialized Transformers; checkpoint loading is always local."""

from transformers import GPT2Config, GPT2LMHeadModel, T5Config, T5ForConditionalGeneration

from .tokens import MusicTokenizer


def build_model(config: dict, tokenizer: MusicTokenizer):
    model = config["model"]
    common = {
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_id,
        "bos_token_id": tokenizer.bos_id,
        "eos_token_id": tokenizer.eos_id,
    }
    width, heads = model["width"], model["heads"]
    if width % heads:
        raise ValueError("Model width must be divisible by number of heads")
    if config["task"] == "basic":
        architecture = GPT2Config(
            **common,
            n_positions=config["data"]["max_tokens"],
            n_ctx=config["data"]["max_tokens"],
            n_embd=width,
            n_layer=model["layers"],
            n_head=heads,
            resid_pdrop=model.get("dropout", 0.1),
            embd_pdrop=model.get("dropout", 0.1),
            attn_pdrop=model.get("dropout", 0.1),
            use_cache=False,
        )
        architecture._attn_implementation = "sdpa"
        return GPT2LMHeadModel(architecture)
    if config["task"] != "bonus":
        raise ValueError("Task must be basic or bonus")
    architecture = T5Config(
        **common,
        d_model=width,
        d_kv=width // heads,
        d_ff=width * 4,
        num_layers=model["encoder_layers"],
        num_decoder_layers=model["decoder_layers"],
        num_heads=heads,
        dropout_rate=model.get("dropout", 0.1),
        decoder_start_token_id=tokenizer.bos_id,
        use_cache=False,
    )
    return T5ForConditionalGeneration(architecture)
