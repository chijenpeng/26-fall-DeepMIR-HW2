import copy

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from deepmir_hw2.attention import use_chunked_t5_attention


def models():
    torch.manual_seed(42)
    config = T5Config(
        vocab_size=97,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        dropout_rate=0,
        pad_token_id=0,
        decoder_start_token_id=1,
        eos_token_id=2,
        query_block_size=3,
    )
    original = T5ForConditionalGeneration(config)
    chunked = use_chunked_t5_attention(copy.deepcopy(original))
    return original, chunked


@pytest.mark.parametrize("outer_checkpoint", [False, True])
def test_chunked_t5_matches_eager_loss_and_all_parameter_gradients(outer_checkpoint):
    original, chunked = models()
    if outer_checkpoint:
        chunked.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    inputs = torch.tensor([[3, 4, 5, 6, 7, 0, 0], [8, 9, 10, 11, 12, 13, 14]])
    labels = torch.tensor([[15, 16, 17, 18, 2, -100], [19, 20, 21, 22, 23, 2]])
    outputs = [
        model(input_ids=inputs, attention_mask=inputs.ne(0), labels=labels, use_cache=False)
        for model in (original, chunked)
    ]
    torch.testing.assert_close(outputs[0].logits, outputs[1].logits, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(outputs[0].loss, outputs[1].loss)
    for output in outputs:
        output.loss.backward()
    for (name, expected), (_, actual) in zip(
        original.named_parameters(), chunked.named_parameters(), strict=True
    ):
        assert actual.grad is not None, name
        torch.testing.assert_close(expected.grad, actual.grad, rtol=2e-4, atol=2e-6, msg=name)


@torch.inference_mode()
def test_chunked_t5_cached_generation_matches_full_context_and_eager():
    original, chunked = models()
    original.eval()
    chunked.eval()
    inputs = torch.tensor([[3, 4, 5, 6, 7]])
    decoder = torch.tensor([[1, 8, 9, 10, 11, 12, 13]])
    expected = original(input_ids=inputs, decoder_input_ids=decoder, use_cache=False).logits
    encoder = chunked.get_encoder()(input_ids=inputs, return_dict=True)
    cache, parts = None, []
    for start, end in [(0, 2), (2, 3), (3, 6), (6, 7)]:
        output = chunked(
            encoder_outputs=encoder,
            decoder_input_ids=decoder[:, start:end],
            past_key_values=cache,
            use_cache=True,
        )
        parts.append(output.logits)
        cache = output.past_key_values
        assert cache.self_attention_cache.layers[0].keys.shape[-2] == end
        assert cache.cross_attention_cache.layers[0].keys.shape[-2] == inputs.shape[-1]
    torch.testing.assert_close(expected, torch.cat(parts, dim=1), rtol=1e-5, atol=2e-6)
