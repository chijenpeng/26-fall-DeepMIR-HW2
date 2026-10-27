import copy
import json
import math
from argparse import Namespace
from types import SimpleNamespace

import mido
import pretty_midi
import pytest
import torch

from deepmir_hw2.generate import (
    CachedGenerator,
    ContextLimitError,
    RawBarSidecar,
    SamplingConfig,
    beam_integer,
    generate_bars,
    sampling_metadata,
)
from deepmir_hw2.models import build_model
from deepmir_hw2.music import BarTiming
from deepmir_hw2.tokens import MusicTokenizer, TokenizerConfig


class ScriptModel:
    """Known next-token logits exercise the actual sampler, grammar and cache."""

    def __init__(self, script, vocabulary):
        self.script, self.vocabulary = script, vocabulary
        self.encoder_calls = 0

    def get_encoder(self):
        def encode(**kwargs):
            self.encoder_calls += 1
            self.encoder_input_ids = kwargs["input_ids"]
            return SimpleNamespace(last_hidden_state=torch.zeros(1, 1, 1))

        return encode

    def load_state_dict(self, state):
        return None

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids=None, decoder_input_ids=None, past_key_values=None, **kwargs):
        ids = input_ids if input_ids is not None else decoder_input_ids
        seen = (past_key_values or 0) + ids.shape[1]
        target = self.script[seen]
        logits = torch.full((1, ids.shape[1], self.vocabulary), -1000.0)
        logits[0, -1, target] = 1000.0
        return SimpleNamespace(logits=logits, past_key_values=seen)


class FinalEosGateModel(ScriptModel):
    def __init__(self, script, vocabulary, gate_logits, tokenizer):
        super().__init__(script, vocabulary)
        self.gate_logits = gate_logits
        self.tokenizer = tokenizer

    def __call__(self, input_ids=None, decoder_input_ids=None, past_key_values=None, **kwargs):
        output = super().__call__(input_ids, decoder_input_ids, past_key_values, **kwargs)
        seen = output.past_key_values
        if seen in self.gate_logits:
            logits = output.logits[0, -1]
            logits.fill_(-100.0)
            bar_logit, pos_logit, eos_logit = self.gate_logits[seen]
            logits[self.tokenizer.bar_id] = bar_logit
            logits[self.tokenizer.tok2idx["Pos"]] = pos_logit
            logits[self.tokenizer.eos_id] = eos_logit
        return output


class CompetingIntegerModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids=None, decoder_input_ids=None, past_key_values=None, **kwargs):
        ids = input_ids if input_ids is not None else decoder_input_ids
        seen = (past_key_values or 0) + ids.shape[1]
        logits = torch.zeros((1, ids.shape[1], len(self.tokenizer)))
        if seen == 1:
            logits.fill_(-100.0)
            logits[0, -1, self.tokenizer.tok2idx["More_1"]] = math.log(0.6)
            logits[0, -1, self.tokenizer.tok2idx["End_50"]] = math.log(0.4)
        return SimpleNamespace(logits=logits, past_key_values=seen)


class LegacyTupleCacheModel:
    def __call__(self, input_ids=None, decoder_input_ids=None, past_key_values=None, **kwargs):
        ids = input_ids if input_ids is not None else decoder_input_ids
        new_keys = ids.float().view(1, 1, -1, 1)
        if past_key_values is None:
            keys = new_keys
        else:
            keys = torch.cat([past_key_values[0][0], new_keys], dim=-2)
        cache = ((keys, keys),)
        logits = torch.zeros((1, ids.shape[1], 512))
        return SimpleNamespace(logits=logits, past_key_values=cache)


@pytest.mark.parametrize("task", ["basic", "bonus"])
def test_silence_nonfour_meter_and_duplicate_dense_onsets_are_not_forbidden(task):
    tokenizer = MusicTokenizer(TokenizerConfig((2, 6), ((3, 4),)))
    empty = tokenizer.header(BarTiming((200_000,) * 2, 3, 4))
    dense = tokenizer.header(BarTiming((300_000,) * 6, 3, 4))
    event = [
        tokenizer.tok2idx["Pos"],
        *tokenizer.encode_uint(0),
        tokenizer.tok2idx["Pitch_60"],
        tokenizer.tok2idx["Vel_17"],
        tokenizer.tok2idx["Duration"],
        *tokenizer.encode_uint(30_000_000),
    ]
    dense += event * 130
    expected = [empty, dense]
    script = [tokenizer.bos_id, *empty, *dense, tokenizer.eos_id]
    model = ScriptModel(script, len(tokenizer))
    output, _ = generate_bars(
        model,
        tokenizer,
        task,
        2,
        len(script),
        "cpu",
        SamplingConfig(top_k=1),
        42,
        melody=[tokenizer.bos_id, *empty, tokenizer.eos_id] if task == "bonus" else None,
        conditioned_headers=expected if task == "bonus" else None,
    )
    assert output == expected
    assert len(tokenizer.events(output[0])) == 0
    assert len(tokenizer.events(output[1])) == 130
    assert tokenizer.events(output[1])[0][2][2] == 30_000_000
    if task == "bonus":
        assert model.encoder_calls == 1


def test_context_exhaustion_raises_instead_of_dropping_history():
    tokenizer = MusicTokenizer()
    model = ScriptModel([0] * 10, len(tokenizer))
    engine = CachedGenerator(model, tokenizer, "basic", 4, "cpu", [tokenizer.bos_id])
    engine.append([tokenizer.bar_id] * 3)
    before = list(engine.history)
    with pytest.raises(ContextLimitError, match="discarded"):
        engine.append([tokenizer.bar_id])
    assert engine.history == before


def test_trace_reconstructs_complete_generated_note_bar(tmp_path):
    tokenizer = MusicTokenizer()
    header = tokenizer.header(BarTiming((500_000,) * 4, 4, 4))
    expected = [
        *header,
        tokenizer.tok2idx["Pos"],
        *tokenizer.encode_uint(125_000),
        tokenizer.tok2idx["Pitch_60"],
        tokenizer.tok2idx["Vel_80"],
        tokenizer.tok2idx["Duration"],
        *tokenizer.encode_uint(30_000_000),
    ]
    model = ScriptModel([tokenizer.bos_id, *expected, tokenizer.eos_id], len(tokenizer))
    sidecar = RawBarSidecar(tmp_path / "song.bars.json", "song", "beam")
    bars, _ = generate_bars(
        model,
        tokenizer,
        "basic",
        1,
        len(expected) + 1,
        "cpu",
        SamplingConfig(top_k=1, numeric_strategy="beam"),
        321,
        on_bar=lambda bar, _, tokens, complete: sidecar.record(bar, tokens, complete),
    )
    saved = json.loads((tmp_path / "song.bars.json").read_text())
    assert bars[0] == expected
    assert saved["bars"] == [{"bar": 1, "tokens": expected, "complete": True}]
    assert saved["beam_width"] == 4


def test_beam_strategy_metadata_pins_width_four():
    assert sampling_metadata(SamplingConfig()) == {
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
        "numeric_strategy": "sample",
        "beam_width": None,
    }
    assert sampling_metadata(SamplingConfig(numeric_strategy="beam"))["beam_width"] == 4


def test_bar_is_forbidden_at_final_gate_and_eos_is_forbidden_earlier(monkeypatch, tmp_path):
    tokenizer = MusicTokenizer()
    header = tokenizer.header(BarTiming((500_000,) * 4, 4, 4))
    script = [tokenizer.bos_id, *header, *header, tokenizer.eos_id]
    gate_positions = [1 + len(header), 1 + 2 * len(header)]
    # EOS dominates the first gate but is forbidden there. At the final gate,
    # Bar is the model's top choice; it must be excluded so EOS ends the piece.
    gate_logits = {gate_positions[0]: (3.0, 2.0, 10.0), gate_positions[1]: (100.0, 3.0, 10.0)}
    model = FinalEosGateModel(script, len(tokenizer), gate_logits, tokenizer)
    sidecar = RawBarSidecar(tmp_path / "piece.bars.json", "piece", "sample")
    from deepmir_hw2 import generate as generation

    original_sample_token = generation.sample_token
    gate_choices = []

    def record_gate_choice(logits, allowed, sampling, rng):
        gate_tokens = {tokenizer.bar_id, tokenizer.tok2idx["Pos"], tokenizer.eos_id}
        if set(allowed).issubset(gate_tokens):
            selected = original_sample_token(logits, allowed, sampling, rng)
            gate_choices.append((list(allowed), selected))
            return selected
        return original_sample_token(logits, allowed, sampling, rng)

    monkeypatch.setattr(generation, "sample_token", record_gate_choice)
    bars, diagnostic = generate_bars(
        model,
        tokenizer,
        "basic",
        2,
        len(script),
        "cpu",
        SamplingConfig(top_k=1),
        91,
        on_bar=lambda bar, _, tokens, complete: sidecar.record(bar, tokens, complete),
    )
    assert bars == [header, header]
    assert diagnostic["context_tokens"] == len(script) - 1
    raw_bars = json.loads((tmp_path / "piece.bars.json").read_text())
    assert [row["tokens"] for row in raw_bars["bars"]] == bars
    assert all(row["complete"] for row in raw_bars["bars"])
    assert raw_bars["beam_width"] is None
    assert tokenizer.eos_id not in [token for bar in bars for token in bar]
    assert gate_choices == [
        ([tokenizer.tok2idx["Pos"], tokenizer.bar_id], tokenizer.bar_id),
        ([tokenizer.tok2idx["Pos"], tokenizer.eos_id], tokenizer.eos_id),
    ]


def test_integer_beam_selects_complete_integer_with_best_joint_probability():
    tokenizer = MusicTokenizer()
    model = CompetingIntegerModel(tokenizer)
    lower, upper = 1, 100_000
    beam_engine = CachedGenerator(model, tokenizer, "basic", 32, "cpu", [tokenizer.bos_id])
    _, beam_value, beam_tokens, score = beam_integer(
        beam_engine,
        tokenizer,
        lower,
        upper,
        SamplingConfig(temperature=1.0, numeric_strategy="beam"),
    )
    assert beam_value == 50
    assert beam_tokens == (tokenizer.tok2idx["End_50"],)
    assert tokenizer.read_uint(list(beam_tokens), 0)[0] == beam_value
    assert score == pytest.approx(math.log(0.4))


def test_integer_beam_has_no_arbitrary_digit_length_cap():
    tokenizer = MusicTokenizer()
    target = tokenizer.encode_uint((1 << 63) - 1)

    class MaxIntegerModel:
        def __call__(self, input_ids=None, decoder_input_ids=None, past_key_values=None, **kwargs):
            ids = input_ids if input_ids is not None else decoder_input_ids
            seen = (past_key_values or 0) + ids.shape[1]
            digit_index = seen - 1
            logits = torch.full((1, ids.shape[1], len(tokenizer)), -100.0)
            logits[0, -1, target[digit_index]] = 100.0
            return SimpleNamespace(logits=logits, past_key_values=seen)

    model = MaxIntegerModel()
    engine = CachedGenerator(model, tokenizer, "basic", 32, "cpu", [tokenizer.bos_id])
    _, value, tokens, _ = beam_integer(
        engine,
        tokenizer,
        1,
        (1 << 63) - 1,
        SamplingConfig(numeric_strategy="beam"),
    )
    assert value == (1 << 63) - 1
    assert tokens == tuple(target)
    assert len(tokens) == 9


@pytest.mark.parametrize("task", ["basic", "bonus"])
def test_actual_model_cache_matches_full_history(tiny_config, task):
    tokenizer = MusicTokenizer()
    config = copy.deepcopy(tiny_config)
    config["task"] = task
    config["model"] = {
        "width": 16,
        "heads": 2,
        "dropout": 0.0,
        **({"layers": 1} if task == "basic" else {"encoder_layers": 1, "decoder_layers": 1}),
    }
    model = build_model(config, tokenizer).eval()
    header = tokenizer.header(BarTiming((500_000,) * 4, 4, 4))
    prefix, token = [tokenizer.bos_id, *header[:-1]], header[-1]
    melody = [tokenizer.bos_id, *header, tokenizer.eos_id]
    with torch.inference_mode():
        engine = CachedGenerator(
            model, tokenizer, task, 256, "cpu", prefix, melody if task == "bonus" else None
        )
        engine.logits()
        engine.append([token])
        actual = engine.logits()
        ids = torch.tensor([[*prefix, token]])
        if task == "basic":
            expected = model(input_ids=ids, use_cache=False).logits[0, -1]
        else:
            expected = model(
                input_ids=torch.tensor([melody]), decoder_input_ids=ids, use_cache=False
            ).logits[0, -1]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("task", ["basic", "bonus"])
def test_cache_fork_shares_kv_without_mutating_parent_and_matches_full_history(tiny_config, task):
    tokenizer = MusicTokenizer()
    config = copy.deepcopy(tiny_config)
    config["task"] = task
    config["model"] = {
        "width": 16,
        "heads": 2,
        "dropout": 0.0,
        **({"layers": 1} if task == "basic" else {"encoder_layers": 1, "decoder_layers": 1}),
    }
    model = build_model(config, tokenizer).eval()
    header = tokenizer.header(BarTiming((500_000,) * 4, 4, 4))
    prefix = [tokenizer.bos_id, *header]
    melody = [tokenizer.bos_id, *header, tokenizer.eos_id]
    engine = CachedGenerator(
        model, tokenizer, task, 256, "cpu", prefix, melody if task == "bonus" else None
    )
    with torch.inference_mode():
        engine.logits()
        parent_cache = engine.past
        parent_length = (
            parent_cache.self_attention_cache.get_seq_length()
            if task == "bonus"
            else parent_cache.get_seq_length()
        )
        child = engine.fork()
        assert child.past is not parent_cache
        if task == "bonus":
            assert child.past.self_attention_cache is not parent_cache.self_attention_cache
            assert (
                child.past.self_attention_cache.layers[0].keys
                is parent_cache.self_attention_cache.layers[0].keys
            )
            assert child.past.is_updated is not parent_cache.is_updated
        else:
            assert child.past.layers[0].keys is parent_cache.layers[0].keys

        branch_tokens = [tokenizer.tok2idx["Pitch_60"], tokenizer.tok2idx["Vel_80"]]
        child.append(branch_tokens)
        actual_branch = child.logits()
        assert engine.past is parent_cache
        assert engine.history == prefix
        assert (
            parent_cache.self_attention_cache.get_seq_length()
            if task == "bonus"
            else parent_cache.get_seq_length()
        ) == parent_length

        full_branch = [*prefix, *branch_tokens]
        if task == "basic":
            expected_branch = model(input_ids=torch.tensor([full_branch]), use_cache=False).logits[
                0, -1
            ]
        else:
            expected_branch = model(
                input_ids=torch.tensor([melody]),
                decoder_input_ids=torch.tensor([full_branch]),
                use_cache=False,
            ).logits[0, -1]
        torch.testing.assert_close(actual_branch, expected_branch)

        next_token = tokenizer.tok2idx["Pitch_61"]
        engine.append([next_token])
        actual_parent = engine.logits()
        full_parent = [*prefix, next_token]
        if task == "basic":
            expected_parent = model(input_ids=torch.tensor([full_parent]), use_cache=False).logits[
                0, -1
            ]
        else:
            expected_parent = model(
                input_ids=torch.tensor([melody]),
                decoder_input_ids=torch.tensor([full_parent]),
                use_cache=False,
            ).logits[0, -1]
        torch.testing.assert_close(actual_parent, expected_parent)


def test_legacy_tuple_cache_fork_shares_tensor_leaves_without_mutating_parent():
    tokenizer = MusicTokenizer()
    engine = CachedGenerator(
        LegacyTupleCacheModel(), tokenizer, "basic", 32, "cpu", [tokenizer.bos_id]
    )
    with torch.inference_mode():
        engine.logits()
        parent_cache = engine.past
        parent_keys = parent_cache[0][0]
        child = engine.fork()
        assert child.past is not parent_cache
        assert child.past[0] is not parent_cache[0]
        assert child.past[0][0] is parent_keys
        child.append([tokenizer.tok2idx["Pitch_60"], tokenizer.tok2idx["Vel_80"]])
        child.logits()
        assert engine.past is parent_cache
        assert engine.past[0][0] is parent_keys
        assert parent_keys.shape[-2] == 1
        assert child.past[0][0].shape[-2] == 3


@pytest.mark.parametrize(
    "audio",
    [
        "none",
        "success",
        "failure",
        "generation-failure",
        "prompt-failure",
        "export-failure",
        "preflight-failure",
    ],
)
def test_bonus_inference_never_reads_chords_or_reference_piano(
    tmp_path, monkeypatch, tiny_config, audio
):
    import deepmir_hw2.generate as generation
    from deepmir_hw2.utils import read_json

    prompt = tmp_path / "prompts/001"
    prompt.mkdir(parents=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    melody = pretty_midi.Instrument(0, name="MELODY")
    melody.notes = [pretty_midi.Note(81, 60, 2 * bar + 0.125, 2 * bar + 0.75) for bar in range(16)]
    midi.instruments.append(melody)
    midi.write(str(prompt / "melody.mid"))
    (prompt / "beat_midi.txt").write_text(
        "\n".join(f"{i / 2} 0 {int(i % 4 == 0)}" for i in range(65))
    )
    (prompt / "chord_midi.txt").write_text("must not be read")
    (prompt / "piano.mid").write_text("must not be read")
    tokenizer = MusicTokenizer(TokenizerConfig((4,), ((4, 4),), ("C:maj",)))
    header = tokenizer.header(BarTiming((500_000,) * 4, 4, 4))
    model = ScriptModel([tokenizer.bos_id, *(header * 16), tokenizer.bar_id], len(tokenizer))
    config = copy.deepcopy(tiny_config)
    config["task"] = "bonus"
    config["data"]["max_tokens"] = 1024
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint path")
    monkeypatch.setattr(
        generation,
        "load_checkpoint",
        lambda path: {
            "config": config,
            "tokenizer_config": tokenizer.config.__dict__,
            "tokenizer_tokens": tokenizer.tokens,
            "source_programs": {"piano": [0]},
            "model": {},
            "step": 0,
        },
    )
    monkeypatch.setattr(generation, "build_model", lambda *args: model)

    def reject_chord_access(*args):
        raise AssertionError("Bonus inference read chord annotations")

    monkeypatch.setattr(generation, "load_chords", reject_chord_access)
    audio_calls = []

    def fake_render(args):
        metadata = read_json(tmp_path / "output/generation.json")
        assert metadata["midi_complete"] is True
        assert metadata["complete"] is False
        audio_calls.append(args)
        if audio == "failure":
            raise RuntimeError("synthetic renderer failure")

    validation_calls = []

    def fake_validate(*args):
        validation_calls.append(args)
        if audio == "preflight-failure":
            raise RuntimeError("render preflight failure")

    monkeypatch.setattr(generation, "validate_render_settings", fake_validate)
    monkeypatch.setattr(generation, "render", fake_render)
    args = Namespace(
        checkpoint=str(checkpoint),
        mode="accompany",
        prompts=str(prompt.parent),
        output=str(tmp_path / "output"),
        count=20,
        bars=None,
        seed=42,
        temperature=0.9,
        top_k=1,
        top_p=1.0,
        device="cpu",
        soundfont=None if audio == "none" else str(tmp_path / "test.sf2"),
        fluidsynth="fluidsynth",
        sample_rate=44100,
        trace=True,
    )
    if audio == "preflight-failure":
        generation_calls = []
        original_generate = generation.generate_bars

        def track_generation(*args, **kwargs):
            generation_calls.append(True)
            return original_generate(*args, **kwargs)

        monkeypatch.setattr(generation, "generate_bars", track_generation)
        with pytest.raises(RuntimeError, match="preflight"):
            generation.generate(args)
        manifest = read_json(tmp_path / "output/generation.json")
        assert manifest["status"] == "failed"
        assert manifest["complete"] is False
        assert manifest["failure"]["stage"] == "render_preflight"
        assert manifest["failure"]["song"] is None
        assert validation_calls
        assert not generation_calls
        assert not (tmp_path / "output/001.mid").exists()
        return
    if audio in {"generation-failure", "prompt-failure", "export-failure"}:
        if audio == "generation-failure":

            def fail_generation(*args, **kwargs):
                kwargs["on_bar"](1, 1, [tokenizer.bar_id], False)
                raise RuntimeError("generation failure")

            monkeypatch.setattr(
                generation,
                "generate_bars",
                fail_generation,
            )
        elif audio == "prompt-failure":
            monkeypatch.setattr(
                generation.BeatGrid,
                "load",
                classmethod(
                    lambda cls, *args: (_ for _ in ()).throw(RuntimeError("prompt failure"))
                ),
            )
        else:
            monkeypatch.setattr(
                generation,
                "write_midi",
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("export failure")),
            )
        expected_stage = {
            "generation-failure": "generation",
            "prompt-failure": "prompt_parse",
            "export-failure": "decode_midi_export",
        }[audio]
        with pytest.raises(RuntimeError):
            generation.generate(args)
        manifest = read_json(tmp_path / "output/generation.json")
        assert manifest["status"] == "failed"
        assert manifest["complete"] is False
        assert manifest["failure"]["stage"] == expected_stage
        assert manifest["failure"]["song"] == "001"
        assert manifest["current_song"] == "001"
        assert (tmp_path / "output/001.bars.json").exists() is (audio != "prompt-failure")
        if audio == "generation-failure":
            partial = read_json(tmp_path / "output/001.bars.json")
            assert partial["bars"] == [{"bar": 1, "tokens": [tokenizer.bar_id], "complete": False}]
        assert not audio_calls
        return
    if audio == "failure":
        with pytest.raises(RuntimeError, match="renderer failure"):
            generation.generate(args)
        failure_manifest = read_json(tmp_path / "output/generation.json")
        assert failure_manifest["status"] == "failed"
        assert failure_manifest["complete"] is False
        assert failure_manifest["failure"]["stage"] == "render"
        assert failure_manifest["failure"]["song"] is None
        assert read_json(tmp_path / "output/generation.json")["complete"] is False
    else:
        generation.generate(args)
        metadata = read_json(tmp_path / "output/generation.json")
        assert metadata["complete"] is True
        assert metadata["sampling"]["numeric_strategy"] == "sample"
        assert metadata["audio_complete"] is (audio == "success")
    assert len(audio_calls) == (0 if audio == "none" else 1)
    assert model.encoder_calls == 1
    raw_bars = read_json(tmp_path / "output/001.bars.json")
    assert raw_bars["beam_width"] is None
    assert [bar["tokens"] for bar in raw_bars["bars"]] == [header] * 16
    assert all(bar["complete"] for bar in raw_bars["bars"])
    assert not any(token in tokenizer.groups["Chord"] for token in model.encoder_input_ids[0])
    result = mido.MidiFile(tmp_path / "output/001.mid")
    assert result.length == pytest.approx(32.0)
    assert (
        sum(event.type == "marker" and event.text.startswith("BAR_") for event in result.tracks[0])
        == 17
    )
