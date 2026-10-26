import copy
from argparse import Namespace
from types import SimpleNamespace

import mido
import pretty_midi
import pytest
import torch

from deepmir_hw2.generate import CachedGenerator, ContextLimitError, SamplingConfig, generate_bars
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
    script = [tokenizer.bos_id, *empty, *dense, tokenizer.bar_id]
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


@pytest.mark.parametrize("audio", ["none", "success", "failure"])
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

    monkeypatch.setattr(generation, "validate_render_settings", lambda *args: None)
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
    )
    if audio == "failure":
        with pytest.raises(RuntimeError, match="renderer failure"):
            generation.generate(args)
        assert read_json(tmp_path / "output/generation.json")["complete"] is False
    else:
        generation.generate(args)
        metadata = read_json(tmp_path / "output/generation.json")
        assert metadata["complete"] is True
        assert metadata["audio_complete"] is (audio == "success")
    assert len(audio_calls) == (0 if audio == "none" else 1)
    assert model.encoder_calls == 1
    assert not any(token in tokenizer.groups["Chord"] for token in model.encoder_input_ids[0])
    result = mido.MidiFile(tmp_path / "output/001.mid")
    assert result.length == pytest.approx(32.0)
    assert (
        sum(event.type == "marker" and event.text.startswith("BAR_") for event in result.tracks[0])
        == 17
    )
