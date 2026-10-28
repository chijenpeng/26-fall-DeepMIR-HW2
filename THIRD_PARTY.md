# Data, code and citations

- **POP909** (course-provided; Wang et al., *POP909: A Pop-song Dataset for Music
  Arrangement Generation*, ISMIR 2020). Only `POP909/training/` (896 songs) is used for
  training. Not included in this repository.
- **HW2_scripts/** — TA-provided `tokenization.py`, `eval_basic.py`, `eval_bonus.py`,
  unmodified. `evaluate.py` calls the metric functions/scripts directly.
- **Hugging Face Transformers** 4.57.1 (Apache-2.0): `GPT2LMHeadModel` and
  `T5ForConditionalGeneration`, constructed with random weights (no pretrained weights).
- **MusDr** (MIT, https://github.com/slSeanWU/MusDr), used by `eval_basic.py` for
  H1/H4/GS; Wu and Yang, *The Jazz Transformer on the Front Line*, ISMIR 2020.
- **REMI**: Huang and Yang, *Pop Music Transformer*, ACM Multimedia 2020.
- **Bonus metrics**: Ren et al., *PopMAG: Pop Music Accompaniment Generation*,
  ACM Multimedia 2020.
- **TimGM6mb** SoundFont (GPL-2, Tim Brechbill / David Bolton), from the Debian
  `timgm6mb-soundfont` package, used only to render WAV listening samples.

Code written with AI assistance (OpenAI Codex, Anthropic Claude).
