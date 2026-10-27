# Data and source provenance

- `POP909/` is the course-provided POP909 dataset (Wang et al., ISMIR 2020).
  It is local-only and excluded from Git. Training uses only its `training/` partition.
- `HW2_scripts/` contains the unmodified examples supplied by the course TA.
  Their authorship remains with the course/TA. The new pipeline uses the TA bonus
  evaluator directly and follows its expected file/track layout. The tokenizer in
  `src/deepmir_hw2/` is a separate implementation inspired by the supplied example.
- GPT-2 and T5 architectures are implemented by Hugging Face Transformers:
  https://github.com/huggingface/transformers (Apache-2.0). This project uses version
  4.57.1 and constructs both models with random weights.
  `attention.py` adapts T5 projection/cache handling from
  `src/transformers/models/t5/modeling_t5.py` in that version, retaining its Apache
  notice. Query-block recomputation and lazy relative bias are project modifications;
  they preserve full attention rather than introducing a sliding window.
- MusDr provides the original entropy and onset-distance helpers:
  https://github.com/slSeanWU/MusDr (MIT), pinned commit
  `b034cf86550963cf2021299eafb1001d0fe34853`. It is fetched separately, not vendored.
  Cite Wu and Yang, *The Jazz Transformer on the Front Line*, ISMIR 2020.
- The representation follows REMI-style event sequences; cite Huang and Yang,
  *Pop Music Transformer*, ACM Multimedia 2020. The explicit annotated bar-length
  event representation preserves supplied pulse timing and MIDI event values.
  The separate evaluation view follows the TA tokenizer's 16-position-per-beat grid.
- Development WAVs use TimGM6mb, Tim Brechbill (2004) and David Bolton (2010),
  GPL-2. The font is fetched from the Debian timgm6mb-soundfont source package:
  https://deb.debian.org/debian/pool/main/t/timgm6mb-soundfont/ . Archive SHA-256:
  `af8f3a00e416dfb262bcaa904a1c84df04a51b72bbc1313aed012bc754bdf99b`, checked
  against Debian's `.dsc`; extracted font SHA-256:
  `c5378b62028c920cb11e4803327983fee2f2cdff5dc89c708e39da417e51c854`.
  Debian's copyright notice is retained beside the local font under `.tools/`.
  Font binaries are not bundled in Git and are rendering assets, not training data.

The package, tests and documentation were written with OpenAI Codex assistance.
Only checks and experiments explicitly recorded as executed should be reported as results.
