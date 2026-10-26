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
- MusDr provides the original entropy and onset-distance helpers:
  https://github.com/slSeanWU/MusDr (MIT), pinned commit
  `b034cf86550963cf2021299eafb1001d0fe34853`. It is fetched separately, not vendored.
  Cite Wu and Yang, *The Jazz Transformer on the Front Line*, ISMIR 2020.
- The representation follows REMI-style event sequences; cite Huang and Yang,
  *Pop Music Transformer*, ACM Multimedia 2020. The explicit annotated bar-length
  tokens and tempo-normalized codec are this project's implementation choices.
- No SoundFont has been chosen or bundled. Any future SoundFont must retain its
  own attribution/license and is used for audio rendering only.

The package, tests and documentation were written with OpenAI Codex assistance.
Only checks and experiments explicitly recorded as executed should be reported as results.
