# Data Sources — 07-talking-head-motion-opsd

## Public dataset used this round

| Field | Value |
|---|---|
| Name | AISHELL-1 |
| URL (download) | `https://huggingface.co/datasets/AISHELL/AISHELL-1` |
| URL (canonical) | `https://www.openslr.org/33/` |
| License | Apache 2.0 |
| Download date | 2026-10-30 |
| Downloaded subset | 75 speaker archives (`data_aishell/wav/S0002.tar.gz` … `S0076.tar.gz`) + transcript |
| Speakers | 75 total; 60 used for training, 15 for evaluation |
| Segments built | 2000 train, 100 eval |
| Segment length | 8.0–15.0 s (truncated at 15 s), 16 kHz mono PCM_16 |

### Why this replaces the previous data
The previous round used 25 macOS `say` TTS clips. Those are synthetic, single-style,
and are NOT a public dataset. This round uses real multi-speaker human speech.

### Speaker split (no overlap)
Speakers are ordered by numeric id. The first 60 (`S0002`–`S0061`) form the training
pool; the next 15 (`S0062`–`S0076`) form the evaluation pool. Verified disjoint:
the split assertion in `scripts/build_audio_segments.py` fails on any overlap.

### Segment construction
AISHELL-1 utterances average ~5 s, but the motion trainer consumes 4 s windows with
a 1 s stride. Utterances of the SAME speaker are concatenated in numeric-id order
until the running length reaches 8 s; a 0.3 s silence gap is inserted at each join.
Concatenation never crosses speakers. Segments longer than 15 s are truncated.

### Files
- Speaker id list / manifest: `data/audio_manifest_large.json` (in `main`)
- Build script: `scripts/build_audio_segments.py` (in `main`)
- Audio wavs: `data/audio_public/train/`, `data/audio_public/eval/` — **not** in `main`
  (backed up to a `backup/*` branch).

## Public portrait / talking-head datasets (qualitative gallery)

The original 3 portraits in `data/portraits/` are few and give no identity
diversity. To make the demos show real people, two more public sets were added.

### CelebA-HQ — still portraits

| Field | Value |
|---|---|
| Source | `mattymchen/celeba-hq` on Hugging Face (derivative of CelebA) |
| License | CelebA terms: research / non-commercial use |
| Downloaded | 1 of 6 parquet shards (454 MB) |
| Extracted | 12 portraits at 1024×1024 → `data/portraits_public/celebahq_*.png` |

### CelebV-HQ — real talking-head video

| Field | Value |
|---|---|
| Source | `saeed-5959/celebv_hq_head_talking` on Hugging Face |
| Downloaded | 12 of 23 191 clips |
| What is taken | first frame → portrait; audio track → 16 kHz mono wav |
| Outputs | `data/portraits_public/celebv_*.png`, `data/audio_public/celebv/*.wav` |

Using CelebV-HQ gives demos driven by **real human speech** (not only TTS) on
**real faces**, which is what makes blink / nod / listening behaviour visible.

### License note
Both sets cover celebrity imagery and are used here only to demonstrate the
method. They are not redistributed into `main`; generated videos go to a
`backup/*` branch.

## Legacy evaluation set (retained)
The original 20 macOS TTS clips in `data/audio/eval/` are retained as `legacy eval`
so results stay comparable with the previous round.
