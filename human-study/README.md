# Blind human preference study

Executable protocol for the remaining human-preference evidence. It does not contain fabricated responses.

## Design

- 40 project-07 pairs: base vs calibrated R3 seed 0, with matched portrait/audio/generation seed.
- 40 project-08 pairs: base vs development-selected V4 step 1000 at scale 0.05, with matched first frame/prompt/generation seed.
- Candidate appears exactly 20 times as A and 20 times as B within each project.
- Trial IDs, trial order and A/B placement use a newly generated private random seed and salt.
- `study.json` and anonymous `media/` may be distributed.
- `private_key.json` must remain private until responses are frozen.

A participant rating all 80 pairs supplies 80 responses, not 80 independent participants. Recruit multiple participants and report participant-cluster uncertainty.

## Build and run

```bash
python tools/build_blind_study.py
cd human-study
python -m http.server 8000
```

Open `http://localhost:8000`, complete the study, then download `responses.csv`.

## Aggregate real responses

Combine participant files with one CSV header, then run:

```bash
python tools/score_blind_study.py \
  human-study/responses.csv \
  --key human-study/private_key.json \
  --out human-study/results.json
```

The scorer reports response-level Wilson intervals and participant-cluster bootstrap intervals. Do not publish a preference claim before real participant responses exist.
