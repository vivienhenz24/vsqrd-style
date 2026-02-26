# minimoy-40m: Research Findings & Code-Validated Implementation Plan

## Context

- **Dataset**: 24h single-speaker high-quality audio
- **Goal**: train from scratch, real-time CPU inference, quality-preserving compression
- **Base**: StyleTTS2-derived codebase (`minimoy/`)

---

## 1. Research Findings (unchanged direction)

### 1.1 FLY-TTS relevance

FLY-TTS is directly relevant because it was evaluated on LJSpeech (24h single-speaker).

| Model | Params | MOS | Notes |
|---|---|---|---|
| StyleTTS2 | ~190M | 4.28 | Full model |
| FLY-TTS | 17.89M | 4.12 | ConvNeXt decoder |
| Mini FLY-TTS | 10.92M | 4.05 | Smaller full stack |

Key ideas to reuse:
1. Replace heavy transposed-conv decoder with ConvNeXt+iSTFT path
2. Aggressively reduce backbone dimensions
3. Keep heavy adversarial critics in training only

### 1.2 Vocos relevance

Vocos demonstrates large CPU speedups from fixed-rate ConvNeXt features + iSTFT synthesis.

| Model | GPU xRT | CPU xRT |
|---|---|---|
| HiFiGAN | 495 | 5.8 |
| BigVGAN | 98 | 0.4 |
| iSTFTNet | 1045 | 14.4 |
| Vocos | 6696 | 169 |

Core architectural implication:
- No transposed-conv upsampling chain in generator
- Predict iSTFT bins at mel frame rate
- Set generator iSTFT hop equal to mel hop

### 1.3 StyleTTS2 ablations

Diffusion helps quality most in paper ablations, but for single-speaker reference-conditioned deployment we can likely down-prioritize or remove it.

---

## 2. Current Codebase Reality (validated)

### 2.1 Confirmed hardcoded coupling

1. Decoder bugs in `Modules/istftnet.py`:
- device hardcode (`to('cuda')`)
- `asr_res` input hardcoded to `512`
- decoder inner channels hardcoded to `1024`

2. The same decoder device hardcode exists in `Modules/hifigan.py`.

3. `style_dim=128` is assumed in runtime sampling/training utilities:
- hard split `[:, :128]` and `[:, 128:]`
- hard noise shape `(1, 256)` in second-stage sampling
- same 128 split in `Modules/slmadv.py`

4. `hop_length=300` is hardcoded outside config:
- `meldataset.py` constants and mel transform init
- wave crop indexing in `train_first.py`, `train_second.py`, `train_finetune.py`, `train_finetune_accelerate.py`, and `Modules/slmadv.py`

5. `max_conv_dim` in config is currently ignored:
- `models.py` passes `max_conv_dim=args.hidden_dim` into both style encoders.

6. `lambda_diff=0` alone does not disable diffusion compute:
- diffusion forward/sampler still run before loss aggregation.

7. PLBERT loader always expects a checkpoint file `step_*.t7`:
- changing `Utils/PLBERT/config.yml` alone does not produce random-init training; pretraining (or loader change) is required.

### 2.2 Scope implication

The original plan was architecturally right, but underestimated integration work. This is not a `2-file` change; it is a coordinated refactor across decoder, training scripts, dataset, and config plumbing.

---

## 3. Target Architecture (still recommended)

### 3.1 Generator

Replace current iSTFTNet generator (upsampling + NSF source) with Vocos-style generator:
- Conv1d embed
- N ConvNeXt-like 1D residual blocks with AdaIN conditioning
- linear projection to `(n_fft + 2)` bins
- magnitude/phase reconstruction
- iSTFT with `hop == mel_hop`

Drop NSF/Sine source in first pass.

### 3.2 Size targets

Primary downsizing targets:
- `hidden_dim: 512 -> 256`
- `style_dim: 128 -> 64`
- `dim_in: 64 -> 32`
- decoder inner channels config-driven (`dec_inner_dim`)
- Vocos path (`vocos_dim=256`, `vocos_intermediate_dim=768`, `vocos_num_layers=8`)

### 3.3 Hop strategy

Preferred if data is not yet frozen: `hop=256`, `win=1024`, `gen_istft_n_fft=1024`, `gen_istft_hop_size=256`.

If existing data/experiments require compatibility, keep `hop=300` temporarily and refactor constants first, then migrate.

---

## 4. Corrected Config Plan

### 4.1 `Configs/config.yml` (and parity with `config_ft.yml`, `config_libritts.yml`)

| Parameter | Current | Proposed | Notes |
|---|---|---|---|
| `hidden_dim` | 512 | 256 | Text/Prosody backbone |
| `style_dim` | 128 | 64 | Must propagate everywhere |
| `dim_in` | 64 | 32 | Style encoders |
| `max_conv_dim` | 512 | 256 | Must fix `models.py` wiring to take effect |
| `hop_length` | 300 | 256 (or 300) | Must propagate to dataset + wave crop ops |
| `dec_inner_dim` | hardcoded 1024 | 256 or 384 | safer to start at 384 if unstable |
| `vocos_dim` | N/A | 256 | new |
| `vocos_intermediate_dim` | N/A | 768 | new |
| `vocos_num_layers` | N/A | 8 | new |
| `gen_istft_n_fft` | 20 | 1024 | new generator path |
| `gen_istft_hop_size` | 5 | match mel hop | strict |
| `lambda_diff` | 1.0 | 0.0 (optional) | needs code guard to skip compute |

### 4.2 PLBERT decision

Two valid tracks:
1. **Stable track**: keep current pretrained PLBERT while shrinking acoustic stack first.
2. **True from-scratch track**: pretrain a smaller PLBERT externally (or modify loader for random init fallback), then train TTS.

---

## 5. Revised Implementation Plan (code-first order)

### Phase 0: unblock parameterization and portability

1. Fix decoder hardcodes in both `Modules/istftnet.py` and `Modules/hifigan.py`:
- `to('cuda') -> to(tensor.device)`
- `Conv1d(512, 64, ...) -> Conv1d(dim_in, 64, ...)`
- replace fixed `1024` decode channels with `dec_inner_dim`

2. Fix style-encoder wiring in `models.py`:
- pass `max_conv_dim=args.max_conv_dim` (not `args.hidden_dim`)

### Phase 1: remove hidden global constants

1. Make mel params in `meldataset.py` come from config, not hardcoded 300/1200.
2. Replace literal `* 300` waveform indexing in:
- `train_first.py`
- `train_second.py`
- `train_finetune.py`
- `train_finetune_accelerate.py`
- `Modules/slmadv.py`
with config-driven hop.
3. Replace fixed style split constants (`128`, `256`) with `style_dim` and `2*style_dim` in:
- `train_second.py`
- `Modules/slmadv.py`

### Phase 2: swap generator

1. Refactor `Modules/istftnet.py`:
- remove NSF-specific source path classes from active decoder flow
- add Vocos-style AdaIN ConvNeXt block
- replace `Generator` implementation

2. Update `models.py` decoder construction:
- pass new decoder args (`dec_inner_dim`, `vocos_*`)
- keep backward compatibility defaults

### Phase 3: optional diffusion disable path

If operating without diffusion:
1. Keep module for checkpoint compatibility, but skip sampler/forward branches when disabled.
2. Add config gate like `use_diffusion: false` instead of only relying on `lambda_diff=0`.

### Phase 4: config rollout

Update:
- `Configs/config.yml`
- `Configs/config_ft.yml`
- `Configs/config_libritts.yml`
- optionally `Utils/PLBERT/config.yml` only if following from-scratch PLBERT track

---

## 6. Open Risks (updated)

1. **Small `style_dim` migration risk**: every 128/256 assumption must be removed first.
2. **PLBERT cold-start risk**: without PLBERT pretraining, convergence quality may drop hard.
3. **Aggressive decoder shrinking risk**: `dec_inner_dim=256` may be too aggressive; `384` is safer first.
4. **Hop migration risk**: change requires synchronized edits to dataset + train loops + slmadv wave indexing.

---

## 7. Recommendation

Execute in this exact order:
1. Phase 0 + Phase 1 (make system fully config-driven and shape-safe)
2. Phase 2 (Vocos generator replacement)
3. quick sanity training run
4. only then reduce PLBERT or disable diffusion

This keeps changes debuggable and prevents silent shape/runtime failures during the architectural swap.
