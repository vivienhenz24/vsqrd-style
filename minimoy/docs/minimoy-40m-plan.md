# minimoy-40m: Research Findings & Implementation Plan

## Context

- **Dataset**: 24h single-speaker high-quality audio
- **Goal**: Train from scratch, real-time CPU inference, quality-preserving compression
- **Base**: StyleTTS2 codebase (minimoy/)

---

## 1. Research Findings

### 1.1 FLY-TTS (the key paper)

FLY-TTS was evaluated on LJSpeech — which is **exactly 24h single-speaker audio**. Results:

| Model | Params | MOS | Notes |
|---|---|---|---|
| StyleTTS2 | ~190M | 4.28 | Full model |
| FLY-TTS | 17.89M | 4.12 | ConvNeXt decoder |
| Mini FLY-TTS | 10.92M | 4.05 | Smaller everything |

**Key techniques:**
1. ConvNeXt + iSTFT decoder replaces HiFiGAN transposed-conv chain
2. Grouped parameter sharing in text encoder
3. WavLM discriminator (training only)

### 1.2 Vocos (the vocoder paper)

Vocos proves that iSTFT-based synthesis matches BigVGAN quality with dramatically less compute.

**CPU real-time factor (xRT) — higher is better:**

| Model | GPU | CPU |
|---|---|---|
| HiFiGAN | 495× | 5.8× |
| BigVGAN | 98× | **0.4×** ← slower than real-time |
| iSTFTNet | 1045× | 14.4× |
| **Vocos** | **6696×** | **169×** |

The speedup on CPU is enormous. iSTFT (FFT-based) is highly optimized on CPU. Transposed convolutions have poor SIMD utilization.

**Vocos architecture:**
- Input (mel spectrogram) at fixed temporal resolution (mel frame rate, no upsampling)
- `embed`: Conv1d(n_mels, dim, k=7)
- `LayerNorm`
- N× `ConvNeXtBlock`: depthwise Conv1d(k=7) → LayerNorm → Linear(dim→4×dim) → GELU → Linear(4×dim→dim) → gamma scale → residual
- `final LayerNorm`
- `ISTFTHead`: Linear(dim, n_fft+2) → split into magnitude (exp) + phase (cos/sin) → iSTFT

**Proven config:** dim=512, intermediate_dim=1536, num_layers=8, n_fft=1024, hop=256 → **13.5M params, 169× real-time CPU**

**Key insight:** No upsampling in the generator. The iSTFT itself performs the "upsampling" from mel frame rate to audio sample rate, and it's alias-free by construction.

### 1.3 StyleTTS2 Ablations (from the paper)

Which components actually matter for quality:

| Removed | CMOS drop |
|---|---|
| Diffusion | -0.46 |
| SLM (WavLM) discriminator | -0.32 |
| Prosodic style encoder | -0.35 |
| Acoustic style encoder | -0.41 |

Diffusion is the biggest single drop, but for **single-speaker** TTS it's largely irrelevant:
- Diffusion generates diverse styles from text alone (no reference audio)
- For single-speaker deployment you always use the reference audio style
- The style vector `s` is computed directly from the reference — no diffusion needed

---

## 2. Current minimoy Architecture (248M total)

### 2.1 Inference model (~192M)

| Component | Params | Config driver |
|---|---|---|
| PLBERT | ~67M | 12L, h=768, ff=2048 |
| 2× StyleEncoder | ~28M | max_conv_dim=512 |
| Diffusion (Transformer1d) | ~22M | num_layers=3, heads=8, head_features=64 |
| Decoder (front-end + Generator) | ~53M | dec_inner=1024 (hardcoded), upsample_initial=512 |
| ProsodyPredictor | ~16M | hidden_dim=512 |
| TextEncoder | ~5.6M | hidden_dim=512, n_layer=3 |
| bert_encoder (Linear) | ~0.4M | hidden_size→hidden_dim |

### 2.2 Training-only (~56M)

| Component | Params |
|---|---|
| MPD | ~41M |
| ASR (text aligner) | ~7.9M |
| JDC (F0 extractor) | ~5.2M |
| MSD | ~0.3M |
| WavLM discriminator | ~1.2M |

### 2.3 Known bugs in current code

1. **Device hardcode**: `Decoder.forward` uses `torch.ones(...).to('cuda')` — breaks on CPU/MPS
2. **Dimension hardcode**: `asr_res = Conv1d(512, 64)` — 512 is hardcoded, should be `dim_in`. Will crash if `hidden_dim ≠ 512`
3. **Generator hardcode**: `1024` channels in Decoder's AdainResBlk1d blocks is hardcoded in the constructor — not driven by config

---

## 3. The Generator Replacement

### 3.1 Current Generator (iSTFTNet)

Takes features at mel frame rate → 60× transposed-conv upsampling chain → small iSTFT post-processing:

```
features (mel rate, 512-ch)
→ ConvTranspose1d ×10 (512→256)
→ AdaINResBlock1 ×3 (dil=1,3,5)
→ ConvTranspose1d ×6 (256→128)
→ AdaINResBlock1 ×3 (dil=1,3,5)
→ [+ NSF harmonic source from SineGen(F0)]
→ conv_post → n_fft+2 channels
→ iSTFT(n_fft=20, hop=5) at audio rate
```

~20M params. Slow on CPU (transposed conv). Complex (NSF source signal).

### 3.2 Proposed Generator (Vocos-style)

Takes features at mel frame rate → ConvNeXt blocks (no upsampling) → iSTFT at mel frame rate:

```
features (mel rate, dec_inner//2 channels)
→ Conv1d embed (→ vocos_dim)
→ LayerNorm
→ N× ConvNeXtBlockAdaIN (depthwise + AdaIN + inverted bottleneck)
→ final LayerNorm
→ Linear(vocos_dim, n_fft+2)
→ split: mag=exp(), phase=cos/sin
→ iSTFT(n_fft=1024, hop=mel_hop) → audio
```

~4-5M params. Fast on CPU (iSTFT + depthwise conv). No NSF source signal needed.

**The key:** iSTFT hop_length = mel spectrogram hop_length. One ConvNeXt frame → exactly one mel frame → `hop_length` audio samples. Natural, alias-free upsampling.

**AdaIN vs LayerNorm:** Vocos uses LayerNorm (unconditional). We replace it with `AdaIN1d` (already in the codebase) for continuous style conditioning via the style vector `s`. This is the only meaningful difference from stock Vocos.

### 3.3 F0 excitation (NSF/SineGen) — keep or drop?

The current Generator injects a harmonic sine source derived from F0 into each upsampling stage. This helps with pitch accuracy.

Arguments for **dropping** it:
- Vocos proves it's not needed — learns pitch from features directly
- The Decoder front-end already passes F0 explicitly to the encode/decode blocks
- Simpler code, faster inference
- Single speaker with consistent pitch range

Arguments for **keeping** it:
- StyleTTS2 ablation shows F0 conditioning helps prosody accuracy
- The Decoder front-end F0 conditioning is at half mel-rate (strided) — Generator currently gets full-rate F0

**My recommendation: drop it.** The F0 information is already in the features that enter the Generator. Vocos at this scale proves it's sufficient.

---

## 4. Proposed 40M Architecture

### 4.1 Config changes

| Parameter | Current | Proposed | Notes |
|---|---|---|---|
| `hidden_dim` | 512 | 256 | TextEncoder, Predictor |
| `style_dim` | 128 | 64 | Style vectors, all AdaIN layers |
| `dim_in` | 64 | 32 | StyleEncoder input channels |
| `max_conv_dim` | 512 | 256 | StyleEncoder max channels |
| `hop_length` | 300 | 256 or 300 | See Section 5 |
| `dec_inner_dim` | 1024 (hardcoded) | 256 | Decoder AdainResBlk1d channels |
| `vocos_dim` | — | 256 | ConvNeXt hidden dim |
| `vocos_intermediate_dim` | — | 768 | ConvNeXt bottleneck (3× dim) |
| `vocos_num_layers` | — | 8 | Number of ConvNeXt blocks |
| `gen_istft_n_fft` | 20 | 1024 | iSTFT FFT size |
| `gen_istft_hop_size` | 5 | 256 or 300 | Must match mel hop_length |
| `lambda_diff` | 1.0 | 0.0 | Disable diffusion training |

### 4.2 PLBERT config

| Parameter | Current | Proposed |
|---|---|---|
| `num_hidden_layers` | 12 | 3 |
| `hidden_size` | 768 | 256 |
| `num_attention_heads` | 12 | 4 |
| `intermediate_size` | 2048 | 1024 |

Training from scratch — no pretrained weights to preserve.

### 4.3 Estimated parameter counts

| Component | Proposed params |
|---|---|
| PLBERT (3L/h=256) | ~3M |
| bert_encoder (Linear 256→256) | ~0.07M |
| TextEncoder (hidden=256, n_layer=3) | ~2M |
| 2× StyleEncoder (dim_in=32, max=256, style=64) | ~3M |
| ProsodyPredictor (style=64, hidden=256) | ~3M |
| Decoder front-end (dec_inner=256) | ~2M |
| Vocos Generator (dim=256, 8L) | ~4M |
| **Total inference** | **~17M** |

Training-only (discriminators, ASR, JDC) stay the same size — they don't affect inference speed.

---

## 5. Open Decision: hop_length

This is the one decision that affects everything globally.

**Option A: keep hop=300**
- No preprocessing changes
- Mel spectrograms stay the same
- Generator uses `gen_istft_hop_size=300`, `gen_istft_n_fft=1024` (ratio = 3.4)
- Slightly non-standard ratio

**Option B: change to hop=256**
- Change `preprocess_params.spect_params.hop_length: 300 → 256`
- Change `win_length: 1200 → 1024`
- Change `gen_istft_hop_size: 5 → 256`, `gen_istft_n_fft: 1024`
- Matches the **exact Vocos proven config** (n_fft=1024, hop=256, ratio=4)
- Standard across most modern TTS systems (Vocos, VITS, etc.)
- Slightly more mel frames per second (93.75 vs 80) — marginally more compute
- Requires re-processing training data if you've already extracted mels

Both work. Vocos was proven with hop=256. If you haven't preprocessed data yet, go with hop=256.

---

## 6. Implementation Plan

### Files to change (with code)

**`Modules/istftnet.py`**
- Remove: `SineGen`, `SourceModuleHnNSF`, `padDiff`, `AdaINResBlock1` (~200 lines)
- Add: `ConvNeXtBlockAdaIN` class (~55 lines)
- Replace: `Generator` class entirely (~350 lines → ~90 lines)
- Update: `Decoder.__init__` — add `dec_inner_dim`, `vocos_*` params, fix `asr_res` dim, accept `**kwargs` for old params
- Update: `Decoder.forward` — fix device bug (`to('cuda')` → `to(device)`)

**`models.py`**
- Update Decoder instantiation to pass new params (use `getattr` for backward compat)
- ~4 lines changed

### Files to change (numbers only)

**`Configs/config.yml`**
- Model dimensions, Vocos decoder params, `lambda_diff: 0`

**`Utils/PLBERT/config.yml`**
- 3L/h=256 config

### Files untouched

- `models.py` build_model logic (beyond the 4-line decoder call)
- All discriminators
- Training loop scripts
- Data loading
- ASR / JDC auxiliary models
- `Modules/hifigan.py`

---

## 7. Things I'm Uncertain About

1. **PLBERT train-from-scratch stability**: StyleTTS2 was designed for a pretrained PLBERT. Training it end-to-end from scratch might cause instability early in training (PLBERT and acoustic model learning simultaneously). Might need a warmup strategy.

2. **dec_inner_dim=256 vs 384**: I estimated 256 is enough for single-speaker but haven't seen an ablation at exactly this scale with AdaIN blocks. 384 is safer; 256 is more aggressive.

3. **No NSF source signal**: I'm confident based on Vocos results, but the Vocos paper tested vocoding (mel→audio), not full TTS. For TTS, the Generator gets text-derived features (not GT mel), which may benefit more from explicit F0 guidance. Low risk to drop it, but worth noting.

4. **hop_length change impact**: Changing hop=300→256 slightly changes the acoustic resolution of the mel spectrogram. For training from scratch this is fine, but worth deciding before preprocessing data.
