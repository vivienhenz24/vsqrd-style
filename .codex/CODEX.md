# CODEX Context: Embellie

## Project Goal
Build an extremely small, StyleTTS2-like non-autoregressive TTS model by mimicking the Kokoro approach while keeping the system minimal and practical.

Primary objective:
- Highest quality per parameter count.
- Small enough to train and iterate quickly on cloud GPUs.

## Reference Architecture (Kokoro-Inspired)
Kokoro is treated as a 3-stage non-autoregressive TTS stack:
1. Text/phoneme frontend (`KPipeline`)
2. Acoustic + prosody prediction (`KModel`: ALBERT + duration/prosody nets)
3. Neural vocoder-style waveform synthesis (Decoder + ISTFT-based Generator)

Core reference files in Kokoro:
- `pipeline.py`
- `model.py`
- `modules.py`
- `istftnet.py`

## Detailed Notes From Kokoro

### 1) Frontend and chunking
- `KPipeline` handles language-specific G2P and chunking.
- English path uses `misaki.en.G2P`; other languages may use espeak/language-specific G2P.
- English chunking keeps phoneme chunks <= 510 chars with punctuation-aware splitting.
- Inference entrypoint eventually calls `KPipeline.infer(...)`.

Implication for Embellie:
- Start with strong English-first G2P + chunking.
- Keep chunking deterministic and punctuation-aware.

### 2) Voice conditioning
- Voices are `.pt` tensors downloaded from Hugging Face and cached.
- Multiple voices can be blended by averaging tensors.
- In original inference, one row from the voice pack is selected based on phoneme length (`pack[len(ps)-1]`).

Implication for Embellie:
- Keep voice embedding format simple and cacheable.
- Support blendable style vectors from day 1.

### 3) Phonemes -> token IDs
- `KModel.forward` maps each phoneme symbol via vocab to integer IDs.
- BOS/EOS are added; max context is enforced.
- Phoneme strings are treated as discrete token sequences before timing/acoustic prediction.

Implication for Embellie:
- Stable phoneme vocabulary and tokenization is core infrastructure.
- Do not postpone vocab design.

### 4) Duration and alignment (non-AR timing)
In `forward_with_tokens`:
- ALBERT encoder builds contextual token features.
- Projection to hidden dimension (`bert_encoder`).
- Duration-side style conditioning uses half of style vector (`ref_s[:, 128:]`).
- `ProsodyPredictor.text_encoder` + BiLSTM + duration head predicts per-token durations.
- Durations are squashed, speed-scaled, rounded, and min-clamped.
- Hard alignment matrix is built by repeating token indices (`repeat_interleave`).

Key trick:
- Predict durations first, then expand token features into frame-level features.

Implication for Embellie:
- Duration predictor is a first-class model component, not an afterthought.

### 5) Prosody prediction (F0 and noise)
Using aligned features, Kokoro predicts:
- `F0_pred` (pitch contour)
- `N_pred` (noise/aperiodicity-like control)

Implemented by `ProsodyPredictor.F0Ntrain` with:
- Duration encoder with alternating LSTM + adaptive layer norm conditioned on style.
- Separate AdaIN residual stacks for F0 and N branches.

Implication for Embellie:
- Keep separate controllable channels for periodic (pitch) and aperiodic/noise structure.

### 6) Text encoder path for content features
A separate `TextEncoder` (embedding + conv blocks + BiLSTM) generates linguistic content features.
These are aligned to frame space with the same duration alignment matrix.

Frame-level streams:
- Aligned text content (`asr`)
- `F0`
- `N`

Implication for Embellie:
- Preserve separation of content and prosody paths.

### 7) Decoder + ISTFT-based generator
`Decoder` combines `asr`, `F0`, `N`, and style conditioning using AdaIN residual blocks and upsampling.

`Generator` (vocoder core):
- Builds harmonic source excitation from predicted `F0` (`SineGen` / `SourceModuleHnNSF`).
- Injects source-derived features across upsampling stages.
- Predicts magnitude/phase-like outputs.
- Performs inverse STFT to waveform (`TorchSTFT` or ONNX-friendly `CustomSTFT`).

Implication for Embellie:
- Final stage should be source-filter + ISTFT-like neural waveform synthesis, not only mel->HiFiGAN.

### 8) Timing metadata
- Predicted duration is also used to approximate per-token timestamps for alignment/word timing output.

Implication for Embellie:
- Expose timing metadata as a product feature, not just internal debug info.

## Embellie Build Priorities
1. Reliable English G2P and phoneme normalization.
2. Deterministic chunking and max-context handling.
3. Phoneme vocab/tokenizer with BOS/EOS and OOV policy.
4. Minimal non-AR duration model + hard alignment expansion.
5. Lightweight prosody heads (`F0`, `N`).
6. Compact decoder + ISTFT-based generator.
7. Voice/style embedding format, caching, and simple blending.

## Constraints and Direction
- Prefer smallest model that still sounds natural.
- Keep components modular to swap encoders/decoders later.
- Optimize for reproducible cloud training and fast iteration loops.
- Start English-first, then generalize multilingual.
