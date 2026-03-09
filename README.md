# mifi

Training-loop repo for a TTS model.

## Docker (GPU)

This repo includes a GPU-ready Docker setup for cloud training.

### Prerequisites

- NVIDIA GPU instance
- NVIDIA driver installed on host
- Docker installed
- NVIDIA Container Toolkit installed (`nvidia-smi` should work on host and in containers)

### Build image

```bash
docker compose build
```

### Run training container

```bash
docker compose run --rm trainer
```

### Run a custom command

```bash
docker compose run --rm trainer python -c "import torch; print(torch.cuda.is_available())"
```
q
## Notes

- `docker-compose.yml` mounts the repo into `/workspace` for live code edits.
- `shm_size` is set to `16gb` to avoid dataloader shared-memory issues during training.
- The image pre-downloads NLTK tagger data required by `g2p-en` (`averaged_perceptron_tagger_eng`).
- The Docker image also installs NeMo text normalization (`nemo_text_processing`) via the `tn` extra.
- Replace `command: python -m mifi.main` with your actual training entrypoint when ready.


bash minimoy/scripts/runpod_setup.sh
bash minimoy/scripts/runpod_train.sh first minimoy/Configs/config.yml
bash minimoy/scripts/runpod_train.sh second minimoy/Configs/config.yml

---

## Turkish Kokoro — Research Notes

Goal: train a Kokoro-equivalent multilingual TTS model for Turkish using 40h male + 40h female voice actor data.

### How Kokoro relates to StyleTTS2

Kokoro-82M is the StyleTTS2 LibriTTS checkpoint with training infrastructure stripped. Confirmed by weight comparison:

| Module | Kokoro | StyleTTS2 LibriTTS | Match |
|---|---|---|---|
| `bert` | 6.29M | 6.29M | identical |
| `bert_encoder` | 0.39M | 0.39M | identical |
| `predictor` | 16.19M | 16.19M | identical |
| `text_encoder` | 5.61M | 5.61M | identical |
| `decoder` | 53.28M | 54.29M | ~1M diff |

Dropped at inference: `style_encoder` (13.88M), `predictor_encoder` (13.88M), `diffusion` (50.65M), `text_aligner` (7.87M), `pitch_extractor` (5.25M), discriminators (~43M). Total training checkpoint: ~218M → 82M inference.

### Architecture

StyleTTS2 has two training stages:

- **Stage 1** (`train_first.py`): trains text-mel alignment (TMA) + acoustic encoder + decoder. No PLBERT, no diffusion.
- **Stage 2** (`train_second.py`): adds style diffusion + WavLM adversarial training. Uses PLBERT for duration prediction.

Inference-time modules only:
- `text_encoder` — IPA token embedding + 3×CNN + BiLSTM
- `bert` + `bert_encoder` — PLBERT (ALBERT) for duration/prosody context
- `predictor` — DurationEncoder + F0/N LSTMs
- `decoder` — AdainResBlk encode/decode chain + iSTFTNet Generator with F0 excitation

### The model is IPA-first by design

`TextCleaner` in `text_utils.py` is a plain character→index dict over `_letters_ipa`. There is no G2P inside training — the text column in the data list must already be IPA phonemes. The symbol table covers all standard IPA.

For Turkish: use `misaki.espeak.EspeakG2P('tr')` at data prep time:
```python
from misaki.espeak import EspeakG2P
g2p = EspeakG2P('tr')
ipa, _ = g2p("Merhaba dünya")
# store "filename.wav|<ipa>|speaker_id" in train_list.txt
```

Verify Turkish IPA coverage before training at scale:
```python
from minimoy.text_utils import dicts
unknown = [c for c in ipa if c not in dicts and c != ' ']
# should be empty
```

### PLBERT for Turkish

PLBERT (ALBERT) was pre-trained on English IPA sequences derived from Wikipedia. Its `vocab_size=178` matches the IPA token count — it operates on phoneme token IDs, not raw text. During stage 2 training it is fine-tuned (`model.bert.train()`).

Options in priority order:
1. **`papercup-ai/multilingual-pl-bert`** — supports 14 languages, check if Turkish is included (zero effort if so)
2. **Train Turkish PLBERT from scratch** using `yl4579/PL-BERT` repo: download Turkish Wikipedia → convert to IPA with espeak-ng → train ALBERT MLM on phoneme sequences (~1 day on a single A100, no audio needed)
3. **Use English PLBERT as-is** — fine-tunes during stage 2, sufficient with 80h of data. Note: `lambda_diff: 0` / `use_diffusion: false` in minimoy config disables the diffusion component which is the most PLBERT-dependent part.

### How voicepacks are created

A voicepack is `[510, 1, 256]`. Confirmed by inspection: `pack[n]` is the **cumulative mean** of style vectors from the first `n+1` utterances sorted by phoneme count ascending. Reconstruction error vs cumsum/n is ~1e-10 (floating point).

Mechanism:
1. Sort all utterances from the speaker by phoneme string length ascending
2. For each utterance: run mel through `style_encoder` → 128-dim acoustic style; run through `predictor_encoder` → 128-dim prosodic style; concatenate → 256-dim `ref_s`
3. Compute running cumulative mean: `pack[n] = mean(ref_s_0 ... ref_s_n)`

At inference: `ref_s = pack[len(phoneme_string) - 1]`. The acoustic half (`ref_s[:, :128]`) goes to the decoder; the prosodic half (`ref_s[:, 128:]`) goes to the predictor.

The length-indexing is an artifact of processing utterances in length order — longer synthesis uses a more averaged/stable speaker representation.

### Kokoro's multilingual approach

One shared acoustic model + language-specific G2P → IPA. The model is language-blind: it only sees IPA token IDs. Languages:
- English: `misaki.en.G2P` (custom)
- Japanese/Chinese: `misaki[ja]` / `misaki[zh]`
- Spanish, French, Hindi, Italian, Portuguese: `EspeakG2P(language)` — just espeak-ng

Turkish is not in Kokoro-82M (model was only trained on supported languages). A Turkish model needs to be trained from scratch.

### Training plan for maximum quality

Use `minimoy/Configs/config_ft.yml` as the base — it already has full Kokoro-scale dimensions (`hidden_dim: 512`, `style_dim: 128`, `dec_inner_dim: 1024`, `use_diffusion: true`, `multispeaker: true`).

Key config changes for Turkish from scratch:
```yaml
pretrained_model: ""          # train from scratch (80h is enough)
load_only_params: false
PLBERT_dir: 'Utils/PLBERT_TR/'
data_params:
  train_data: "Data/tr_train.txt"
  val_data: "Data/tr_val.txt"
  root_path: "/workspace/datasets/turkish/wavs"
  OOD_data: "Data/tr_ood.txt"   # Turkish IPA sentences
model_params:
  multispeaker: true
  decoder:
    type: 'istftnet'            # matches Kokoro weight format
```

Training:
```bash
# Stage 1 (~2-3 days on 4×A100)
accelerate launch train_first.py --config_path minimoy/Configs/config_tr.yml

# Stage 2 (~2-3 days)
python train_second.py --config_path minimoy/Configs/config_tr.yml
```

After training, create voicepacks by sorting all utterances by phoneme count, running both style encoders, storing the cumulative mean. Then strip training modules and ship the 82M inference subset exactly as Kokoro does.

### Parameter budget

Full training checkpoint: ~218M params. Inference model: ~82M. To get to a 15M inference model:

| Lever | Savings | Notes |
|---|---|---|
| Drop style encoders at inference (use voicepack) | −27.8M | free, zero quality loss |
| Shrink PLBERT (hidden 768→256, layers 12→4) | −5M | retrain from scratch anyway |
| Shrink decoder (vocos_dim 256→128, layers 8→4) | −4M | risks audio quality |

Current `minimoy/Configs/config.yml` already uses reduced dims (hidden_dim=256, style_dim=64) — that's the lightweight experiment config, not the full-quality one.