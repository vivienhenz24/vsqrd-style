#!/usr/bin/env bash
set -euo pipefail

# One-command cloud setup + train for Kokoro finetune on LJSpeech.
# Usage:
#   bash scripts/train_lj_cloud.sh
#
# Optional env overrides:
#   EPOCHS=8 STAGE1_EPOCHS=3 TRAIN_DEVICE=cuda MAX_ROWS=2000 bash scripts/train_lj_cloud.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}"

LJ_URL="${LJ_URL:-https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2}"
LJ_ARCHIVE="${LJ_ARCHIVE:-${ROOT_DIR}/data/LJSpeech-1.1.tar.bz2}"
LJ_DIR="${LJ_DIR:-${ROOT_DIR}/data/LJSpeech-1.1}"
LJ_WAV_DIR="${LJ_WAV_DIR:-${LJ_DIR}/wavs}"
LJ_METADATA="${LJ_METADATA:-${LJ_DIR}/metadata.csv}"

MANIFEST_PATH="${MANIFEST_PATH:-${ROOT_DIR}/data/lj_manifest.tsv}"
BOOTSTRAP_VOICEPACK="${BOOTSTRAP_VOICEPACK:-${ROOT_DIR}/voices/lj_bootstrap.pt}"
OUT_MODEL="${OUT_MODEL:-${ROOT_DIR}/weights/kokoro-lj-ft.pth}"
FINAL_VOICEPACK="${FINAL_VOICEPACK:-${ROOT_DIR}/voices/lj_finetuned.pt}"

EPOCHS="${EPOCHS:-12}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-4}"
VAL_RATIO="${VAL_RATIO:-0.02}"
SEED="${SEED:-42}"
LR="${LR:-2e-5}"
BERT_LR="${BERT_LR:-5e-6}"
STAGE2_LR="${STAGE2_LR:-1e-5}"
STAGE2_BERT_LR="${STAGE2_BERT_LR:-2e-6}"

# Set MAX_ROWS to limit training set size for faster first run (e.g., MAX_ROWS=2000)
MAX_ROWS="${MAX_ROWS:-0}"
MAX_CLIP_SEC="${MAX_CLIP_SEC:-12.0}"

install_system_deps() {
  if command -v apt-get >/dev/null 2>&1; then
    local SUDO=""
    if command -v sudo >/dev/null 2>&1; then
      SUDO="sudo"
    fi
    $SUDO apt-get update -y
    $SUDO apt-get install -y espeak-ng ffmpeg libsndfile1 curl bzip2
  fi
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --upgrade pip
    python3 -m pip install uv
  fi
}

install_python_env() {
  uv sync
}

repair_misaki_if_needed() {
  # In some cloud builds, local-source misaki can miss packaged data modules.
  if ! .venv/bin/python - <<'PY'
import importlib
import sys
try:
    import misaki
    importlib.import_module("misaki.data")
    importlib.import_module("misaki.en")
except Exception as e:
    print(f"[misaki-check] broken: {e}")
    sys.exit(1)
print("[misaki-check] ok")
PY
  then
    echo "Reinstalling misaki[en] from PyPI to fix missing data module..."
    .venv/bin/python -m pip install --upgrade --force-reinstall "misaki[en]"
    .venv/bin/python - <<'PY'
import importlib
importlib.import_module("misaki.data")
importlib.import_module("misaki.en")
print("[misaki-check] repaired")
PY
  fi
}

resolve_train_device() {
  if [[ -n "${TRAIN_DEVICE:-}" ]]; then
    echo "$TRAIN_DEVICE"
    return 0
  fi
  .venv/bin/python - <<'PY'
import torch
if torch.cuda.is_available():
    print('cuda')
elif torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
PY
}

download_ljspeech() {
  mkdir -p "${ROOT_DIR}/data"
  if [[ ! -f "$LJ_ARCHIVE" ]]; then
    echo "Downloading LJSpeech archive..."
    curl -L "$LJ_URL" -o "$LJ_ARCHIVE"
  fi

  if [[ ! -d "$LJ_DIR" ]]; then
    echo "Extracting LJSpeech..."
    tar -xjf "$LJ_ARCHIVE" -C "${ROOT_DIR}/data"
  fi
}

download_kokoro_weights() {
  mkdir -p "${ROOT_DIR}/weights"
  .venv/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

weights_dir = Path('weights')
weights_dir.mkdir(parents=True, exist_ok=True)

for filename in ['kokoro-v1_0.pth', 'config.json']:
    p = hf_hub_download(repo_id='hexgrad/Kokoro-82M', filename=filename)
    dst = weights_dir / filename
    if not dst.exists():
        dst.write_bytes(Path(p).read_bytes())
        print('saved', dst)
    else:
        print('exists', dst)
PY
}

build_manifest_and_verify() {
  .venv/bin/python - <<PY
import csv
import random
from pathlib import Path
import soundfile as sf

metadata = Path(r"${LJ_METADATA}")
wav_dir = Path(r"${LJ_WAV_DIR}")
out_path = Path(r"${MANIFEST_PATH}")
max_rows = int(r"${MAX_ROWS}")
max_clip_sec = float(r"${MAX_CLIP_SEC}")
seed = int(r"${SEED}")

if not metadata.exists():
    raise SystemExit(f"metadata missing: {metadata}")
if not wav_dir.exists():
    raise SystemExit(f"wav dir missing: {wav_dir}")

rows = []
with metadata.open('r', encoding='utf-8') as f:
    for line in f:
        parts = line.rstrip('\n').split('|')
        if len(parts) < 2:
            continue
        utt_id = parts[0].strip()
        text = (parts[2] if len(parts) >= 3 and parts[2].strip() else parts[1]).strip()
        wav = wav_dir / f"{utt_id}.wav"
        if not wav.exists():
            continue
        info = sf.info(str(wav))
        dur = float(info.frames) / float(info.samplerate)
        if dur <= 0.2:
            continue
        if dur > max_clip_sec:
            continue
        rows.append({
            'filename': wav.name,
            'start_sec': 0.0,
            'end_sec': dur,
            'text': text,
        })

if not rows:
    raise SystemExit('No valid rows found after pairing metadata + wav files')

if max_rows > 0 and len(rows) > max_rows:
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    rows = rows[:max_rows]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open('w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['filename', 'start_sec', 'end_sec', 'text'])
    for r in rows:
        w.writerow([r['filename'], f"{r['start_sec']:.3f}", f"{r['end_sec']:.3f}", r['text']])

print(f"manifest={out_path}")
print(f"rows={len(rows)}")
print(f"example={rows[0]['filename']} duration={rows[0]['end_sec']:.2f}s")
print(f"max_clip_sec={max_clip_sec}")
PY
}

bootstrap_voicepack() {
  local device="$1"
  mkdir -p "${ROOT_DIR}/voices"
  if [[ -f "$BOOTSTRAP_VOICEPACK" ]]; then
    echo "Bootstrap voicepack exists: $BOOTSTRAP_VOICEPACK"
    return 0
  fi

  .venv/bin/python - <<PY
import csv
from pathlib import Path

manifest = Path(r"${MANIFEST_PATH}")
out = Path(r"${BOOTSTRAP_VOICEPACK}")
rows = []
with manifest.open('r', encoding='utf-8') as f:
    rd = csv.DictReader(f, delimiter='\t')
    for r in rd:
        rows.append(r)

if not rows:
    raise SystemExit('Manifest is empty')

# pick one medium/long utterance for stable inversion bootstrap
rows.sort(key=lambda r: float(r['end_sec']) - float(r['start_sec']), reverse=True)
chosen = rows[min(10, len(rows)-1)]
print(chosen['filename'])
PY
  local chosen_file
  chosen_file="$(.venv/bin/python - <<PY
import csv
from pathlib import Path
manifest = Path(r"${MANIFEST_PATH}")
rows=[]
with manifest.open('r', encoding='utf-8') as f:
    rd=csv.DictReader(f, delimiter='\t')
    rows=[r for r in rd]
rows.sort(key=lambda r: float(r['end_sec']) - float(r['start_sec']), reverse=True)
print(rows[min(10, len(rows)-1)]['filename'])
PY
)"

  echo "Building bootstrap voicepack from ${chosen_file}"
  .venv/bin/python -m mifi.style_inversion \
    --audio "${LJ_WAV_DIR}/${chosen_file}" \
    --output "$BOOTSTRAP_VOICEPACK" \
    --model "${ROOT_DIR}/weights/kokoro-v1_0.pth" \
    --config "${ROOT_DIR}/weights/config.json" \
    --steps 120 \
    --lr 0.02 \
    --device "$device"
}

run_training() {
  local device="$1"
  .venv/bin/python -m mifi.train_kokoro_finetune \
    --manifest "$MANIFEST_PATH" \
    --segments-dir "$LJ_WAV_DIR" \
    --voicepack "$BOOTSTRAP_VOICEPACK" \
    --out-model "$OUT_MODEL" \
    --model "${ROOT_DIR}/weights/kokoro-v1_0.pth" \
    --config "${ROOT_DIR}/weights/config.json" \
    --epochs "$EPOCHS" \
    --stage1-epochs "$STAGE1_EPOCHS" \
    --lr "$LR" \
    --bert-lr "$BERT_LR" \
    --stage2-lr "$STAGE2_LR" \
    --stage2-bert-lr "$STAGE2_BERT_LR" \
    --stage2-style-mode fixed \
    --val-ratio "$VAL_RATIO" \
    --seed "$SEED" \
    --sample-every 1 \
    --final-voicepack "$FINAL_VOICEPACK" \
    --final-voicepack-mode invert_longest \
    --final-invert-steps 120 \
    --final-invert-lr 0.02 \
    --final-invert-device "$device" \
    --device "$device"
}

echo "[1/7] Install system dependencies"
install_system_deps

echo "[2/7] Ensure uv"
ensure_uv

echo "[3/7] Install python env"
install_python_env
repair_misaki_if_needed

echo "[4/7] Download LJSpeech"
download_ljspeech

echo "[5/7] Download Kokoro weights/config"
download_kokoro_weights

echo "[6/7] Build + verify manifest"
build_manifest_and_verify

DEVICE="$(resolve_train_device)"
echo "Resolved train device: ${DEVICE}"

echo "[7/7] Bootstrap voicepack + train"
bootstrap_voicepack "$DEVICE"
run_training "$DEVICE"

echo "Training complete."
echo "Model: ${OUT_MODEL}"
echo "Voicepack: ${FINAL_VOICEPACK}"
