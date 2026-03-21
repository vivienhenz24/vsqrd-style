#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
HF_DATASET="vsqrd/styletts2-turkish"
HF_PLBERT="vsqrd/pl-bert-turkish"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MEL_CACHE_WORKERS="${MEL_CACHE_WORKERS:-4}"
MEL_CACHE_CHUNKSIZE="${MEL_CACHE_CHUNKSIZE:-32}"

echo "=== Setting up Turkish StyleTTS2 training ==="

# Require HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    read -rsp "Enter HuggingFace token: " HF_TOKEN
    echo
    export HF_TOKEN
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

cd $REPO_ROOT

# Install dependencies
echo "--- Installing dependencies ---"
uv sync
PY="$REPO_ROOT/.venv/bin/python"

# Download all files via Python API
echo "--- Downloading dataset + alignments + models ---"
"$PY" -c "
import os
from huggingface_hub import hf_hub_download
from tqdm import tqdm

token = os.environ['HF_TOKEN']
repo_root = '$REPO_ROOT'

downloads = [
    ('vsqrd/styletts2-turkish', 'combined_dataset.tar.gz', 'dataset', repo_root, os.path.join(repo_root, 'combined_dataset')),
    ('vsqrd/styletts2-turkish', 'alignments.tar.gz',       'dataset', repo_root, os.path.join(repo_root, 'alignments')),
    ('vsqrd/pl-bert-turkish',   'step_160000.t7',          'model',   os.path.join(repo_root, 'StyleTTS2/Utils/PLBERT_turkish'), None),
    ('yl4579/StyleTTS2-LibriTTS', 'Models/LibriTTS/epochs_2nd_00020.pth', 'model', os.path.join(repo_root, 'weights'), None),
]

for repo, filename, repo_type, local_dir, extracted_dir in downloads:
    dest = os.path.join(local_dir, filename)
    if extracted_dir and os.path.isdir(extracted_dir):
        print(f'  skipping {filename} ({extracted_dir} already exists)')
        continue
    if os.path.exists(dest):
        print(f'  skipping {filename} (already exists)')
        continue
    os.makedirs(local_dir, exist_ok=True)
    kwargs = dict(repo_type=repo_type, local_dir=local_dir, token=token)
    if repo_type == 'model':
        kwargs.pop('repo_type')
    print(f'Downloading {filename} from {repo}...')
    hf_hub_download(repo, filename, **kwargs)
    print(f'  done.')

print('All downloads complete')
"


echo "--- Extracting dataset ---"
if [ ! -d "$REPO_ROOT/combined_dataset" ]; then
    tar --warning=no-unknown-keyword -xzf $REPO_ROOT/combined_dataset.tar.gz -C $REPO_ROOT
    rm $REPO_ROOT/combined_dataset.tar.gz
else
    echo "  skipping (already extracted)"
    rm -f $REPO_ROOT/combined_dataset.tar.gz
fi

echo "--- Extracting alignments ---"
if [ ! -d "$REPO_ROOT/alignments" ]; then
    tar --warning=no-unknown-keyword -xzf $REPO_ROOT/alignments.tar.gz -C $REPO_ROOT
    rm $REPO_ROOT/alignments.tar.gz
else
    echo "  skipping (already extracted)"
    rm -f $REPO_ROOT/alignments.tar.gz
fi

echo "--- Sanitizing manifests ---"
"$PY" -c "
from pathlib import Path
import soundfile as sf

repo_root = Path('$REPO_ROOT')
manifests = [
    repo_root / 'StyleTTS2/Data/tr_train.txt',
    repo_root / 'StyleTTS2/Data/tr_val.txt',
]

for manifest in manifests:
    cleaned = []
    removed = 0

    with manifest.open(encoding='utf-8') as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split('|')
            if len(parts) != 3:
                removed += 1
                print(f'  removing {manifest.name}:{line_no} (bad column count)')
                continue

            wav_path = repo_root / parts[0]
            try:
                sf.info(str(wav_path))
            except Exception as exc:
                removed += 1
                print(f'  removing {manifest.name}:{line_no} ({parts[0]}: {exc})')
                continue

            cleaned.append(line)

    with manifest.open('w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned))
        if cleaned:
            f.write('\n')

    print(f'  {manifest.name}: kept {len(cleaned)}, removed {removed}')
"

echo "--- Precomputing mel cache ---"
"$PY" precompute_mels.py \
    --root "$REPO_ROOT" \
    --cache-dir "$REPO_ROOT/mel_cache" \
    --workers "$MEL_CACHE_WORKERS" \
    --chunksize "$MEL_CACHE_CHUNKSIZE" \
    --manifests \
    "$REPO_ROOT/StyleTTS2/Data/tr_train.txt" \
    "$REPO_ROOT/StyleTTS2/Data/tr_val.txt"

echo "--- Verifying setup ---"
"$PY" -c "
import os
checks = [
    'combined_dataset/manifest_phonemized.csv',
    'alignments/male_000001.pt',
    'StyleTTS2/Utils/PLBERT_turkish/step_160000.t7',
    'StyleTTS2/Utils/JDC/bst.t7',
    'StyleTTS2/Data/tr_train.txt',
    'StyleTTS2/Data/tr_val.txt',
    'StyleTTS2/Data/tr_ood.txt',
    'mel_cache/combined_dataset/male_000001.pt',
]
all_ok = True
for f in checks:
    exists = os.path.exists(f)
    print(f'  {\"OK\" if exists else \"MISSING\"}: {f}')
    if not exists:
        all_ok = False
if not all_ok:
    raise SystemExit('Missing files — aborting')
print('All checks passed')
"

# Launch training
echo "=== Launching Turkish finetune training ==="
cd $REPO_ROOT/StyleTTS2

GPU_COUNT=$("$PY" -c "import torch; print(torch.cuda.device_count())")
echo "  GPUs detected: $GPU_COUNT"

mkdir -p Models/Turkish
"$PY" -m accelerate.commands.launch \
    --num_processes "$GPU_COUNT" \
    --mixed_precision "bf16" \
    train_finetune_tr.py -p Configs/config_turkish.yml \
    2>&1 | tee Models/Turkish/stdout.log
