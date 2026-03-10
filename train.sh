#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
HF_DATASET="vsqrd/styletts2-turkish"
HF_PLBERT="vsqrd/pl-bert-turkish"

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
import os, subprocess, sys
from huggingface_hub import hf_hub_download, snapshot_download

token = os.environ['HF_TOKEN']

print('Downloading combined_dataset.tar.gz...')
hf_hub_download('vsqrd/styletts2-turkish', 'combined_dataset.tar.gz',
    repo_type='dataset', local_dir='$REPO_ROOT', token=token)

print('Downloading alignments.tar.gz...')
hf_hub_download('vsqrd/styletts2-turkish', 'alignments.tar.gz',
    repo_type='dataset', local_dir='$REPO_ROOT', token=token)

print('Downloading Turkish PL-BERT checkpoint...')
os.makedirs('$REPO_ROOT/StyleTTS2/Utils/PLBERT_turkish', exist_ok=True)
hf_hub_download('vsqrd/pl-bert-turkish', 'step_160000.t7',
    local_dir='$REPO_ROOT/StyleTTS2/Utils/PLBERT_turkish', token=token)

print('Downloading F0 model...')
hf_hub_download('yl4579/StyleTTS2-LibriTTS', 'Utils/JDC/bst.t7',
    local_dir='$REPO_ROOT/StyleTTS2')

print('All downloads complete')
"

echo "--- Extracting dataset ---"
tar -xzf $REPO_ROOT/combined_dataset.tar.gz -C $REPO_ROOT
rm $REPO_ROOT/combined_dataset.tar.gz

echo "--- Extracting alignments ---"
tar -xzf $REPO_ROOT/alignments.tar.gz -C $REPO_ROOT
rm $REPO_ROOT/alignments.tar.gz

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
echo "=== Launching stage 1 training ==="
cd $REPO_ROOT/StyleTTS2
"$PY" train_first_tr.py -p Configs/config_turkish.yml
