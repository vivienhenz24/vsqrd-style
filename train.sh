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
source "$REPO_ROOT/.venv/bin/activate"

# Download and extract dataset
echo "--- Downloading dataset ---"
huggingface-cli download $HF_DATASET combined_dataset.tar.gz \
    --repo-type dataset --local-dir $REPO_ROOT --quiet
echo "--- Extracting dataset ---"
tar -xzf $REPO_ROOT/combined_dataset.tar.gz -C $REPO_ROOT
rm $REPO_ROOT/combined_dataset.tar.gz

# Download and extract alignments
echo "--- Downloading alignments ---"
huggingface-cli download $HF_DATASET alignments.tar.gz \
    --repo-type dataset --local-dir $REPO_ROOT --quiet
echo "--- Extracting alignments ---"
tar -xzf $REPO_ROOT/alignments.tar.gz -C $REPO_ROOT
rm $REPO_ROOT/alignments.tar.gz

# Download Turkish PL-BERT (160k checkpoint only)
echo "--- Downloading Turkish PL-BERT ---"
mkdir -p $REPO_ROOT/StyleTTS2/Utils/PLBERT_turkish
huggingface-cli download $HF_PLBERT \
    config.yml \
    step_160000.t7 \
    --local-dir $REPO_ROOT/StyleTTS2/Utils/PLBERT_turkish \
    --quiet

# Rename config_ml.yml if needed (already handled by our config.yml)
# The config.yml is already in the repo

# Download JDC pitch model (from StyleTTS2 LibriTTS repo)
echo "--- Downloading F0 model ---"
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('yl4579/StyleTTS2-LibriTTS', 'Utils/JDC/bst.t7', local_dir='StyleTTS2')
print('F0 model downloaded')
"

echo "--- Verifying setup ---"
python3 -c "
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
python train_first_tr.py -p Configs/config_turkish.yml
