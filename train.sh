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

downloads = [
    ('vsqrd/styletts2-turkish', 'combined_dataset.tar.gz', 'dataset', '$REPO_ROOT'),
    ('vsqrd/styletts2-turkish', 'alignments.tar.gz',       'dataset', '$REPO_ROOT'),
    ('vsqrd/pl-bert-turkish',   'step_160000.t7',          'model',   '$REPO_ROOT/StyleTTS2/Utils/PLBERT_turkish'),
]

for repo, filename, repo_type, local_dir in downloads:
    dest = os.path.join(local_dir, filename)
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
echo "  num_processes=$NUM_PROCESSES"
echo "  mixed_precision=$MIXED_PRECISION"
exec "$PY" -m accelerate.commands.launch \
    --num_processes "$NUM_PROCESSES" \
    --num_machines "$NUM_MACHINES" \
    --machine_rank "$MACHINE_RANK" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision "$MIXED_PRECISION" \
    train_first_tr.py -p Configs/config_turkish.yml
