"""
Convert manifest_phonemized.csv to StyleTTS2 pipe-separated train/val lists.

Output format per line:
    combined_dataset/male_000001.wav|mˈɛrhaba metˈɪn bˈɛj|0

Speaker mapping: male_speaker → 0, female_speaker → 1
"""

import csv
import random
from pathlib import Path

MANIFEST = "combined_dataset/manifest_phonemized.csv"
OUT_DIR = Path("StyleTTS2/Data")
TRAIN_FILE = OUT_DIR / "tr_train.txt"
VAL_FILE = OUT_DIR / "tr_val.txt"
VAL_RATIO = 0.05
SEED = 42

SPEAKER_MAP = {"male_speaker": 0, "female_speaker": 1}

random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
skipped = 0

with open(MANIFEST, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        phonemes = row["phonemes"].replace("\r", "").replace("\n", " ").strip()

        # skip corrupted rows (non-Turkish text phonemized with language tags)
        if "(" in phonemes or ")" in phonemes:
            skipped += 1
            continue

        # strip leading "data/" so path is relative to repo root
        wav_path = row["file"]
        if wav_path.startswith("data/"):
            wav_path = wav_path[len("data/"):]

        speaker_id = SPEAKER_MAP[row["speaker_id"]]
        rows.append(f"{wav_path}|{phonemes}|{speaker_id}")

print(f"Total rows: {len(rows)} | Skipped: {skipped}")

# stratified split per speaker
male = [r for r in rows if r.endswith("|0")]
female = [r for r in rows if r.endswith("|1")]

random.shuffle(male)
random.shuffle(female)

def split(lst):
    n_val = max(1, int(len(lst) * VAL_RATIO))
    return lst[n_val:], lst[:n_val]

male_train, male_val = split(male)
female_train, female_val = split(female)

train = male_train + female_train
val = male_val + female_val
random.shuffle(train)

with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(train))

with open(VAL_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(val))

print(f"Train: {len(train)} | Val: {len(val)}")
print(f"Written to {TRAIN_FILE} and {VAL_FILE}")
