"""
Phonemize manifest.csv using espeak-ng Turkish IPA.
Writes manifest_phonemized.csv with an added 'phonemes' column.
"""

import csv
import subprocess
import sys
from pathlib import Path

ESPEAK = "/opt/homebrew/bin/espeak-ng"
MANIFEST = Path("combined_dataset/manifest.csv")
OUTPUT = Path("combined_dataset/manifest_phonemized.csv")
BATCH_SIZE = 500  # lines per espeak call


def phonemize_batch(texts: list[str]) -> list[str]:
    # Blank line between texts causes espeak-ng to output one IPA line per text
    joined = "\n\n".join(texts) + "\n"
    result = subprocess.run(
        [ESPEAK, "-v", "tr", "--ipa", "-q", "--stdin"],
        input=joined,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        print(f"espeak-ng error: {result.stderr}", file=sys.stderr)
    phoneme_lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
    if len(phoneme_lines) != len(texts):
        # fallback: phonemize individually
        phoneme_lines = []
        for t in texts:
            r = subprocess.run(
                [ESPEAK, "-v", "tr", "--ipa", "-q", "--stdin"],
                input=t + "\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            phoneme_lines.append(r.stdout.strip())
    return phoneme_lines


def main():
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["phonemes"]
        rows = list(reader)

    total = len(rows)
    print(f"Phonemizing {total} rows...")

    results = []
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        phonemes = phonemize_batch(texts)
        results.extend(phonemes)
        print(f"  {min(i + BATCH_SIZE, total)}/{total}", end="\r")

    print()

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, ph in zip(rows, results):
            row["phonemes"] = ph
            writer.writerow(row)

    print(f"Saved to {OUTPUT}")
    # Show a few examples
    for row, ph in zip(rows[:3], results[:3]):
        print(f"  {row['text']!r}")
        print(f"  -> {ph!r}")
        print()


if __name__ == "__main__":
    main()
