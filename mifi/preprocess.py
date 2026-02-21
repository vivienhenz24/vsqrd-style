"""
Preprocess a dataset into the StyleTTS2 training format.

Input filelist format (one per line):
    /path/to/audio.wav|raw transcript text|speaker_id

Output filelist format (one per line):
    /path/to/audio.wav|IPA phonemes|speaker_id

Usage:
    uv run python -m mifi.preprocess --input data/raw.txt --output data/train_list.txt
"""

import argparse
from pathlib import Path

from loguru import logger

from mifi.fe.txt_norm import normalize_text
from mifi.fe.g2p import text_to_phonemes


def process_line(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None

    parts = line.split("|")
    if len(parts) < 2:
        logger.warning(f"Skipping malformed line: {line!r}")
        return None

    wav_path, text = parts[0], parts[1]
    speaker_id = parts[2] if len(parts) > 2 else "0"

    normalized = normalize_text(text)
    phonemes = text_to_phonemes(normalized)
    ipa = " ".join(phonemes)

    return f"{wav_path}|{ipa}|{speaker_id}"


def main():
    parser = argparse.ArgumentParser(description="Preprocess text to IPA for StyleTTS2 training")
    parser.add_argument("--input", required=True, help="Input filelist with raw transcripts")
    parser.add_argument("--output", required=True, help="Output filelist with IPA phonemes")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    logger.info(f"Processing {len(lines)} lines from {input_path}")

    processed, skipped = [], 0
    for i, line in enumerate(lines):
        result = process_line(line)
        if result is None:
            skipped += 1
        else:
            processed.append(result)
        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{len(lines)}")

    output_path.write_text("\n".join(processed) + "\n", encoding="utf-8")
    logger.info(f"Done. {len(processed)} written, {skipped} skipped → {output_path}")


if __name__ == "__main__":
    main()
