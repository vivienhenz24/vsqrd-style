"""
Compute phoneme-level alignment matrices for StyleTTS2 training.

For each utterance in combined_dataset:
1. Call ElevenLabs STT API (async, batched) to get character-level timestamps
2. Map character timestamps → phoneme timestamps (Turkish is ~phonemic)
3. Save hard monotonic alignment matrix as .pt file: shape (T_phoneme, T_frames)

Usage:
    uv run align_dataset.py [--concurrency 64] [--batch-size 200] [--output-dir alignments/]
"""

import os
import csv
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

import torch
import numpy as np
import soundfile as sf
from elevenlabs import AsyncElevenLabs

load_dotenv()

# StyleTTS2 mel params (must match meldataset.py)
SR = 24000
HOP_LENGTH = 300

API_KEY = os.environ["ELEVENLABS_API_KEY"]


def get_audio_duration_frames(wav_path: str) -> int:
    info = sf.info(wav_path)
    n_samples = int(info.frames * SR / info.samplerate)
    n_samples += 10000  # StyleTTS2 pads 5000 zeros each side
    n_frames = n_samples // HOP_LENGTH
    if n_frames % 2 != 0:
        n_frames -= 1
    return n_frames // 2  # ASRCNN stride-2 downsample


async def elevenlabs_align_async(client: AsyncElevenLabs, wav_path: str) -> list[dict]:
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    response = await client.speech_to_text.convert(
        file=("audio.wav", audio_bytes, "audio/wav"),
        model_id="scribe_v1",
        language_code="tur",
        timestamps_granularity="character",
    )

    result = []
    for word in (response.words or []):
        if word.type != "word":
            continue
        for c in (word.characters or []):
            if c.text.strip():
                result.append({"character": c.text, "start": c.start, "end": c.end})
    return result


def chars_to_phoneme_spans(char_timings: list[dict], phonemes: str, audio_duration: float) -> list[tuple]:
    LEAD_PAD = 5000 / SR
    DIACRITICS = set("ˈˌːˑ")

    phoneme_str = phonemes.replace("\n", " ").strip()
    tokens = list(phoneme_str)

    if not char_timings:
        dur = audio_duration / max(len(tokens), 1)
        return [(LEAD_PAD + i * dur, LEAD_PAD + (i + 1) * dur) for i in range(len(tokens))]

    timings = [
        {"character": c["character"], "start": c["start"] + LEAD_PAD, "end": c["end"] + LEAD_PAD}
        for c in char_timings
    ]

    segments = [t for t in tokens if t not in DIACRITICS and t != " "]
    n_seg = len(segments)
    n_chars = len(timings)

    if n_seg == 0:
        return []

    def seg_span(seg_idx):
        ci = int(round(seg_idx * (n_chars - 1) / max(n_seg - 1, 1)))
        return timings[min(ci, n_chars - 1)]["start"], timings[min(ci, n_chars - 1)]["end"]

    seg_idx = 0
    spans = []
    for tok in tokens:
        if tok in DIACRITICS:
            prev = next((s for s in reversed(spans) if s is not None), (timings[0]["start"], timings[0]["end"]))
            spans.append(prev)
        elif tok == " ":
            prev = next((s for s in reversed(spans) if s is not None), None)
            future_segs = segments[seg_idx:]
            if prev and future_segs:
                next_span = seg_span(seg_idx)
                spans.append((prev[1], (prev[1] + next_span[0]) / 2))
            elif prev:
                spans.append((prev[1], prev[1]))
            else:
                spans.append((LEAD_PAD, LEAD_PAD))
        else:
            spans.append(seg_span(seg_idx))
            seg_idx += 1

    return spans


def build_alignment_matrix(spans: list[tuple], n_frames: int) -> torch.Tensor:
    T_ph = len(spans)
    if T_ph == 0:
        raise ValueError("No spans")

    frame_dur = HOP_LENGTH * 2 / SR  # 0.025 s per output frame

    centers = np.array([(s + e) / 2 for s, e in spans])

    boundaries = np.zeros(T_ph + 1, dtype=np.int64)
    boundaries[0] = 0
    boundaries[T_ph] = n_frames
    for i in range(1, T_ph):
        boundaries[i] = int((centers[i - 1] + centers[i]) / 2 / frame_dur)

    boundaries = np.clip(boundaries, 0, n_frames)
    for i in range(1, T_ph + 1):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + 1
    boundaries = np.clip(boundaries, 0, n_frames)

    attn = torch.zeros(T_ph, n_frames)
    for ph in range(T_ph):
        lo, hi = int(boundaries[ph]), int(boundaries[ph + 1])
        if lo < hi:
            attn[ph, lo:hi] = 1.0
        else:
            attn[ph, min(lo, n_frames - 1)] = 1.0

    return attn


async def process_row_async(
    client: AsyncElevenLabs,
    sem: asyncio.Semaphore,
    row: dict,
    root_dir: Path,
    output_dir: Path,
    retry: int = 3,
) -> str:
    wav_rel = row["file"]
    wav_path = root_dir / Path(wav_rel).relative_to("data") if wav_rel.startswith("data/") else root_dir / wav_rel
    out_path = output_dir / (Path(wav_rel).stem + ".pt")

    if out_path.exists():
        return "skip"

    info = sf.info(str(wav_path))
    audio_duration = info.frames / info.samplerate
    n_frames = get_audio_duration_frames(str(wav_path))

    for attempt in range(retry):
        try:
            async with sem:
                char_timings = await elevenlabs_align_async(client, str(wav_path))
            break
        except Exception as e:
            if attempt == retry - 1:
                raise
            wait = 5 if "429" in str(e) or "concurrent_limit" in str(e) else 2 ** attempt
            await asyncio.sleep(wait)

    spans = chars_to_phoneme_spans(char_timings, row["phonemes"], audio_duration)
    attn = build_alignment_matrix(spans, n_frames)
    torch.save(attn, str(out_path))
    return "done"


async def run(args):
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.limit:
        rows = rows[: args.limit]

    # Filter already-done rows upfront
    pending = [r for r in rows if not (output_dir / (Path(r["file"]).stem + ".pt")).exists()]
    n_skip = len(rows) - len(pending)
    total = len(rows)
    print(f"Total: {total} | Already done: {n_skip} | To process: {len(pending)}")

    if not pending:
        return

    sem = asyncio.Semaphore(args.concurrency)
    done = fail = 0
    completed = 0

    client = AsyncElevenLabs(api_key=API_KEY)
    if True:
        # Process in batches so memory doesn't blow up with 47k tasks at once
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start: batch_start + args.batch_size]
            tasks = [
                asyncio.create_task(process_row_async(client, sem, row, root_dir, output_dir))
                for row in batch
            ]
            for coro in asyncio.as_completed(tasks):
                row = batch[tasks.index(coro)] if False else None  # just for error reporting
                try:
                    result = await coro
                    done += (result == "done")
                except Exception as e:
                    fail += 1
                    print(f"FAIL: {e}")
                completed += 1
                n_total_done = n_skip + done
                if completed % 500 == 0 or completed == len(pending):
                    print(f"[{n_total_done + fail}/{total}] done={n_total_done} fail={fail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="combined_dataset/manifest_phonemized.csv")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-dir", default="alignments/")
    parser.add_argument("--concurrency", type=int, default=32, help="max simultaneous API requests")
    parser.add_argument("--batch-size", type=int, default=500, help="rows per asyncio batch")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
