import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

import numpy as np
import torch
from tqdm import tqdm

from StyleTTS2.meldataset import preprocess
import soundfile as sf
import librosa


def load_wave(path, target_sr=24000):
    wave, sr = sf.read(path)
    if wave.ndim == 2:
        wave = wave[:, 0].squeeze()
    if sr != target_sr:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
    return wave


def iter_manifest_paths(manifest_path):
    with open(manifest_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 1:
                continue
            yield parts[0]


def build_cache_path(cache_root, wav_rel_path):
    return cache_root / Path(wav_rel_path).with_suffix(".pt")


def process_one(wav_rel_path, root, cache_root, sr, overwrite):
    cache_path = build_cache_path(cache_root, wav_rel_path)
    if cache_path.exists() and not overwrite:
        return "skipped"

    wav_path = root / wav_rel_path
    wave = load_wave(wav_path, target_sr=sr)
    wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
    mel = preprocess(wave).squeeze(0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mel, cache_path)
    return "written"


def main():
    parser = argparse.ArgumentParser(description="Precompute mel caches for StyleTTS2 manifests.")
    parser.add_argument("--root", required=True, help="Root directory used to resolve wav paths from manifests.")
    parser.add_argument("--cache-dir", required=True, help="Output directory for cached mel tensors.")
    parser.add_argument("--manifests", nargs="+", required=True, help="Manifest files to scan.")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing cache files.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    args = parser.parse_args()

    root = Path(args.root)
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    wav_paths = []
    seen = set()
    for manifest in args.manifests:
        for wav_rel_path in iter_manifest_paths(manifest):
            if wav_rel_path not in seen:
                seen.add(wav_rel_path)
                wav_paths.append(wav_rel_path)

    written = 0
    skipped = 0
    if args.workers <= 1:
        for wav_rel_path in tqdm(wav_paths, desc="Caching mels"):
            status = process_one(wav_rel_path, root, cache_root, args.sr, args.overwrite)
            if status == "written":
                written += 1
            else:
                skipped += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one, wav_rel_path, root, cache_root, args.sr, args.overwrite): wav_rel_path
                for wav_rel_path in wav_paths
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Caching mels"):
                status = future.result()
                if status == "written":
                    written += 1
                else:
                    skipped += 1

    print(f"Processed {len(wav_paths)} files | wrote {written} | skipped {skipped}")


if __name__ == "__main__":
    main()
