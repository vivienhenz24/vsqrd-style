import argparse
from pathlib import Path

import torch

from loguru import logger
from kokoro.model import KModel
from mifi.style_inversion import build_voicepack, invert_style_vector


def _read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        idx = {k: i for i, k in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "filename": parts[idx["filename"]],
                    "start_sec": float(parts[idx["start_sec"]]),
                    "end_sec": float(parts[idx["end_sec"]]),
                    "text": parts[idx["text"]],
                    "duration": float(parts[idx["end_sec"]]) - float(parts[idx["start_sec"]]),
                }
            )
    return rows


def _select_segments(rows: list[dict], max_segments: int, min_sec: float, max_sec: float) -> list[dict]:
    filtered = [r for r in rows if min_sec <= r["duration"] <= max_sec]
    filtered.sort(key=lambda r: r["duration"], reverse=True)
    return filtered[:max_segments]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Kokoro voicepack from sentence segments.")
    parser.add_argument("--manifest", required=True, help="TSV manifest with filename/start/end/text")
    parser.add_argument("--segments-dir", required=True, help="Directory containing segment wav files")
    parser.add_argument("--output", required=True, help="Output voicepack .pt path")
    parser.add_argument("--model", default="weights/kokoro-v1_0.pth", help="Kokoro model weights")
    parser.add_argument("--config", default=None, help="Optional Kokoro config json")
    parser.add_argument("--max-segments", type=int, default=6, help="Max segments to invert")
    parser.add_argument("--min-sec", type=float, default=3.0, help="Min segment duration")
    parser.add_argument("--max-sec", type=float, default=14.0, help="Max segment duration")
    parser.add_argument("--steps", type=int, default=200, help="Optimization steps per segment")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--phonemes", default=None, help="Optional IPA string override")
    parser.add_argument("--device", default=None, help="cpu/cuda/mps (auto if omitted)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    segments_dir = Path(args.segments_dir)
    output_path = Path(args.output)

    rows = _read_manifest(manifest_path)
    selected = _select_segments(rows, args.max_segments, args.min_sec, args.max_sec)
    if not selected:
        raise ValueError("No segments matched selection criteria.")

    logger.info(f"Selected {len(selected)} segments for inversion")
    for i, r in enumerate(selected, start=1):
        logger.info(f"  {i:02d}. {r['filename']}  ({r['duration']:.2f}s)")

    model = KModel(repo_id="hexgrad/Kokoro-82M", config=args.config, model=args.model).eval()

    s_list = []
    for i, r in enumerate(selected, start=1):
        clip_path = segments_dir / r["filename"]
        logger.info(f"[{i}/{len(selected)}] Inverting {clip_path.name}")
        kwargs = dict(
            reference_audio_path=str(clip_path),
            model=model,
            n_steps=args.steps,
            lr=args.lr,
            device=args.device,
        )
        if args.phonemes is not None:
            kwargs["phonemes"] = args.phonemes
        s = invert_style_vector(**kwargs)
        s_list.append(s.detach().cpu())

    s_mean = torch.stack(s_list, dim=0).mean(dim=0)
    pack = build_voicepack(s_mean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, output_path)
    logger.info(f"Saved voicepack: {output_path} shape={tuple(pack.shape)}")


if __name__ == "__main__":
    main()
