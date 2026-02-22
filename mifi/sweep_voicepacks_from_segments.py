import argparse
import csv
from pathlib import Path

import torch
from loguru import logger

from kokoro.model import KModel
from kokoro.pipeline import KPipeline
from mifi.style_inversion import build_voicepack, invert_style_vector


def read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["start_sec"] = float(row["start_sec"])
            row["end_sec"] = float(row["end_sec"])
            row["duration"] = row["end_sec"] - row["start_sec"]
            rows.append(row)
    return rows


def invert_bank(
    rows: list[dict],
    segments_dir: Path,
    model: KModel,
    device: str,
    steps: int,
    lr: float,
    phonemes: str | None,
) -> list[dict]:
    out: list[dict] = []
    for i, row in enumerate(rows, start=1):
        clip = segments_dir / row["filename"]
        logger.info(f"[{i}/{len(rows)}] invert {clip.name} ({row['duration']:.2f}s)")
        kwargs = dict(
            reference_audio_path=str(clip),
            model=model,
            n_steps=steps,
            lr=lr,
            device=device,
            return_loss=True,
        )
        if phonemes:
            kwargs["phonemes"] = phonemes
        s, loss = invert_style_vector(**kwargs)
        out.append(
            {
                **row,
                "loss": float(loss),
                "s": s.detach().cpu().squeeze(0),  # [256]
            }
        )
    return out


def make_candidates(bank: list[dict]) -> list[tuple[str, torch.Tensor, dict]]:
    # Return (name, s_vector[1,256], metadata)
    by_loss = sorted(bank, key=lambda x: x["loss"])
    by_dur = sorted(bank, key=lambda x: x["duration"]) 

    def avg(items: list[dict]) -> torch.Tensor:
        return torch.stack([it["s"] for it in items], dim=0).mean(dim=0, keepdim=True)

    eps = 1e-6
    inv_loss_w = torch.tensor([1.0 / (it["loss"] + eps) for it in bank], dtype=torch.float32)
    inv_loss_w = inv_loss_w / inv_loss_w.sum()
    weighted = torch.sum(
        torch.stack([it["s"] for it in bank], dim=0) * inv_loss_w[:, None], dim=0, keepdim=True
    )

    # keep medium-long clips only (usually cleaner than very short clips)
    med = [it for it in bank if 3.0 <= it["duration"] <= 12.0]
    if not med:
        med = bank

    candidates: list[tuple[str, torch.Tensor, dict]] = [
        ("cand_mean_all", avg(bank), {"n": len(bank), "mean_loss": sum(x["loss"] for x in bank) / len(bank)}),
        ("cand_weighted_inv_loss", weighted, {"n": len(bank), "mean_loss": sum(x["loss"] for x in bank) / len(bank)}),
        ("cand_top8_loss", avg(by_loss[: min(8, len(by_loss))]), {"n": min(8, len(by_loss)), "mean_loss": sum(x["loss"] for x in by_loss[: min(8, len(by_loss))]) / min(8, len(by_loss))}),
        ("cand_top6_loss", avg(by_loss[: min(6, len(by_loss))]), {"n": min(6, len(by_loss)), "mean_loss": sum(x["loss"] for x in by_loss[: min(6, len(by_loss))]) / min(6, len(by_loss))}),
        ("cand_med_duration", avg(med), {"n": len(med), "mean_loss": sum(x["loss"] for x in med) / len(med)}),
    ]

    if len(by_loss) >= 5:
        trimmed = by_loss[1:-1]
        candidates.append(
            (
                "cand_trimmed_loss",
                avg(trimmed),
                {"n": len(trimmed), "mean_loss": sum(x["loss"] for x in trimmed) / len(trimmed)},
            )
        )

    if len(by_dur) >= 6:
        # drop two shortest clips
        no_short = by_dur[2:]
        candidates.append(
            (
                "cand_no_short",
                avg(no_short),
                {"n": len(no_short), "mean_loss": sum(x["loss"] for x in no_short) / len(no_short)},
            )
        )

    return candidates


def synth_test_wavs(candidate_paths: list[Path], test_text: str, out_dir: Path, device: str) -> None:
    pipe = KPipeline(lang_code="a", model=True, device=device)
    import soundfile as sf

    for p in candidate_paths:
        out_wav = out_dir / f"{p.stem}_test.wav"
        for r in pipe(test_text, voice=str(p), speed=1.0):
            if r.audio is not None:
                sf.write(str(out_wav), r.audio, 24000)
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Kokoro voicepack configs from sentence segments")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--segments-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="weights/kokoro-v1_0.pth")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--phonemes", default=None)
    parser.add_argument("--test-text", default="This is a test for choosing the best Andre Santana voice candidate.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(Path(args.manifest))
    logger.info(f"Loaded {len(rows)} segments")

    model = KModel(repo_id="hexgrad/Kokoro-82M", config=args.config, model=args.model).eval()

    bank = invert_bank(
        rows=rows,
        segments_dir=Path(args.segments_dir),
        model=model,
        device=args.device,
        steps=args.steps,
        lr=args.lr,
        phonemes=args.phonemes,
    )

    bank_csv = out_dir / "segment_losses.tsv"
    with bank_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["filename", "duration", "loss", "text"])
        for r in sorted(bank, key=lambda x: x["loss"]):
            w.writerow([r["filename"], f"{r['duration']:.3f}", f"{r['loss']:.6f}", r["text"]])

    candidates = make_candidates(bank)

    ranking_rows = []
    candidate_paths: list[Path] = []
    for name, s, meta in candidates:
        pack = build_voicepack(s)
        p = out_dir / f"{name}.pt"
        torch.save(pack, p)
        candidate_paths.append(p)
        ranking_rows.append((name, meta["n"], meta["mean_loss"]))
        logger.info(f"Saved {p.name}  n={meta['n']}  mean_loss={meta['mean_loss']:.5f}")

    ranking_rows.sort(key=lambda x: x[2])
    rank_tsv = out_dir / "candidate_ranking.tsv"
    with rank_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["candidate", "n_segments", "mean_segment_loss"])
        for row in ranking_rows:
            w.writerow([row[0], row[1], f"{row[2]:.6f}"])

    synth_test_wavs(candidate_paths, args.test_text, out_dir, args.device)
    logger.info(f"Done. Ranking: {rank_tsv}")


if __name__ == "__main__":
    main()
