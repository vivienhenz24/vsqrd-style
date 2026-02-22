import argparse
import csv
import random
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger
from misaki import en, espeak

from kokoro.model import KModel
from mifi.style_inversion import (
    SAMPLE_RATE,
    _load_audio,
    _mel_band_means,
    _tokenize,
    build_voicepack,
    invert_style_vector,
)
from mifi.losses import MultiResolutionSTFTLoss


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            start = float(row["start_sec"])
            end = float(row["end_sec"])
            rows.append(
                {
                    "filename": row["filename"],
                    "start_sec": start,
                    "end_sec": end,
                    "duration": end - start,
                    "text": row["text"].strip(),
                }
            )
    return rows


def split_train_val(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rows = rows.copy()
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio)) if len(rows) > 1 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:] if n_val > 0 else rows
    if not train_rows:
        train_rows = rows
        val_rows = rows[:1]
    return train_rows, val_rows


def load_style_from_voicepack(path: Path) -> torch.Tensor:
    pack = torch.load(path, map_location="cpu", weights_only=True)
    # [510,1,256] -> [1,256]
    return pack[pack.shape[0] // 2].clone()


def build_g2p():
    fallback = espeak.EspeakFallback(british=False)
    return en.G2P(trf=False, british=False, fallback=fallback, unk="")


def text_to_phonemes(g2p, text: str) -> str:
    phonemes, _ = g2p(text)
    return phonemes.strip()


def pretokenize_rows(rows: list[dict], g2p, model: KModel) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        phonemes = text_to_phonemes(g2p, row["text"])
        ids = _tokenize(phonemes, model.vocab).squeeze(0)  # [T]
        if ids.shape[0] <= 2:
            continue
        if ids.shape[0] > model.context_length:
            continue
        out.append({**row, "phonemes": phonemes, "token_ids": ids, "token_len": int(ids.shape[0])})
    return out


def set_trainable_modules(model: KModel, train_modules: set[str]) -> None:
    for p in model.parameters():
        p.requires_grad_(False)

    allowed = {
        "bert": model.bert,
        "bert_encoder": model.bert_encoder,
        "text_encoder": model.text_encoder,
        "predictor": model.predictor,
        "decoder": model.decoder,
    }
    unknown = sorted(train_modules - set(allowed.keys()))
    if unknown:
        raise ValueError(f"Unknown modules in --train-modules: {unknown}")

    for name in train_modules:
        for p in allowed[name].parameters():
            p.requires_grad_(True)


def build_optimizer(model: KModel, train_modules: set[str], lr_main: float, lr_bert: float) -> torch.optim.Optimizer:
    params = []
    if "decoder" in train_modules:
        params.append({"params": model.decoder.parameters(), "lr": lr_main})
    if "predictor" in train_modules:
        params.append({"params": model.predictor.parameters(), "lr": lr_main})
    if "text_encoder" in train_modules:
        params.append({"params": model.text_encoder.parameters(), "lr": lr_main})
    if "bert_encoder" in train_modules:
        params.append({"params": model.bert_encoder.parameters(), "lr": lr_main})
    if "bert" in train_modules:
        params.append({"params": model.bert.parameters(), "lr": lr_bert})

    if not params:
        raise ValueError("No trainable modules selected.")
    return torch.optim.AdamW(params, betas=(0.9, 0.99), weight_decay=1e-4)


def synth_train_forward(
    model: KModel,
    input_ids: torch.LongTensor,
    ref_s: torch.Tensor,
    speed: float = 1.0,
) -> torch.Tensor:
    # Same as KModel.forward_with_tokens, but returns device tensor with grad path.
    input_lengths = torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1],
        device=input_ids.device,
        dtype=torch.long,
    )

    text_mask = (
        torch.arange(input_lengths.max(), device=input_ids.device)
        .unsqueeze(0)
        .expand(input_lengths.shape[0], -1)
        .type_as(input_lengths)
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(model.device)

    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

    s_prosody = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s_prosody, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed

    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
    indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=model.device), pred_dur)

    pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=model.device)
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0)

    en = d.transpose(-1, -2) @ pred_aln_trg
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s_prosody)

    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg

    return model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()


def synth_train_forward_batched(
    model: KModel,
    input_ids: torch.LongTensor,   # [B, T]
    input_lengths: torch.LongTensor,  # [B]
    ref_s: torch.Tensor,  # [B, 256]
    speed: float = 1.0,
) -> torch.Tensor:
    text_mask = (
        torch.arange(input_ids.shape[1], device=input_ids.device)
        .unsqueeze(0)
        .expand(input_ids.shape[0], -1)
        .type_as(input_lengths)
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(model.device)

    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

    s_prosody = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s_prosody, input_lengths, text_mask)  # [B, T, H]
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed  # [B, T]
    pred_dur = torch.round(duration).clamp(min=1).long()

    bsz, tlen = pred_dur.shape
    frame_lens = pred_dur.sum(dim=1)
    max_frames = int(frame_lens.max().item())
    pred_aln_trg = torch.zeros((bsz, tlen, max_frames), device=model.device)
    token_ix = torch.arange(tlen, device=model.device)
    for b in range(bsz):
        repeats = pred_dur[b]
        idx = torch.repeat_interleave(token_ix, repeats)
        pred_aln_trg[b, idx, torch.arange(idx.shape[0], device=model.device)] = 1

    en = d.transpose(-1, -2) @ pred_aln_trg
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s_prosody)
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg
    return model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze(1)  # [B, S]


def crop_to_min_len(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(min(a.shape[-1], b.shape[-1]))
    if n <= 0:
        raise ValueError("Encountered empty audio while cropping.")
    return a[..., :n], b[..., :n]


def run_epoch(
    rows: list[dict],
    model: KModel,
    optimizer: torch.optim.Optimizer,
    segments_dir: Path,
    ref_s: torch.Tensor,
    device: torch.device,
    to_mel_cpu,
    stft_loss,
    lambda_stft: float,
    lambda_melmean: float,
    train: bool,
    style_map: dict[str, torch.Tensor] | None = None,
    batch_size: int = 4,
    log_every: int = 0,
    epoch_idx: int = 0,
    total_epochs: int = 0,
    stage_name: str = "stage1",
    trace_every_step: bool = False,
    hang_threshold_sec: float = 20.0,
    log_timings: bool = False,
) -> float:
    model.train(train)
    total = 0.0
    count = 0
    t_load = 0.0
    t_g2p = 0.0
    t_forward = 0.0
    t_loss = 0.0
    t_backward_step = 0.0

    num_batches = (len(rows) + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_rows = rows[i * batch_size : (i + 1) * batch_size]
        mode = "train" if train else "val"
        if trace_every_step:
            logger.info(
                f"TRACE {mode} epoch={epoch_idx}/{total_epochs} stage={stage_name} "
                f"step={i+1}/{num_batches} batch_n={len(batch_rows)} phase=begin"
            )

        t0 = time.perf_counter()
        targets = []
        for row in batch_rows:
            wav_path = segments_dir / row["filename"]
            targets.append(_load_audio(str(wav_path)).to(device))
        dt = time.perf_counter() - t0
        t_load += dt
        if trace_every_step:
            logger.info(f"TRACE {mode} step={i+1} phase=load dt={dt:.3f}s")
        if dt > hang_threshold_sec:
            logger.warning(f"SLOW {mode} step={i+1} phase=load dt={dt:.3f}s")

        t0 = time.perf_counter()
        lengths = torch.LongTensor([row["token_len"] for row in batch_rows]).to(device)
        max_len = int(lengths.max().item())
        input_ids = torch.zeros((len(batch_rows), max_len), dtype=torch.long, device=device)
        for b, row in enumerate(batch_rows):
            tid = row["token_ids"].to(device)
            input_ids[b, : tid.shape[0]] = tid
        dt = time.perf_counter() - t0
        t_g2p += dt
        if trace_every_step:
            logger.info(f"TRACE {mode} step={i+1} phase=tokens dt={dt:.3f}s max_tokens={input_ids.shape[1]}")
        if dt > hang_threshold_sec:
            logger.warning(f"SLOW {mode} step={i+1} phase=tokens dt={dt:.3f}s")

        if train:
            optimizer.zero_grad(set_to_none=True)

        try:
            t0 = time.perf_counter()
            conds = []
            for row in batch_rows:
                c = style_map.get(row["filename"], ref_s) if style_map is not None else ref_s
                conds.append(c.to(device).squeeze(0))
            cond_s = torch.stack(conds, dim=0)  # [B,256]
            pred = synth_train_forward_batched(model, input_ids, lengths, cond_s)  # [B,S]
            dt = time.perf_counter() - t0
            t_forward += dt
            if trace_every_step:
                logger.info(f"TRACE {mode} step={i+1} phase=forward dt={dt:.3f}s")
            if dt > hang_threshold_sec:
                logger.warning(f"SLOW {mode} step={i+1} phase=forward dt={dt:.3f}s")

            t0 = time.perf_counter()
            # Per-sample losses, averaged over the batch.
            batch_losses = []
            for b, target in enumerate(targets):
                p = pred[b]
                p, t = crop_to_min_len(p, target)
                ls = stft_loss(p.unsqueeze(0), t.unsqueeze(0))
                lm = F.mse_loss(_mel_band_means(p, to_mel_cpu), _mel_band_means(t, to_mel_cpu))
                batch_losses.append(lambda_stft * ls + lambda_melmean * lm)
            loss = torch.stack(batch_losses).mean()
            dt = time.perf_counter() - t0
            t_loss += dt
            if trace_every_step:
                logger.info(f"TRACE {mode} step={i+1} phase=loss dt={dt:.3f}s loss={loss.item():.5f}")
            if dt > hang_threshold_sec:
                logger.warning(f"SLOW {mode} step={i+1} phase=loss dt={dt:.3f}s")

            if train:
                t0 = time.perf_counter()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                dt = time.perf_counter() - t0
                t_backward_step += dt
                if trace_every_step:
                    logger.info(f"TRACE {mode} step={i+1} phase=backward_step dt={dt:.3f}s")
                if dt > hang_threshold_sec:
                    logger.warning(f"SLOW {mode} step={i+1} phase=backward_step dt={dt:.3f}s")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"OOM at {mode} step={i+1}/{num_batches} (batch_n={len(batch_rows)}). "
                    "Skipping batch and clearing CUDA cache."
                )
                if train:
                    optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

        total += float(loss.item())
        count += 1

        if log_every > 0 and (i + 1) % log_every == 0:
            denom = max(1, count)
            logger.info(
                f"Epoch {epoch_idx:03d}/{total_epochs}  stage={stage_name}  "
                f"{mode} step {i+1:>5}/{num_batches}  "
                f"loss={loss.item():.5f}  running={total / max(1, count):.5f}"
            )
            if log_timings:
                logger.info(
                    f"  timing avg/batch ({mode}): "
                    f"load={1000*t_load/denom:.1f}ms  "
                    f"tokens={1000*t_g2p/denom:.1f}ms  "
                    f"forward={1000*t_forward/denom:.1f}ms  "
                    f"loss={1000*t_loss/denom:.1f}ms  "
                    f"backward_step={1000*t_backward_step/denom:.1f}ms"
                )
        if trace_every_step:
            logger.info(f"TRACE {mode} step={i+1} phase=end")

    return total / max(1, count)


def save_kokoro_checkpoint(model: KModel, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "bert": model.bert.state_dict(),
        "bert_encoder": model.bert_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "text_encoder": model.text_encoder.state_dict(),
        "decoder": model.decoder.state_dict(),
    }
    torch.save(state, out_path)


def load_kokoro_checkpoint(model: KModel, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("bert", "bert_encoder", "predictor", "text_encoder", "decoder"):
        if key in state:
            getattr(model, key).load_state_dict(state[key], strict=False)


def save_training_state(
    model: KModel,
    optimizer: torch.optim.Optimizer,
    out_path: Path,
    epoch: int,
    best_val: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val": float(best_val),
            "net": {
                "bert": model.bert.state_dict(),
                "bert_encoder": model.bert_encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "text_encoder": model.text_encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
            },
            "optimizer": optimizer.state_dict(),
        },
        out_path,
    )


def maybe_write_sample(
    model: KModel,
    g2p,
    ref_s: torch.Tensor,
    sample_text: str,
    sample_speed: float,
    out_wav: Path,
    device: torch.device,
    style_map: dict[str, torch.Tensor] | None = None,
    sample_filename: str | None = None,
) -> bool:
    phonemes = text_to_phonemes(g2p, sample_text)
    input_ids = _tokenize(phonemes, model.vocab).to(device)
    if input_ids.shape[1] <= 2:
        logger.warning("Skipping sample synthesis: no valid tokens in sample text.")
        return False
    if input_ids.shape[1] > model.context_length:
        logger.warning(
            f"Skipping sample synthesis: token length {input_ids.shape[1]} > context {model.context_length}."
        )
        return False

    sample_style = ref_s
    if style_map:
        if sample_filename and sample_filename in style_map:
            sample_style = style_map[sample_filename].to(device)
        else:
            sample_style = (
                torch.stack([v.squeeze(0) for v in style_map.values()], dim=0)
                .mean(dim=0, keepdim=True)
                .to(device)
            )

    model.eval()
    with torch.no_grad():
        audio = synth_train_forward(model, input_ids, sample_style, speed=sample_speed).detach().cpu().numpy()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, SAMPLE_RATE)
    return True


def epoch_stage_name(epoch: int, stage1_epochs: int) -> str:
    return "stage1" if epoch <= stage1_epochs else "stage2"


def maybe_build_style_cache(
    rows: list[dict],
    model: KModel,
    g2p,
    segments_dir: Path,
    cache_path: Path,
    cache_device: str,
    cache_steps: int,
    cache_lr: float,
) -> dict[str, torch.Tensor]:
    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        styles = obj["styles"] if isinstance(obj, dict) and "styles" in obj else obj
        logger.info(f"Loaded style cache from {cache_path} ({len(styles)} entries)")
        return styles

    logger.info(
        f"Building stage2 style cache ({len(rows)} utterances, steps={cache_steps}, "
        f"lr={cache_lr}, device={cache_device})"
    )
    styles: dict[str, torch.Tensor] = {}
    for i, row in enumerate(rows, start=1):
        phonemes = text_to_phonemes(g2p, row["text"])
        input_ids = _tokenize(phonemes, model.vocab)
        if input_ids.shape[1] <= 2 or input_ids.shape[1] > model.context_length:
            logger.warning(f"[{i}/{len(rows)}] style cache skip {row['filename']} (token issue)")
            continue
        wav_path = segments_dir / row["filename"]
        logger.info(f"[{i}/{len(rows)}] style cache invert {row['filename']}")
        s, loss = invert_style_vector(
            reference_audio_path=str(wav_path),
            model=model,
            phonemes=phonemes,
            n_steps=cache_steps,
            lr=cache_lr,
            device=cache_device,
            return_loss=True,
        )
        styles[row["filename"]] = s.detach().cpu()
        logger.info(f"  style cache loss={loss:.5f}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"styles": styles}, cache_path)
    logger.info(f"Saved style cache -> {cache_path} ({len(styles)} entries)")
    return styles


def generate_final_voicepack(
    out_path: Path,
    mode: str,
    rows: list[dict],
    model: KModel,
    g2p,
    ref_s: torch.Tensor,
    style_map: dict[str, torch.Tensor] | None,
    segments_dir: Path,
    invert_steps: int,
    invert_lr: float,
    invert_device: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "cache_mean":
        if not style_map:
            raise ValueError("final voicepack mode=cache_mean requires non-empty stage2 style cache.")
        s = torch.stack([v.squeeze(0) for v in style_map.values()], dim=0).mean(dim=0, keepdim=True)
    elif mode == "invert_longest":
        row = max(rows, key=lambda r: r["duration"])
        phonemes = text_to_phonemes(g2p, row["text"])
        s, _ = invert_style_vector(
            reference_audio_path=str(segments_dir / row["filename"]),
            model=model,
            phonemes=phonemes,
            n_steps=invert_steps,
            lr=invert_lr,
            device=invert_device,
            return_loss=True,
        )
        s = s.detach().cpu()
    elif mode == "ref_copy":
        s = ref_s.detach().cpu()
    else:
        raise ValueError(f"Unknown final voicepack mode: {mode}")

    pack = build_voicepack(s)
    torch.save(pack, out_path)
    logger.info(f"Saved final voicepack -> {out_path} shape={tuple(pack.shape)}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Kokoro on single-speaker paired data")
    parser.add_argument("--manifest", required=True, help="TSV: filename/start_sec/end_sec/text")
    parser.add_argument("--segments-dir", required=True)
    parser.add_argument("--voicepack", required=True, help="Reference voicepack .pt (used as fixed speaker style)")
    parser.add_argument("--out-model", required=True, help="Output fine-tuned kokoro .pth")
    parser.add_argument("--model", default="weights/kokoro-v1_0.pth", help="Base Kokoro weights")
    parser.add_argument("--config", default=None, help="Optional config.json path")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5, help="Stage1 main LR")
    parser.add_argument("--bert-lr", type=float, default=5e-6, help="Stage1 BERT LR")
    parser.add_argument("--stage1-epochs", type=int, default=4, help="Epochs in stage1 (fixed style)")
    parser.add_argument(
        "--stage2-train-modules",
        default="decoder,predictor,text_encoder,bert_encoder,bert",
        help="Stage2 modules from: bert,bert_encoder,text_encoder,predictor,decoder",
    )
    parser.add_argument("--stage2-lr", type=float, default=1e-5, help="Stage2 main LR")
    parser.add_argument("--stage2-bert-lr", type=float, default=2e-6, help="Stage2 BERT LR")
    parser.add_argument("--stage2-style-mode", choices=["fixed", "cache"], default="cache")
    parser.add_argument("--stage2-cache-path", default=None, help="Per-utterance style cache path (.pt)")
    parser.add_argument("--stage2-cache-steps", type=int, default=60, help="Inversion steps per utterance for style cache")
    parser.add_argument("--stage2-cache-lr", type=float, default=0.02, help="Inversion LR for style cache")
    parser.add_argument("--stage2-cache-device", default="cpu", help="Device for building style cache")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu/cuda/mps")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size for training/validation")
    parser.add_argument(
        "--train-modules",
        default="decoder,predictor,text_encoder,bert_encoder",
        help="Comma-separated modules from: bert,bert_encoder,text_encoder,predictor,decoder",
    )
    parser.add_argument("--lambda-stft", type=float, default=1.0)
    parser.add_argument("--lambda-melmean", type=float, default=0.2)
    parser.add_argument("--resume", default=None, help="Path to training state checkpoint (.train_state.pt) or model checkpoint (.pth)")
    parser.add_argument("--sample-every", type=int, default=1, help="Write synthesis sample every N epochs (0 disables)")
    parser.add_argument("--sample-text", default=None, help="Text used for periodic synthesis samples")
    parser.add_argument("--sample-speed", type=float, default=1.0, help="Speed used for periodic synthesis samples")
    parser.add_argument("--log-every", type=int, default=100, help="Per-epoch step logging interval (0 disables)")
    parser.add_argument("--trace-every-step", action="store_true", help="Verbose per-step phase tracing (very noisy)")
    parser.add_argument("--hang-threshold-sec", type=float, default=20.0, help="Warn if any phase exceeds this duration")
    parser.add_argument("--log-timings", action="store_true", help="Print averaged timing logs at --log-every interval")
    parser.add_argument("--final-voicepack", default=None, help="Optional output .pt voicepack written after training")
    parser.add_argument(
        "--final-voicepack-mode",
        choices=["cache_mean", "invert_longest", "ref_copy"],
        default="cache_mean",
        help="How to build final voicepack",
    )
    parser.add_argument("--final-invert-steps", type=int, default=120, help="Used when final-voicepack-mode=invert_longest")
    parser.add_argument("--final-invert-lr", type=float, default=0.02, help="Used when final-voicepack-mode=invert_longest")
    parser.add_argument("--final-invert-device", default="cpu", help="Used when final-voicepack-mode=invert_longest")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_str = args.device or auto_device()
    device = torch.device(device_str)
    logger.info(f"Device: {device_str}")

    rows = read_manifest(Path(args.manifest))
    train_rows, val_rows = split_train_val(rows, args.val_ratio, args.seed)
    logger.info(f"Loaded {len(rows)} rows -> train={len(train_rows)} val={len(val_rows)}")

    model = KModel(repo_id="hexgrad/Kokoro-82M", config=args.config, model=args.model).to(device)
    g2p = build_g2p()
    rows = pretokenize_rows(rows, g2p, model)
    train_rows = pretokenize_rows(train_rows, g2p, model)
    val_rows = pretokenize_rows(val_rows, g2p, model)
    logger.info(
        f"After tokenization filter -> all={len(rows)} train={len(train_rows)} val={len(val_rows)} "
        f"(batch_size={args.batch_size})"
    )

    ref_s = load_style_from_voicepack(Path(args.voicepack)).to(device)
    stage1_modules = {m.strip() for m in args.train_modules.split(",") if m.strip()}
    stage2_modules = {m.strip() for m in args.stage2_train_modules.split(",") if m.strip()}

    to_mel_cpu = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=2048, win_length=1200, hop_length=300, n_mels=80
    )
    stft_loss = MultiResolutionSTFTLoss().to(device)

    best_val = float("inf")
    start_epoch = 1
    out_model = Path(args.out_model)
    out_dir = out_model.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    train_state_path = out_dir / f"{out_model.stem}.train_state.pt"
    sample_dir = out_dir / "samples"
    stage2_cache_path = (
        Path(args.stage2_cache_path)
        if args.stage2_cache_path
        else (out_dir / f"{out_model.stem}.stage2_style_cache.pt")
    )
    stage2_style_map: dict[str, torch.Tensor] | None = None

    resume_obj = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume file not found: {resume_path}")
        resume_obj = torch.load(resume_path, map_location="cpu", weights_only=False)
        if isinstance(resume_obj, dict) and "net" in resume_obj:
            for key in ("bert", "bert_encoder", "predictor", "text_encoder", "decoder"):
                if key in resume_obj["net"]:
                    getattr(model, key).load_state_dict(resume_obj["net"][key], strict=False)
            start_epoch = int(resume_obj.get("epoch", 0)) + 1
            best_val = float(resume_obj.get("best_val", float("inf")))
            logger.info(
                f"Resumed training state from {resume_path} "
                f"(start_epoch={start_epoch}, best_val={best_val:.5f})"
            )
        else:
            load_kokoro_checkpoint(model, resume_path)
            logger.info(f"Loaded model weights from {resume_path}; starting fresh optimizer state.")

    if start_epoch > args.epochs:
        logger.info(
                f"Nothing to run: start_epoch={start_epoch} is greater than --epochs={args.epochs}. "
                "Increase --epochs when resuming."
        )
        return

    current_stage = epoch_stage_name(start_epoch, args.stage1_epochs)
    if current_stage == "stage2" and args.stage2_style_mode == "cache":
        stage2_style_map = maybe_build_style_cache(
            rows=rows,
            model=model,
            g2p=g2p,
            segments_dir=Path(args.segments_dir),
            cache_path=stage2_cache_path,
            cache_device=args.stage2_cache_device,
            cache_steps=args.stage2_cache_steps,
            cache_lr=args.stage2_cache_lr,
        )

    if current_stage == "stage1":
        active_modules = stage1_modules
        active_lr = args.lr
        active_bert_lr = args.bert_lr
    else:
        active_modules = stage2_modules
        active_lr = args.stage2_lr
        active_bert_lr = args.stage2_bert_lr
    set_trainable_modules(model, active_modules)
    logger.info(f"Start stage={current_stage} trainable={sorted(active_modules)}")
    optimizer = build_optimizer(model, active_modules, active_lr, active_bert_lr)
    if (
        resume_obj is not None
        and isinstance(resume_obj, dict)
        and "optimizer" in resume_obj
        and resume_obj.get("stage_name") == current_stage
    ):
        optimizer.load_state_dict(resume_obj["optimizer"])
        logger.info("Loaded optimizer state from resume checkpoint.")

    for epoch in range(start_epoch, args.epochs + 1):
        stage_name = epoch_stage_name(epoch, args.stage1_epochs)
        if stage_name != current_stage:
            current_stage = stage_name
            if current_stage == "stage2":
                if args.stage2_style_mode == "cache":
                    stage2_style_map = maybe_build_style_cache(
                        rows=rows,
                        model=model,
                        g2p=g2p,
                        segments_dir=Path(args.segments_dir),
                        cache_path=stage2_cache_path,
                        cache_device=args.stage2_cache_device,
                        cache_steps=args.stage2_cache_steps,
                        cache_lr=args.stage2_cache_lr,
                    )
                active_modules = stage2_modules
                active_lr = args.stage2_lr
                active_bert_lr = args.stage2_bert_lr
            else:
                active_modules = stage1_modules
                active_lr = args.lr
                active_bert_lr = args.bert_lr
            set_trainable_modules(model, active_modules)
            optimizer = build_optimizer(model, active_modules, active_lr, active_bert_lr)
            logger.info(f"Switched to stage={current_stage} trainable={sorted(active_modules)}")

        random.shuffle(train_rows)
        train_style_map = stage2_style_map if current_stage == "stage2" and args.stage2_style_mode == "cache" else None

        train_loss = run_epoch(
            rows=train_rows,
            model=model,
            optimizer=optimizer,
            segments_dir=Path(args.segments_dir),
            ref_s=ref_s,
            device=device,
            to_mel_cpu=to_mel_cpu,
            stft_loss=stft_loss,
            lambda_stft=args.lambda_stft,
            lambda_melmean=args.lambda_melmean,
            train=True,
            style_map=train_style_map,
            batch_size=args.batch_size,
            log_every=args.log_every,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            stage_name=current_stage,
            trace_every_step=args.trace_every_step,
            hang_threshold_sec=args.hang_threshold_sec,
            log_timings=args.log_timings,
        )

        with torch.no_grad():
            val_loss = run_epoch(
                rows=val_rows,
                model=model,
                optimizer=optimizer,
                segments_dir=Path(args.segments_dir),
                ref_s=ref_s,
                device=device,
                to_mel_cpu=to_mel_cpu,
                stft_loss=stft_loss,
                lambda_stft=args.lambda_stft,
                lambda_melmean=args.lambda_melmean,
                train=False,
                style_map=train_style_map,
                batch_size=args.batch_size,
                log_every=args.log_every,
                epoch_idx=epoch,
                total_epochs=args.epochs,
                stage_name=current_stage,
                trace_every_step=args.trace_every_step,
                hang_threshold_sec=args.hang_threshold_sec,
                log_timings=args.log_timings,
            )

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs}  stage={current_stage}  "
            f"train={train_loss:.5f}  val={val_loss:.5f}"
        )

        epoch_ckpt = out_dir / f"{out_model.stem}.epoch{epoch:03d}.pth"
        save_kokoro_checkpoint(model, epoch_ckpt)
        save_training_state(model, optimizer, train_state_path, epoch, best_val)
        state_obj = torch.load(train_state_path, map_location="cpu", weights_only=False)
        state_obj["stage_name"] = current_stage
        torch.save(state_obj, train_state_path)

        if val_loss < best_val:
            best_val = val_loss
            save_kokoro_checkpoint(model, out_model)
            logger.info(f"  New best -> {out_model}  val={best_val:.5f}")
            save_training_state(model, optimizer, train_state_path, epoch, best_val)
            state_obj = torch.load(train_state_path, map_location="cpu", weights_only=False)
            state_obj["stage_name"] = current_stage
            torch.save(state_obj, train_state_path)

        if args.sample_every > 0 and epoch % args.sample_every == 0:
            if args.sample_text:
                sample_text = args.sample_text
                sample_filename = None
            elif val_rows:
                sample_text = val_rows[0]["text"]
                sample_filename = val_rows[0]["filename"]
            else:
                sample_text = train_rows[0]["text"]
                sample_filename = train_rows[0]["filename"]
            sample_wav = sample_dir / f"{out_model.stem}.epoch{epoch:03d}.wav"
            wrote = maybe_write_sample(
                model=model,
                g2p=g2p,
                ref_s=ref_s,
                sample_text=sample_text,
                sample_speed=args.sample_speed,
                out_wav=sample_wav,
                device=device,
                style_map=train_style_map,
                sample_filename=sample_filename,
            )
            if wrote:
                logger.info(f"  Wrote sample -> {sample_wav}")

    if args.final_voicepack:
        final_vp_path = Path(args.final_voicepack)
        generate_final_voicepack(
            out_path=final_vp_path,
            mode=args.final_voicepack_mode,
            rows=rows,
            model=model,
            g2p=g2p,
            ref_s=ref_s,
            style_map=stage2_style_map,
            segments_dir=Path(args.segments_dir),
            invert_steps=args.final_invert_steps,
            invert_lr=args.final_invert_lr,
            invert_device=args.final_invert_device,
        )

    logger.info(f"Done. Best val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()
