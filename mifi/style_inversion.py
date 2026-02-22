"""
Style vector inversion for Kokoro-compatible voicepack creation.

Given reference audio, finds a style vector s ∈ R^256 such that
Kokoro's decoder produces audio matching the reference speaker's timbre.

Strategy:
  1. Fix the alignment (duration/attention) from an initial forward pass.
  2. Optimize s directly through the differentiable decoder + F0Ntrain.
  3. Loss: per-band mel mean MSE (captures timbre, content-independent).

Usage:
    uv run python -m mifi.style_inversion \\
        --audio path/to/speaker.wav \\
        --output voices/newspeaker.pt \\
        --model weights/kokoro-v1_0.pth
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger

SAMPLE_RATE = 24000
MEL_PARAMS = dict(
    sample_rate=SAMPLE_RATE, n_fft=2048, win_length=1200, hop_length=300, n_mels=80
)

# A short IPA string that covers common English phonemes.
# Used when no phoneme string is provided by the caller.
_DEFAULT_PHONEMES = "hɛloʊ, ðɪs ɪz ə vɔɪs tɛst fɔːɹ vɔɪs klonɪŋ."


def _load_audio(path: str) -> torch.Tensor:
    """Load audio, resample to 24 kHz, mono. Returns [samples]."""
    wave, sr = sf.read(path, always_2d=True)
    wave = wave.mean(axis=1)
    wave = torch.from_numpy(wave.astype(np.float32))
    if sr != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    return wave


def _mel_band_means(audio: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute mean log-mel energy per frequency band.
    Returns [80] tensor — content-independent timbre signature.
    """
    to_mel = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS).to(device)
    mel = to_mel(audio.to(device))
    mel = torch.log(mel.clamp(min=1e-5))
    return mel.mean(dim=-1)  # [80]


def _tokenize(phonemes: str, vocab: dict) -> torch.LongTensor:
    """Map IPA chars → token IDs, wrapped with BOS/EOS (0)."""
    ids = [vocab[c] for c in phonemes if c in vocab]
    return torch.LongTensor([[0, *ids, 0]])


def _get_alignment(model, input_ids: torch.LongTensor, s_init: torch.Tensor):
    """
    Run one Kokoro forward pass to get the hard alignment tensors.
    Returns (en, asr, s_init) — all detached from the graph.

    `en`:  features expanded by alignment, shape [1, hidden, T_frames]
    `asr`: text encoder features expanded,  shape [1, hidden, T_frames]
    """
    device = model.device
    input_lengths = torch.full(
        (input_ids.shape[0],), input_ids.shape[-1],
        device=device, dtype=torch.long
    )
    text_mask = (
        torch.arange(input_lengths.max())
        .unsqueeze(0)
        .expand(input_lengths.shape[0], -1)
        .type_as(input_lengths)
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)

    s_prosody = s_init[:, 128:].detach()

    with torch.no_grad():
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        d = model.predictor.text_encoder(d_en, s_prosody, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = (d.transpose(-1, -2) @ pred_aln_trg).detach()

        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = (t_en @ pred_aln_trg).detach()

    return en, asr


def _synth_with_fixed_alignment(
    model,
    en: torch.Tensor,
    asr: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """
    Synthesize audio with a FIXED alignment (en, asr) and a LEARNABLE s.
    Gradients flow through F0Ntrain and the decoder w.r.t. s.
    """
    s_acoustic = s[:, :128]
    s_prosody = s[:, 128:]
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s_prosody)
    audio = model.decoder(asr, F0_pred, N_pred, s_acoustic).squeeze()
    return audio


@torch.no_grad()
def _init_s_from_voicepack(voicepack_path: str | None) -> torch.Tensor | None:
    """Load the middle entry of an existing voicepack as a warm start."""
    if voicepack_path is None:
        return None
    pack = torch.load(voicepack_path, map_location="cpu", weights_only=True)
    return pack[pack.shape[0] // 2].clone()  # [1, 256]


def invert_style_vector(
    reference_audio_path: str,
    model,
    phonemes: str = _DEFAULT_PHONEMES,
    n_steps: int = 400,
    lr: float = 0.02,
    warmstart_voicepack: str | None = None,
    device: str | None = None,
) -> torch.Tensor:
    """
    Find a style vector [1, 256] compatible with Kokoro's decoder that
    reproduces the timbre of the reference speaker.

    Args:
        reference_audio_path: Path to reference speaker audio.
        model: Loaded KModel (weights will be frozen).
        phonemes: IPA phoneme string used for synthesis during optimization.
        n_steps: Gradient descent steps.
        lr: Learning rate.
        warmstart_voicepack: Optional path to existing .pt voicepack for warm start.
        device: Device override (defaults to model's device).

    Returns:
        Optimized style vector, shape [1, 256].
    """
    if device is None:
        device = str(model.device)

    # Freeze all model parameters
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # Reference timbre signature
    ref_wave = _load_audio(reference_audio_path)
    ref_mel_means = _mel_band_means(ref_wave, torch.device(device))  # [80]
    logger.info(
        f"Reference audio: {ref_wave.shape[0]/SAMPLE_RATE:.2f}s, "
        f"mel shape computed"
    )

    # Tokenize the phoneme string for synthesis
    input_ids = _tokenize(phonemes, model.vocab).to(device)
    if input_ids.shape[1] <= 2:
        raise ValueError(
            "No recognised phoneme tokens in the provided phoneme string. "
            "Check that the IPA characters are in Kokoro's vocab."
        )
    logger.info(f"Synthesis phonemes: {input_ids.shape[1]-2} tokens")

    # Initialise style vector
    warm = _init_s_from_voicepack(warmstart_voicepack)
    if warm is not None:
        s = warm.to(device).requires_grad_(True)
        logger.info("Warm-starting from existing voicepack entry")
    else:
        # Zero init → decoder produces a "neutral" voice; we perturb from there
        s = torch.zeros(1, 256, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([s], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # Fix the alignment once (uses initial s — refined below)
    with torch.no_grad():
        s_tmp = s.detach().clone()
    en, asr = _get_alignment(model, input_ids, s_tmp)

    best_loss = float("inf")
    best_s = s.detach().clone()

    for step in range(n_steps):
        optimizer.zero_grad()

        audio = _synth_with_fixed_alignment(model, en, asr, s)
        synth_mel_means = _mel_band_means(audio, torch.device(device))  # [80]

        loss = F.mse_loss(synth_mel_means, ref_mel_means)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_s = s.detach().clone()

        if (step + 1) % 50 == 0:
            logger.info(f"  step {step+1:>4}/{n_steps}  loss={loss.item():.5f}  lr={scheduler.get_last_lr()[0]:.5f}")

    # Re-fix alignment with the optimised s and run one more refinement pass
    logger.info("Re-fixing alignment with optimised s …")
    en, asr = _get_alignment(model, input_ids, best_s.to(device))

    s2 = best_s.clone().requires_grad_(True)
    optimizer2 = torch.optim.Adam([s2], lr=lr * 0.5)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=n_steps // 2)

    for step in range(n_steps // 2):
        optimizer2.zero_grad()
        audio = _synth_with_fixed_alignment(model, en, asr, s2)
        loss = F.mse_loss(_mel_band_means(audio, torch.device(device)), ref_mel_means)
        loss.backward()
        optimizer2.step()
        scheduler2.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_s = s2.detach().clone()

    logger.info(f"Optimisation done. Best loss: {best_loss:.5f}")
    return best_s  # [1, 256]


def build_voicepack(
    style_vector: torch.Tensor,
    n_entries: int = 510,
) -> torch.Tensor:
    """
    Build a [n_entries, 1, 256] voicepack from a single style vector.

    All entries use the same style vector (Kokoro indexes by phoneme-count
    but the style doesn't meaningfully vary once the audio is long enough).
    """
    assert style_vector.shape == (1, 256), style_vector.shape
    pack = style_vector.unsqueeze(0).expand(n_entries, -1, -1).clone()
    return pack  # [510, 1, 256]


def main():
    parser = argparse.ArgumentParser(
        description="Create a Kokoro voicepack by optimising a style vector."
    )
    parser.add_argument("--audio", required=True, help="Reference speaker audio (wav/flac/mp3)")
    parser.add_argument("--output", required=True, help="Output voicepack .pt path")
    parser.add_argument("--model", default="weights/kokoro-v1_0.pth",
                        help="Kokoro model weights (.pth)")
    parser.add_argument("--config", default=None,
                        help="Kokoro config.json (downloads from HF if omitted)")
    parser.add_argument("--phonemes", default=_DEFAULT_PHONEMES,
                        help="IPA phoneme string used during optimisation")
    parser.add_argument("--steps", type=int, default=400,
                        help="Optimisation steps (default 400)")
    parser.add_argument("--lr", type=float, default=0.02,
                        help="Learning rate (default 0.02)")
    parser.add_argument("--warmstart", default=None,
                        help="Existing voicepack .pt to warm-start from")
    parser.add_argument("--device", default=None,
                        help="Device (cpu / cuda / mps). Defaults to KModel device.")
    args = parser.parse_args()

    from kokoro.model import KModel
    import json

    # Build model
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=config,
        model=args.model,
    )
    if args.device:
        model = model.to(args.device)
    model.eval()

    s = invert_style_vector(
        reference_audio_path=args.audio,
        model=model,
        phonemes=args.phonemes,
        n_steps=args.steps,
        lr=args.lr,
        warmstart_voicepack=args.warmstart,
        device=args.device,
    )

    pack = build_voicepack(s)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, output)
    logger.info(f"Saved voicepack → {output}  shape={pack.shape}")


if __name__ == "__main__":
    main()
