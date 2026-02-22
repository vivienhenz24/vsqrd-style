"""
Style vector inversion for Kokoro-compatible voicepack creation.

Given reference audio, finds a style vector s ∈ R^256 such that
Kokoro's decoder produces audio matching the reference speaker's timbre.

Strategy:
  1. Fix the alignment (duration/attention) from an initial forward pass.
  2. Optimize s directly through the differentiable decoder + F0Ntrain.
  3. Loss: single-scale per-band mel mean MSE (content-independent timbre).
  4. One mid-point alignment re-fix + a refinement phase with re-fixed alignment.

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
_MEL_PARAMS = dict(
    sample_rate=SAMPLE_RATE, n_fft=2048, win_length=1200, hop_length=300, n_mels=80
)

_DEFAULT_PHONEMES = "hɛloʊ, ðɪs ɪz ə vɔɪs tɛst fɔːɹ vɔɪs klonɪŋ."


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_audio(path: str) -> torch.Tensor:
    """Load audio, resample to 24 kHz, mono. Returns [samples]."""
    wave, sr = sf.read(path, always_2d=True)
    wave = wave.mean(axis=1)
    wave = torch.from_numpy(wave.astype(np.float32))
    if sr != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    return wave


def _mel_band_means(audio: torch.Tensor, to_mel_cpu) -> torch.Tensor:
    """
    Per-band log-mel mean — content-independent timbre signature. Returns [80].

    Always computed on CPU: MPS STFT has gradient instabilities that prevent
    the loss from converging. The device transfer preserves autograd, so
    gradients still flow back through the MPS decoder to s.
    """
    mel = torch.log(to_mel_cpu(audio.cpu()).clamp(min=1e-5))
    return mel.mean(dim=-1)


def _tokenize(phonemes: str, vocab: dict) -> torch.LongTensor:
    """Map IPA chars → token IDs, wrapped with BOS/EOS (0)."""
    ids = [vocab[c] for c in phonemes if c in vocab]
    return torch.LongTensor([[0, *ids, 0]])


def _get_alignment(model, input_ids: torch.LongTensor, s_detached: torch.Tensor):
    """
    Run one Kokoro forward pass to compute the hard duration alignment.
    Returns (en, asr) — both detached from the graph.
    """
    device = model.device
    input_lengths = torch.full(
        (input_ids.shape[0],), input_ids.shape[-1],
        device=device, dtype=torch.long,
    )
    text_mask = (
        torch.arange(input_lengths.max())
        .unsqueeze(0)
        .expand(input_lengths.shape[0], -1)
        .type_as(input_lengths)
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)
    s_prosody = s_detached[:, 128:]

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
    Synthesize audio with FIXED alignment and LEARNABLE s.
    Gradients flow through F0Ntrain and the decoder w.r.t. s.
    """
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s[:, 128:])
    return model.decoder(asr, F0_pred, N_pred, s[:, :128]).squeeze()


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
    return_loss: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """
    Find a style vector [1, 256] compatible with Kokoro's decoder that
    reproduces the timbre of the reference speaker.

    Optimization runs in two phases:
      Phase 1 (n_steps):    cosine-annealed lr, alignment re-fixed at midpoint.
      Phase 2 (n_steps//2): lr/2, alignment re-fixed from best phase-1 s.

    Args:
        reference_audio_path: Path to reference speaker audio.
        model: Loaded KModel (weights will be frozen).
        phonemes: IPA phoneme string used for synthesis during optimization.
        n_steps: Phase-1 gradient descent steps (phase 2 is n_steps//2).
        lr: Peak learning rate.
        warmstart_voicepack: Optional path to existing .pt voicepack for warm start.
        device: Device to run on. Auto-detects mps/cuda/cpu if None.

    Returns:
        Optimized style vector, shape [1, 256].
    """
    if device is None:
        device = _auto_device()
    logger.info(f"Device: {device}")

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Mel transform always on CPU — avoids MPS STFT gradient instabilities.
    # The model (synthesis) stays on MPS/GPU; only the loss moves to CPU.
    to_mel_cpu = torchaudio.transforms.MelSpectrogram(**_MEL_PARAMS)

    # Reference timbre signature (CPU)
    ref_wave = _load_audio(reference_audio_path)
    with torch.no_grad():
        ref_mel_means = _mel_band_means(ref_wave, to_mel_cpu).detach()
    logger.info(f"Reference audio: {ref_wave.shape[0]/SAMPLE_RATE:.2f}s")

    # Tokenize synthesis phonemes
    input_ids = _tokenize(phonemes, model.vocab).to(dev)
    if input_ids.shape[1] <= 2:
        raise ValueError(
            "No recognised phoneme tokens in the provided phoneme string. "
            "Check that the IPA characters are in Kokoro's vocab."
        )
    logger.info(f"Synthesis phonemes: {input_ids.shape[1]-2} tokens")

    # Initialise style vector
    warm = _init_s_from_voicepack(warmstart_voicepack)
    if warm is not None:
        s = warm.to(dev).requires_grad_(True)
        logger.info("Warm-starting from existing voicepack entry")
    else:
        s = torch.zeros(1, 256, device=dev, requires_grad=True)

    best_loss = float("inf")
    best_s = s.detach().clone()

    def _run_phase(s_init, steps, peak_lr, phase_label):
        nonlocal best_loss, best_s
        s_var = s_init.clone().requires_grad_(True)
        opt = torch.optim.Adam([s_var], lr=peak_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        # Fix alignment once at the start of this phase from s_init
        en, asr = _get_alignment(model, input_ids, s_var.detach())

        for step in range(steps):
            opt.zero_grad()
            audio = _synth_with_fixed_alignment(model, en, asr, s_var)
            loss = F.mse_loss(_mel_band_means(audio, to_mel_cpu), ref_mel_means)
            loss.backward()
            opt.step()
            sched.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_s = s_var.detach().clone()

            if (step + 1) % 50 == 0:
                logger.info(
                    f"  [{phase_label}] step {step+1:>4}/{steps}  "
                    f"loss={loss.item():.5f}  lr={sched.get_last_lr()[0]:.5f}"
                )

        return best_s.clone()

    # Phase 1
    best_s = _run_phase(s.detach(), n_steps, lr, "phase1")

    # Phase 2: refinement from best phase-1 result, halved lr
    logger.info("Starting refinement phase …")
    best_s = _run_phase(best_s, n_steps // 2, lr * 0.5, "phase2")

    logger.info(f"Optimisation done. Best loss: {best_loss:.5f}")
    if return_loss:
        return best_s, float(best_loss)
    return best_s  # [1, 256]


def build_voicepack(
    style_vector: torch.Tensor,
    n_entries: int = 510,
) -> torch.Tensor:
    """
    Build a [n_entries, 1, 256] voicepack from a single optimised style vector.

    All entries use the same vector — Kokoro indexes by phoneme sequence length
    but the style converges once the audio context is long enough.
    """
    assert style_vector.shape == (1, 256), style_vector.shape
    return style_vector.cpu().unsqueeze(0).expand(n_entries, -1, -1).clone()


def main():
    parser = argparse.ArgumentParser(
        description="Create a Kokoro voicepack by optimising a style vector."
    )
    parser.add_argument("--audio", required=True,
                        help="Reference speaker audio (wav/flac/mp3)")
    parser.add_argument("--output", required=True,
                        help="Output voicepack .pt path")
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
                        help="Device: cpu / cuda / mps (auto-detects if omitted)")
    args = parser.parse_args()

    from kokoro.model import KModel
    import json

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=config,
        model=args.model,
    )
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
