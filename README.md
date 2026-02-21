# mifi

Training-loop repo for a TTS model.

## Docker (GPU)

This repo includes a GPU-ready Docker setup for cloud training.

### Prerequisites

- NVIDIA GPU instance
- NVIDIA driver installed on host
- Docker installed
- NVIDIA Container Toolkit installed (`nvidia-smi` should work on host and in containers)

### Build image

```bash
docker compose build
```

### Run training container

```bash
docker compose run --rm trainer
```

### Run a custom command

```bash
docker compose run --rm trainer python -c "import torch; print(torch.cuda.is_available())"
```

## Notes

- `docker-compose.yml` mounts the repo into `/workspace` for live code edits.
- `shm_size` is set to `16gb` to avoid dataloader shared-memory issues during training.
- The image pre-downloads NLTK tagger data required by `g2p-en` (`averaged_perceptron_tagger_eng`).
- The Docker image also installs NeMo text normalization (`nemo_text_processing`) via the `tn` extra.
- Replace `command: python -m mifi.main` with your actual training entrypoint when ready.
