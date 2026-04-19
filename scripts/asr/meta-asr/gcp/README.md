# GCP Spot Instance Training Scripts

End-to-end scripts for fine-tuning WhissleAI ASR models on GCP spot instances with persistent storage, Docker isolation, and TensorBoard tracking.

## Prerequisites

- `gcloud` CLI authenticated with access to the `deepvoice-468015` project
- Docker Desktop (for local TensorBoard)
- Set your Whissle username before running any remote commands:

```bash
export GCP_USER=yourname        # your GCP/Whissle username
export GCP_PROJECT=deepvoice-468015  # optional, this is the default
export GCP_ZONE=us-central1-c        # optional, this is the default
```

## Architecture

```
/mnt/training/                    (500GB persistent SSD, survives preemption)
├── models/                       Downloaded .nemo checkpoints
├── data/                         HF datasets as NeMo manifests + WAV
├── experiments/                  Training runs (TB logs, checkpoints, configs)
├── code/                         PromptingNemo repo
└── .hf_cache/                    HuggingFace cache
```

## Quick Start

```bash
# 1. Create persistent disk (one-time)
./create-training-disk.sh

# 2. Launch a spot instance
./launch-experiment.sh --name nemo-train-1 --gpu t4

# 3. SSH in and set up the instance (one-time per instance)
./setup-instance.sh

# 4. Download model and data
./download-model.sh --model WhissleAI/STT-meta-1B
./download-data.sh --dataset WhissleAI/Meta_STT_ZH_AIShell3 --lang MANDARIN

# 5. Run fine-tuning
./run-finetune.sh \
  --model WhissleAI/STT-meta-1B \
  --dataset WhissleAI/Meta_STT_ZH_AIShell3 \
  --lang MANDARIN \
  --mode adapter \
  --name zh-mandarin-adapter-v1

# 6. Monitor
./status.sh
./tensorboard.sh
```

## After Spot Preemption

```bash
# Restart the instance
gcloud compute instances start nemo-train-1 --zone=us-central1-c

# Resume from last checkpoint
./resume-training.sh --name zh-mandarin-adapter-v1
```

## Scripts

| Script | Purpose |
|--------|---------|
| `create-training-disk.sh` | Create persistent SSD (one-time) |
| `launch-experiment.sh` | Create spot instance with GPU + training disk |
| `setup-instance.sh` | Install Docker, NVIDIA toolkit, mount disk, build image |
| `download-model.sh` | Download .nemo model from HuggingFace |
| `download-data.sh` | Download dataset from HuggingFace and prepare NeMo manifests |
| `run-finetune.sh` | Run fine-tuning (generates config, launches Docker) |
| `resume-training.sh` | Resume training after preemption |
| `tensorboard.sh` | Launch TensorBoard (local or remote with SSH tunnel) |
| `sync-tb-logs.sh` | Sync TensorBoard logs across multiple instances |
| `status.sh` | Check GPU, disk, experiments, WER progress |
| `benchmark.sh` | Benchmark a trained model (WER, CER, RTF) |
| `upload-model.sh` | Upload fine-tuned model to HuggingFace |

## Training Modes

**Adapter** (default): Freezes encoder, trains adapter layers + decoder (~25M params). Faster, less GPU memory.

```bash
./run-finetune.sh --mode adapter --adapter-dim 128 ...
```

**Full**: Fine-tunes all parameters. Better WER potential, needs more VRAM.

```bash
./run-finetune.sh --mode full --lr 0.0001 ...
```

## GPU Options

```bash
./launch-experiment.sh --name my-exp --gpu t4       # T4 (16GB, cheapest)
./launch-experiment.sh --name my-exp --gpu v100     # V100 (16GB)
./launch-experiment.sh --name my-exp --gpu l4       # L4 (24GB)
./launch-experiment.sh --name my-exp --gpu a100     # A100 (40GB)
./launch-experiment.sh --name my-exp --gpu a100-80  # A100 (80GB)
./launch-experiment.sh --name my-exp --gpu h100     # H100 (80GB)
```

## Custom Config

Provide your own YAML config instead of auto-generating:

```bash
./run-finetune.sh --name my-exp --config /mnt/training/experiments/my-config.yml
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_USER` | (required) | Your Whissle/GCP username |
| `GCP_PROJECT` | `deepvoice-468015` | GCP project ID |
| `GCP_ZONE` | `us-central1-c` | GCP zone |
| `TRAINING_ROOT` | `/mnt/training` | Root directory on the training instance |
| `TRAINING_DISK` | `nemo-training-disk` | Persistent disk name |
