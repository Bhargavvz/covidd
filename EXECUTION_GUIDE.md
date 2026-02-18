# Full Execution Guide — COVID-19 Lung Recovery Registration Pipeline

> Complete step-by-step instructions for running this project end-to-end on an NVIDIA GPU server (H200/A100/V100).

---

## Overview: What Happens at Each Stage

```
┌──────────────────────────────────────────────────────────────────────┐
│                        EXECUTION PIPELINE                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: ENVIRONMENT SETUP                                          │
│  ├─ SSH into server                                                  │
│  ├─ Create Python virtual environment                                │
│  ├─ Install dependencies (PyTorch, SimpleITK, etc.)                  │
│  └─ Verify GPU (CUDA, memory, BF16 support)                         │
│                                                                      │
│  PHASE 2: DATA PREPARATION                                          │
│  ├─ Download datasets (STOIC / COVID-CT+ / BIMCV)                   │
│  ├─ Convert DICOM/MHA → NIfTI (.nii.gz)                             │
│  ├─ Generate synthetic longitudinal pairs                            │
│  ├─ Preprocess: windowing → resampling → normalization               │
│  └─ Segment lungs automatically                                      │
│                                                                      │
│  PHASE 3: TRAINING                                                   │
│  ├─ Run smoke test (5 epochs, small data)                            │
│  ├─ Full training with H200 optimizations                            │
│  ├─ Monitor via TensorBoard                                          │
│  └─ Best model saved automatically                                   │
│                                                                      │
│  PHASE 4: INFERENCE & ANALYSIS                                       │
│  ├─ Register CT pairs (baseline → follow-up)                         │
│  ├─ Compute Jacobian maps & recovery scores                          │
│  ├─ Longitudinal trajectory analysis                                  │
│  └─ Generate visualizations & figures                                 │
│                                                                      │
│  PHASE 5: RESULTS & EXPORT                                           │
│  ├─ Collect metrics (NCC, SSIM, Dice, TRE)                           │
│  ├─ Generate figures for paper                                       │
│  └─ Download results to local machine                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 1: Environment Setup

### 1.1 — SSH into Your Server

```bash
ssh username@your-server-ip

# If using a specific port or key:
ssh -i ~/.ssh/your_key.pem -p 22 username@your-server-ip
```

### 1.2 — Transfer the Project

```bash
# Option A: SCP from local machine
scp -r "/path/to/papers/new" username@server-ip:~/covid-lung-recovery/

# Option B: Git clone (if pushed to a repo)
git clone https://github.com/your-username/covid-lung-recovery.git
cd covid-lung-recovery
```

### 1.3 — Create Virtual Environment

```bash
cd ~/covid-lung-recovery

# Create environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 1.4 — Install Dependencies

```bash
pip install -r requirements.txt
```

> [!NOTE]
> If your server has a specific CUDA version, install the matching PyTorch first:
> ```bash
> # For CUDA 12.1 (H200 / A100)
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> # Then install the rest
> pip install -r requirements.txt
> ```

### 1.5 — Verify GPU Setup

```bash
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name:        {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'GPU Memory:      {mem:.1f} GB')
    print(f'BF16 Support:    {torch.cuda.is_bf16_supported()}')
    print(f'CUDA Version:    {torch.version.cuda}')
"
```

**Expected output (H200):**
```
PyTorch Version: 2.2.0
CUDA Available:  True
GPU Name:        NVIDIA H200
GPU Memory:      141.1 GB
BF16 Support:    True
CUDA Version:    12.1
```

### 1.6 — Run Unit Tests

```bash
python -m pytest tests/ -v
```

**Expected: 28/28 passed ✅**

---

## PHASE 2: Data Preparation

### 2.1 — Option A: Quick Start with Demo Data (Recommended First)

```bash
# Generate 20 small synthetic volumes (64³) for testing
python data/download_datasets.py --action demo --num-demo 20 --demo-size 64
```

This creates `datasets/demo/processed/` with synthetic NIfTI volumes.

### 2.2 — Option B: Generate Synthetic Longitudinal Pairs

```bash
# Generate 100 synthetic pairs from existing volumes
python data/download_datasets.py --action synthetic \
    --output-dir ./datasets/demo/processed \
    --num-pairs 100
```

This creates pairs in `datasets/demo/processed/synthetic_pairs/` with known ground-truth deformations.

### 2.3 — Option C: Download Real Datasets

#### STOIC Dataset (Primary)
```bash
# Download STOIC (requires registration at https://stoic2021.grand-challenge.org/)
python data/download_datasets.py --action download --dataset stoic --data-dir ./datasets/stoic
```

#### BIMCV COVID-19+
```bash
# Download from https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/
python data/download_datasets.py --action download --dataset bimcv --data-dir ./datasets/bimcv
```

### 2.4 — Verify Data

```bash
# Check what we have
find datasets/ -name "*.nii.gz" | wc -l      # Count volumes
find datasets/ -name "*.nii.gz" | head -10     # Show first 10

# Verify a volume loads correctly
python -c "
from data.preprocessing import CTPreprocessor
p = CTPreprocessor(target_size=(192, 192, 192))
vol = p.process('datasets/demo/processed/volume_0000.nii.gz')
print(f'Volume shape: {vol.shape}')
print(f'Value range:  [{vol.min():.3f}, {vol.max():.3f}]')
"
```

---

## PHASE 3: Training

### 3.1 — Smoke Test (DO THIS FIRST)

```bash
# Quick 5-epoch test with tiny data to verify everything works
python training/train.py \
    --config configs/default.yaml \
    --smoke-test
```

**Expected output:**
```
SMOKE TEST MODE: Using synthetic demo data
Generating demo data...
Train: 8 pairs, 4 batches
Val:   1 pairs, 1 batches
Starting Training
  Model:     VoxelMorphDiff
  Epochs:    5
  Batch:     2
  AMP:       torch.bfloat16
...
Epoch 1/5 (3.2s) | Train: loss=0.923 sim=0.891 | Val: loss=0.910 sim=0.880
...
Training complete!
```

> [!IMPORTANT]
> If smoke test works, proceed to full training below. If it fails, check the error and fix before scaling up.

### 3.2 — Full Training (H200 Optimized)

```bash
# Launch full training with H200 optimizations
python training/train.py \
    --config configs/default.yaml \
    --override configs/h200_optimized.yaml \
    --data-dir ./datasets \
    --output-dir ./outputs \
    --epochs 200
```

**H200 optimizations enabled automatically:**
- BF16 mixed precision (Hopper Tensor Cores)
- `torch.compile()` with reduce-overhead mode
- Batch size 8 (×4 accumulation = 32 effective)
- 16 data workers with prefetch_factor=4
- cuDNN benchmark + TF32 matmul

### 3.3 — Run in Background (Survive SSH Disconnect)

```bash
# Method 1: tmux (RECOMMENDED)
tmux new -s training
python training/train.py \
    --config configs/default.yaml \
    --override configs/h200_optimized.yaml \
    --data-dir ./datasets
# Press Ctrl+B, then D to detach
# Reconnect: tmux attach -t training

# Method 2: nohup
nohup python training/train.py \
    --config configs/default.yaml \
    --override configs/h200_optimized.yaml \
    --data-dir ./datasets \
    > training.log 2>&1 &

echo $!  # Note the process ID
tail -f training.log  # Watch progress
```

### 3.4 — Monitor Training with TensorBoard

```bash
# On the server:
tensorboard --logdir ./logs --port 6006 --bind_all &

# On your LOCAL machine (port forwarding):
ssh -L 6006:localhost:6006 username@server-ip

# Open in browser: http://localhost:6006
```

**TensorBoard shows:**
- Training/validation loss curves
- Similarity loss (1-NCC)
- Negative Jacobian % (topology violations)
- Learning rate schedule

### 3.5 — Resume from Checkpoint (If Training Interrupted)

```bash
python training/train.py \
    --config configs/default.yaml \
    --override configs/h200_optimized.yaml \
    --resume outputs/checkpoints/checkpoint_epoch_0050.pth
```

### 3.6 — Training Outputs

After training, you'll have:
```
outputs/
├── checkpoints/
│   ├── best_model.pth              ← Best model (lowest validation loss)
│   ├── final_model.pth             ← Last epoch model
│   └── checkpoint_epoch_XXXX.pth   ← Periodic checkpoints
├── training_history.json           ← Loss curves data
logs/
└── events.out.tfevents.*           ← TensorBoard logs
```

---

## PHASE 4: Inference & Analysis

### 4.1 — Register a Single CT Pair

```bash
python inference/register.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --moving path/to/baseline_scan.nii.gz \
    --fixed path/to/followup_scan.nii.gz \
    --output-dir ./results/pair_001
```

**Outputs:**
```
results/pair_001/
├── warped.nii.gz                   ← Registered (warped) moving image
├── displacement.nii.gz             ← Dense displacement field
├── jacobian_det.nii.gz             ← Jacobian determinant map
├── moving_preprocessed.nii.gz      ← Input moving (preprocessed)
├── fixed_preprocessed.nii.gz       ← Input fixed (preprocessed)
└── metrics.json                    ← Registration quality metrics
```

### 4.2 — Longitudinal Recovery Analysis (Multi-Timepoint)

```bash
# Organize patient scans in a directory:
# datasets/patient_001/
# ├── baseline.nii.gz
# ├── 3months.nii.gz
# ├── 6months.nii.gz
# └── 12months.nii.gz

python inference/analyze_recovery.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --patient-dir datasets/patient_001/ \
    --output-dir results/recovery/patient_001 \
    --timepoint-labels "Baseline" "3 Months" "6 Months" "12 Months"
```

**Outputs:**
```
results/recovery/patient_001/
├── recovery_report.json            ← Full recovery metrics
└── recovery_trajectory.png         ← Recovery score over time plot
```

**Console output example:**
```
RECOVERY ANALYSIS SUMMARY
============================================================
  Baseline:  score=0.0000 | baseline
  3 Months:  score=0.4523 | persistent_abnormality
  6 Months:  score=0.6891 | partial_recovery
  12 Months: score=0.8712 | complete_recovery

  Trend: improving
  Rate:  0.1837 per timepoint
  Est. complete recovery: timepoint 3.8
============================================================
```

### 4.3 — Generate Visualizations

```bash
python inference/visualize.py \
    --results-dir results/pair_001/ \
    --output-dir results/pair_001/figures/
```

**Generated figures:**
```
results/pair_001/figures/
├── registration_comparison.png     ← Fixed | Moving | Warped | Difference
├── deformation_grid.png            ← Deformed grid overlay
├── jacobian_map.png                ← Color-coded Jacobian + histogram
├── displacement_magnitude.png      ← Displacement magnitude in mm
└── training_curves.png             ← Loss/LR curves (if history available)
```

---

## PHASE 5: Results & Export

### 5.1 — Batch Processing Multiple Patients

```bash
# Process all patients in a loop
for patient_dir in datasets/patients/*/; do
    patient_name=$(basename "$patient_dir")
    echo "Processing $patient_name..."
    
    python inference/analyze_recovery.py \
        --checkpoint outputs/checkpoints/best_model.pth \
        --patient-dir "$patient_dir" \
        --output-dir "results/recovery/$patient_name"
done
```

### 5.2 — Collect All Metrics

```bash
# View all metrics
python -c "
import json, glob
for f in sorted(glob.glob('results/recovery/*/recovery_report.json')):
    with open(f) as fh:
        data = json.load(fh)
    patient = f.split('/')[-2]
    final = data['timepoints'][-1]
    print(f'{patient}: score={final[\"recovery_score\"]:.4f} | {final[\"status\"]}')
"
```

### 5.3 — Download Results to Local Machine

```bash
# From your LOCAL machine:
scp -r username@server-ip:~/covid-lung-recovery/results/ ./local_results/
scp -r username@server-ip:~/covid-lung-recovery/outputs/ ./local_outputs/

# Or just the figures and reports:
scp -r username@server-ip:~/covid-lung-recovery/results/*/figures/ ./paper_figures/
scp username@server-ip:~/covid-lung-recovery/outputs/training_history.json ./
```

---

## Quick Reference: Essential Commands

| Task | Command |
|------|---------|
| **Run tests** | `python -m pytest tests/ -v` |
| **Generate demo data** | `python data/download_datasets.py --action demo --num-demo 20` |
| **Smoke test** | `python training/train.py --config configs/default.yaml --smoke-test` |
| **Full H200 train** | `python training/train.py --config configs/default.yaml --override configs/h200_optimized.yaml` |
| **Resume training** | `python training/train.py --config configs/default.yaml --resume outputs/checkpoints/checkpoint_epoch_0050.pth` |
| **Register pair** | `python inference/register.py --checkpoint outputs/checkpoints/best_model.pth --moving a.nii.gz --fixed b.nii.gz` |
| **Recovery analysis** | `python inference/analyze_recovery.py --checkpoint outputs/checkpoints/best_model.pth --patient-dir datasets/patient_001/` |
| **Visualize** | `python inference/visualize.py --results-dir results/pair_001/` |
| **TensorBoard** | `tensorboard --logdir ./logs --port 6006` |
| **Check GPU** | `nvidia-smi` or `watch -n1 nvidia-smi` |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `batch_size` in config or use `--batch-size 2` |
| `torch.compile fails` | Use `--no-compile` flag, or upgrade PyTorch to ≥2.1 |
| `NaN loss during training` | Reduce learning rate (`--lr 5e-5`), increase `smooth_weight` in config |
| `No .nii.gz files found` | Check `--data-dir` path, ensure data preparation ran successfully |
| `ModuleNotFoundError` | Run from project root directory, ensure `venv` is activated |
| `SSH disconnects kill training` | Use `tmux` or `nohup` (see Phase 3.3) |
| `Negative Jacobian % too high` | Increase `jac_weight` in loss config (e.g., 0.5 → 1.0) |
