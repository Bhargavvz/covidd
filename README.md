# AI-Based Longitudinal Analysis of Post-COVID-19 Lung Recovery Using Deformable CT Image Registration

A deep learning pipeline for tracking post-COVID-19 lung recovery over time using **deformable image registration** on chest CT scans. The system registers follow-up CT scans to baseline scans using a **VoxelMorph** architecture, produces dense displacement fields, and quantifies tissue recovery through **Jacobian analysis**, **density tracking**, and **structural similarity scoring**.

**Optimized for NVIDIA H200 GPU** (141 GB HBM3e, BF16 Tensor Cores, Hopper Architecture).

---

## 🏗️ Architecture

```
Input: (Moving CT, Fixed CT) → U-Net → Velocity Field → Scaling & Squaring → Displacement Field → Spatial Transformer → Warped CT
                                                                                   ↓
                                                                         Jacobian Analysis → Recovery Scoring
```

### Key Components:
| Component | Description |
|-----------|-------------|
| **3D U-Net Backbone** | Encoder-decoder with skip connections for displacement/velocity field prediction |
| **Spatial Transformer** | Differentiable 3D warping with trilinear interpolation |
| **Diffeomorphic Integration** | Scaling-and-squaring for topology-preserving transforms |
| **Recovery Analyzer** | Jacobian-based volume change scoring with regional analysis |

---

## 📂 Project Structure

```
├── configs/
│   ├── default.yaml              # Default hyperparameters
│   └── h200_optimized.yaml       # H200 GPU-specific overrides
├── data/
│   ├── download_datasets.py      # Dataset download & preparation
│   ├── preprocessing.py          # CT windowing, resampling, normalization
│   ├── lung_segmentation.py      # Automated lung ROI extraction
│   ├── dataset.py                # PyTorch Dataset + DataLoaders
│   └── augmentation.py           # 3D spatial + intensity augmentation
├── models/
│   ├── unet3d.py                 # 3D U-Net encoder-decoder
│   ├── spatial_transformer.py    # STN + diffeomorphic integration
│   ├── voxelmorph.py             # VoxelMorph & VoxelMorph-Diff models
│   ├── losses.py                 # NCC, SSIM, bending energy, Jacobian
│   └── recovery_analyzer.py      # Longitudinal recovery scoring
├── training/
│   ├── train.py                  # Main training script
│   ├── trainer.py                # Training loop (H200 optimized)
│   └── lr_scheduler.py           # Warmup + cosine annealing
├── inference/
│   ├── register.py               # Registration inference
│   ├── analyze_recovery.py       # Longitudinal analysis
│   └── visualize.py              # Visualization suite
├── utils/
│   ├── metrics.py                # Dice, TRE, SSIM, Jacobian stats
│   ├── io_utils.py               # File I/O helpers
│   └── logging_utils.py          # TensorBoard + W&B
├── tests/
│   ├── test_model.py             # Model forward pass tests
│   ├── test_losses.py            # Loss function tests
│   └── test_data.py              # Data pipeline tests
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

| Dataset | Size | Source | Use |
|---------|------|--------|-----|
| **STOIC** | 2,000 CT scans | [grand-challenge.org](https://stoic2021.grand-challenge.org/) | Primary training |
| **COVID-CT+** | 400K+ images | [NIH](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8411519/) | Supplementary |
| **BIMCV COVID-19+** | Large annotated | [bimcv.cipf.es](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/) | Validation |

### Synthetic Longitudinal Pair Generation
Since true public longitudinal COVID-CT datasets are scarce, the pipeline includes synthetic pair generation that simulates recovery by applying controlled deformations and density changes with known ground-truth displacement fields.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Demo Data (for testing)
```bash
python data/download_datasets.py --action demo --num-demo 20 --demo-size 64
```

### 3. Generate Synthetic Pairs
```bash
python data/download_datasets.py --action synthetic --num-pairs 5
```

### 4. Smoke Test (5 epochs on demo data)
```bash
python training/train.py --config configs/default.yaml --smoke-test
```

### 5. Full Training (H200 Optimized)
```bash
python training/train.py \
    --config configs/default.yaml \
    --override configs/h200_optimized.yaml \
    --data-dir ./datasets
```

### 6. Inference
```bash
# Register a pair
python inference/register.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --moving path/to/baseline.nii.gz \
    --fixed path/to/followup.nii.gz

# Longitudinal analysis
python inference/analyze_recovery.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --patient-dir datasets/patient_001/ \
    --timepoint-labels "Baseline" "3 Months" "6 Months" "12 Months"

# Generate visualizations
python inference/visualize.py --results-dir results/
```

---

## ⚡ H200 GPU Optimizations

| Feature | Setting | Benefit |
|---------|---------|---------|
| **BF16 Mixed Precision** | Native Hopper BF16 Tensor Cores | 2× throughput vs FP32 |
| **TF32 Matmul** | `torch.set_float32_matmul_precision('high')` | Faster FP32 operations |
| **`torch.compile()`** | Graph-mode optimization | Kernel fusion for Hopper |
| **cuDNN Benchmark** | `cudnn.benchmark = True` | Optimal conv algorithms |
| **Large Batch Size** | 8 (×4 accumulation = 32 effective) | Leverages 141 GB HBM3e |
| **Multi-worker DataLoader** | 16 workers, prefetch_factor=4 | Saturates 4.8 TB/s bandwidth |
| **Gradient Accumulation** | 4 steps | Effective batch size 32 |

---

## 📈 Loss Functions

**Total Loss = λ_sim × L_sim + λ_smooth × L_smooth + λ_jac × L_jac**

| Loss | Purpose | Default Weight |
|------|---------|----------------|
| **NCC** (Normalized Cross-Correlation) | Image similarity | 1.0 |
| **Bending Energy** | Deformation smoothness | 3.0 |
| **Jacobian Determinant** | Topology preservation | 0.1 |
| **Dice** (optional) | Segmentation alignment | 0.5 |

---

## 🔬 Recovery Analysis

The recovery analyzer quantifies lung recovery by computing:

1. **Jacobian Determinant Maps**: Local volume change at each voxel
   - det(J) > 1 → expansion
   - det(J) = 1 → no change  
   - 0 < det(J) < 1 → contraction
   - det(J) ≤ 0 → topology folding (penalized)

2. **Recovery Score**: Fraction of lung voxels with normal Jacobian (0.8–1.2)

3. **Recovery Classification**:
   - **Complete Recovery** (score ≥ 0.85)
   - **Partial Recovery** (0.50 ≤ score < 0.85)
   - **Persistent Abnormality** (score < 0.50)

4. **Trajectory Analysis**: Linear trend fitting across timepoints with recovery time estimation

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Individual test suites
python -m pytest tests/test_model.py -v
python -m pytest tests/test_losses.py -v
python -m pytest tests/test_data.py -v
```

---

## 📄 License

This project is for academic research purposes. Datasets are subject to their respective licenses (CC BY-NC 4.0).

## 📚 References

1. Balakrishnan et al., "VoxelMorph: A Learning Framework for Deformable Medical Image Registration", IEEE TMI, 2019
2. Dalca et al., "Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces", MedIA, 2019
3. STOIC (Study of Thoracic CT in COVID-19), grand-challenge.org
