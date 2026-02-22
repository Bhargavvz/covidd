#!/bin/bash
# ============================================================
# FULL TRAINING PIPELINE — STOIC Data on H200
# Run: tmux new -s train && bash run_full_training.sh
# Estimated: ~4-5 hours total
# ============================================================
set -e

echo "============================================================"
echo "  COVID-19 Lung Recovery — Full Training Pipeline"
echo "  Started at: $(date)"
echo "============================================================"

# ---- STEP 1: Install AWS CLI if missing ----
echo ""
echo "[STEP 1/7] Checking AWS CLI..."
if ! command -v aws &>/dev/null; then
    echo "  Installing AWS CLI..."
    pip install -q awscli
    echo "  ✓ AWS CLI installed"
else
    echo "  ✓ AWS CLI already installed"
fi

# ---- STEP 2: Download STOIC data (50 scans) ----
echo ""
echo "[STEP 2/7] Downloading STOIC dataset..."
mkdir -p datasets/stoic/raw/data/mha
MHA_COUNT=$(find datasets/stoic/raw -name "*.mha" 2>/dev/null | wc -l | tr -d ' ')
if [ "$MHA_COUNT" -lt 10 ]; then
    echo "  Downloading ~50 scans from AWS S3 (this takes ~5-10 min)..."
    # Get list of first 50 files
    aws s3 ls s3://stoic2021-training/data/mha/ --no-sign-request \
        | head -50 \
        | awk '{print $4}' \
        | while read fname; do
            if [ -n "$fname" ]; then
                aws s3 cp "s3://stoic2021-training/data/mha/$fname" \
                    "./datasets/stoic/raw/data/mha/$fname" \
                    --no-sign-request --quiet
                echo "    Downloaded: $fname"
            fi
        done
    MHA_COUNT=$(find datasets/stoic/raw -name "*.mha" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ Downloaded $MHA_COUNT scans"
else
    echo "  ✓ Found $MHA_COUNT MHA files, skipping download"
fi

# ---- STEP 3: Convert MHA → NIfTI ----
echo ""
echo "[STEP 3/7] Converting MHA → NIfTI..."
NIFTI_COUNT=$(find datasets/stoic/processed -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
if [ "$NIFTI_COUNT" -lt 10 ]; then
    python data/download_datasets.py --action organize \
        --dataset stoic \
        --raw-dir ./datasets/stoic/raw/data/mha/
    NIFTI_COUNT=$(find datasets/stoic/processed -name "*.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ Converted $NIFTI_COUNT volumes"
else
    echo "  ✓ Found $NIFTI_COUNT NIfTI files, skipping conversion"
fi

# ---- STEP 4: Generate synthetic longitudinal pairs ----
echo ""
echo "[STEP 4/7] Generating synthetic longitudinal pairs..."
SYNTH_JSON=$(find datasets/ -name "synthetic_pairs.json" 2>/dev/null | head -1)
if [ -z "$SYNTH_JSON" ]; then
    echo "  Creating 3 synthetic pairs per patient (this takes ~30-45 min)..."
    python data/download_datasets.py --action synthetic \
        --raw-dir ./datasets/stoic/processed \
        --num-pairs 3
    echo "  ✓ Synthetic pair generation complete"
else
    PAIR_COUNT=$(python -c "import json; print(len(json.load(open('$SYNTH_JSON'))))" 2>/dev/null || echo "0")
    echo "  ✓ Found $PAIR_COUNT synthetic pairs, skipping generation"
fi

# ---- STEP 5: Set volume size to 128³ for speed ----
echo ""
echo "[STEP 5/7] Configuring for 128³ volume training..."
sed -i 's/volume_size: \[192, 192, 192\]/volume_size: [128, 128, 128]/' configs/default.yaml 2>/dev/null || true
echo "  ✓ Volume size set to [128, 128, 128]"

# ---- STEP 6: Train (100 epochs) ----
echo ""
echo "[STEP 6/7] Starting training (100 epochs, batch_size=8)..."
echo "  Training started at: $(date)"
echo "  Expected duration: ~2-3 hours"
echo ""
python training/train.py \
    --config configs/default.yaml \
    --override configs/h200_optimized.yaml \
    --data-dir ./datasets \
    --epochs 100 \
    --batch-size 8 \
    --no-compile
echo ""
echo "  ✓ Training complete at: $(date)"

# ---- STEP 7: Run inference + analysis ----
echo ""
echo "[STEP 7/7] Running inference and analysis..."

# Find test patients for registration
BASELINE=$(find datasets/synthetic_pairs -name "baseline.nii.gz" | head -1)
FOLLOWUP=$(find datasets/synthetic_pairs -name "followup.nii.gz" | head -1)

if [ -n "$BASELINE" ] && [ -n "$FOLLOWUP" ]; then
    echo "  Registering test pair..."
    python inference/register.py \
        --checkpoint outputs/checkpoints/best_model.pth \
        --moving "$BASELINE" \
        --fixed "$FOLLOWUP" \
        --output-dir ./results/test_pair \
        --volume-size 128 128 128 2>&1 | tail -5

    echo "  Generating visualizations..."
    python inference/visualize.py --results-dir ./results/test_pair/ 2>&1 | tail -3
fi

# Recovery analysis on first 3 patients
echo "  Running recovery analysis..."
for i in 0 1 2; do
    PDIR=$(printf "datasets/stoic/processed/patient_%04d" $i)
    if [ -d "$PDIR" ]; then
        python inference/analyze_recovery.py \
            --checkpoint outputs/checkpoints/best_model.pth \
            --patient-dir "$PDIR" \
            --output-dir "results/recovery/patient_${i}" \
            --volume-size 128 128 128 2>&1 | tail -3
    fi
done
echo "  ✓ Inference and analysis complete"

# ---- SUMMARY ----
echo ""
echo "============================================================"
echo "  FULL PIPELINE COMPLETE!"
echo "  Finished at: $(date)"
echo "============================================================"
echo ""
echo "Results:"
echo "  ├── Model:        outputs/checkpoints/best_model.pth"
echo "  ├── History:      outputs/training_history.json"
echo "  ├── Registration: results/test_pair/"
echo "  ├── Figures:      results/test_pair/figures/"
echo "  └── Recovery:     results/recovery/"
echo ""
echo "Download to local machine:"
echo "  scp -r root@<server>:~/covidd/results/ ./results/"
echo "  scp -r root@<server>:~/covidd/outputs/ ./outputs/"
echo "============================================================"
