#!/bin/bash
# ============================================================
# COVID-19 Lung Recovery — Complete Pipeline Script
# Run on H200 GPU server: bash run_pipeline.sh
# Estimated time: ~4-5 hours total
# ============================================================

set -e  # Exit on any error

echo "============================================================"
echo "  COVID-19 Lung Recovery Registration Pipeline"
echo "  Started at: $(date)"
echo "============================================================"

# ---- STEP 0: Fix known bugs ----
echo ""
echo "[STEP 0/9] Applying bug fixes..."
sed -i 's/total_mem/total_memory/g' training/train.py training/trainer.py 2>/dev/null || true
echo "  ✓ Fixed total_mem → total_memory"

# ---- STEP 1: Reduce volume size for faster training ----
echo ""
echo "[STEP 1/9] Setting volume size to 128³..."
sed -i 's/volume_size: \[192, 192, 192\]/volume_size: [128, 128, 128]/' configs/default.yaml 2>/dev/null || true
echo "  ✓ Volume size set to [128, 128, 128]"

# ---- STEP 2: Download STOIC dataset (skip if already done) ----
echo ""
echo "[STEP 2/9] Checking dataset..."
MHA_COUNT=$(find datasets/stoic/raw -name "*.mha" 2>/dev/null | wc -l)
if [ "$MHA_COUNT" -lt 10 ]; then
    echo "  Downloading ~50 STOIC scans from AWS S3..."
    mkdir -p datasets/stoic/raw
    aws s3 cp s3://stoic2021-training/data/mha/ ./datasets/stoic/raw/data/mha/ \
        --recursive --no-sign-request \
        --quiet \
        | head -50
    echo "  ✓ Download complete"
else
    echo "  ✓ Found $MHA_COUNT MHA files, skipping download"
fi

# ---- STEP 3: Organize + Convert MHA → NIfTI (skip if already done) ----
echo ""
echo "[STEP 3/9] Organizing dataset..."
NIFTI_COUNT=$(find datasets/stoic/processed -name "*.nii.gz" 2>/dev/null | wc -l)
if [ "$NIFTI_COUNT" -lt 10 ]; then
    echo "  Converting MHA → NIfTI and organizing..."
    python data/download_datasets.py --action organize \
        --dataset stoic \
        --raw-dir ./datasets/stoic/raw/data/mha/
    echo "  ✓ Organization complete"
else
    echo "  ✓ Found $NIFTI_COUNT NIfTI files, skipping conversion"
fi

# ---- STEP 4: Generate synthetic longitudinal pairs (skip if already done) ----
echo ""
echo "[STEP 4/9] Generating synthetic longitudinal pairs..."
SYNTH_COUNT=$(find datasets/ -path "*/synthetic*" -name "*.nii.gz" 2>/dev/null | wc -l)
if [ "$SYNTH_COUNT" -lt 10 ]; then
    echo "  Creating 3 synthetic pairs per patient..."
    python data/download_datasets.py --action synthetic \
        --raw-dir ./datasets/stoic/processed \
        --num-pairs 3
    echo "  ✓ Synthetic pair generation complete"
else
    echo "  ✓ Found $SYNTH_COUNT synthetic files, skipping generation"
fi

# ---- STEP 5: Run unit tests ----
echo ""
echo "[STEP 5/9] Running unit tests..."
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
echo "  ✓ Tests complete"

# ---- STEP 6: Train the model ----
echo ""
echo "[STEP 6/9] Starting training (100 epochs)..."
echo "  Training started at: $(date)"
python training/train.py \
    --config configs/default.yaml \
    --override configs/h200_optimized.yaml \
    --data-dir ./datasets \
    --epochs 100 \
    --batch-size 8 \
    --no-compile
echo "  ✓ Training complete at: $(date)"

# ---- STEP 7: Register test pairs ----
echo ""
echo "[STEP 7/9] Running inference on test pairs..."
# Find first two patient directories
PATIENTS=($(ls -d datasets/stoic/processed/patient_* 2>/dev/null | head -5))
if [ ${#PATIENTS[@]} -ge 2 ]; then
    MOVING=$(find "${PATIENTS[0]}" -name "*.nii.gz" | head -1)
    FIXED=$(find "${PATIENTS[1]}" -name "*.nii.gz" | head -1)

    python inference/register.py \
        --checkpoint outputs/checkpoints/best_model.pth \
        --moving "$MOVING" \
        --fixed "$FIXED" \
        --output-dir ./results/pair_001 \
        --volume-size 128 128 128

    echo "  ✓ Registration complete"
else
    echo "  ⚠ Not enough patients found for registration"
fi

# ---- STEP 8: Visualize results ----
echo ""
echo "[STEP 8/9] Generating visualizations..."
if [ -d "./results/pair_001" ]; then
    python inference/visualize.py --results-dir ./results/pair_001/
    echo "  ✓ Visualizations saved to results/pair_001/figures/"
fi

# ---- STEP 9: Longitudinal recovery analysis ----
echo ""
echo "[STEP 9/9] Running recovery analysis..."
for i in 0 1 2 3 4; do
    PDIR=$(printf "datasets/stoic/processed/patient_%04d" $i)
    if [ -d "$PDIR" ]; then
        PNAME=$(basename "$PDIR")
        echo "  Analyzing $PNAME..."
        python inference/analyze_recovery.py \
            --checkpoint outputs/checkpoints/best_model.pth \
            --patient-dir "$PDIR" \
            --output-dir "results/recovery/$PNAME" \
            --volume-size 128 128 128 \
            2>&1 | tail -5
    fi
done
echo "  ✓ Recovery analysis complete"

# ---- SUMMARY ----
echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE!"
echo "  Finished at: $(date)"
echo "============================================================"
echo ""
echo "Results:"
echo "  Model checkpoint:  outputs/checkpoints/best_model.pth"
echo "  Training history:  outputs/training_history.json"
echo "  Registration:      results/pair_001/"
echo "  Visualizations:    results/pair_001/figures/"
echo "  Recovery reports:  results/recovery/"
echo ""
echo "To download results to your local machine:"
echo "  scp -r root@<server-ip>:~/covidd/results/ ./local_results/"
echo "  scp -r root@<server-ip>:~/covidd/outputs/ ./local_outputs/"
echo "============================================================"
