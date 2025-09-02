#!/bin/bash
#BSUB -J histosweep_job               # Job name
#BSUB -q standard                     # Queue name
#BSUB -n 1                            # CPU cores (increase if using multi with >1 workers)
#BSUB -R "rusage[mem=32000]"          # Memory (MB)
#BSUB -W 100:00                       # Walltime
#BSUB -o logs_output/hs_output_%J.log # Stdout
#BSUB -e logs_errors/hs_error_%J.log  # Stderr

set -euo pipefail

# ================== User-set parameters  ==================
MODE="single"                        # Options: "single" or "multi"
NUM_WORKERS=1                        # Used only for multi mode
HE_PREFIX="HE/demo/"                  # Path to image or folder of images
OUTPUT_DIR="${HE_PREFIX%/}/HistoSweep_Output"
NEED_SCALING=True                   # Must be True for multi mode
NEED_PREPROCESS=True                # Must be True for multi mode

# Pipeline parameters 
pixel_size_raw=0.5
density_thresh=100
clean_background_flag=True
min_size=10
patch_size=16
pixel_size=0.5

#  Conda environment setup 
source ~/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_DEFAULT_ENV}" #default use current active conda env
CONDA_ENV="HistoSweep_python_env"

conda activate "$CONDA_ENV"

# =============================================================
# =============================================================


# ================== Run HistoSweep ==================
echo "Running HistoSweep in '${MODE}' mode"
echo "Conda env: ${CONDA_ENV}"
echo "Start time: $(date)"
echo "HE_PREFIX: ${HE_PREFIX}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"

if [[ "$MODE" == "single" ]]; then

  python run_HistoSweep_single.py \
    --output_dir "$OUTPUT_DIR" \
    --need_scaling_flag "$NEED_SCALING" \
    --need_preprocessing_flag "$NEED_PREPROCESS" \
    --pixel_size_raw "$pixel_size_raw" \
    --density_thresh "$density_thresh" \
    --clean_background_flag "$clean_background_flag" \
    --min_size "$min_size" \
    --patch_size "$patch_size" \
    --pixel_size "$pixel_size" \
    --HE_prefix "$HE_PREFIX"


elif [[ "$MODE" == "multi" ]]; then
  echo "Multi mode with ${NUM_WORKERS} workers"

  python run_HistoSweep_multi.py \
    --HE_prefix "$HE_PREFIX" \
    --output_dir "$OUTPUT_DIR" \
    --need_scaling_flag "$NEED_SCALING" \
    --need_preprocessing_flag "$NEED_PREPROCESS" \
    --pixel_size_raw "$pixel_size_raw" \
    --density_thresh "$density_thresh" \
    --clean_background_flag "$clean_background_flag" \
    --min_size "$min_size" \
    --patch_size "$patch_size" \
    --pixel_size "$pixel_size" \
    --num_workers "$NUM_WORKERS"
else
  echo "❌ Invalid MODE: $MODE (use 'single' or 'multi')"
  exit 1
fi

echo "✅ Job completed at: $(date)"
