#!/bin/bash
#SBATCH --job-name=SIR_eval_images
#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=18:00:00
#SBATCH --output=logs/eval_out_%A.txt
#SBATCH --error=logs/eval_err_%A.txt

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
module load Pillow/10.0.0-GCCcore-12.3.0
module load tqdm/4.66.1-GCCcore-12.3.0

source .venv/bin/activate
mkdir -p logs

# ==========================================
# --- THE NVMe DATA HEIST ---
# ==========================================
# Slurm dynamically assigns $TMPDIR to an authorized local NVMe path
SCRATCH_DIR="$TMPDIR/dino_db"
mkdir -p "$SCRATCH_DIR"

# IMPORTANT: If your optimal config uses the Giant model, uncomment the line below to stage it to the NVMe drive.
cp /projects/prjs2073/SimilarityImageRetrieval/datasets/ILIAS_100k/DINO_data_vit_giant_patch14_dinov2.lvd142m_518.h5 "$SCRATCH_DIR/" & wait

export LOCAL_DB_PATH="$SCRATCH_DIR"

# ==========================================
# --- EXECUTION ---
# ==========================================
# IMPORTANT: test.py requires explicit arguments. Replace the dummy args below with your optimal configuration!
python -u run_100k_thesis.py TEST.DATASET "'ILIAS_100k'" TEST.EVALUATE True