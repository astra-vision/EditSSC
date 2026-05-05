#!/bin/bash

#SBATCH --job-name=ae_train
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --output=/home/fbalde/EditSSC/models/semantic_ae/vq_vae_bev_sd_bev_codes_512/logs/training_logs/%x_%j.log
#SBATCH --error=/home/fbalde/EditSSC/models/semantic_ae/vq_vae_bev_sd_bev_codes_512/logs/training_logs/%x_%j.err
#SBATCH --partition=astra
#SBATCH --account=astra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32

PROJECT_ROOT="/home/fbalde/EditSSC"
BASE_CONFIG="${PROJECT_ROOT}/configs/common_ae_base.yaml"
TRAIN_SCRIPT="${PROJECT_ROOT}/script/train_ae_main.py"
CONDA_ENV="editssc"

echo "================================================"
echo "SLURM Job Information"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "================================================"
echo ""

mkdir -p /home/fbalde/EditSSC/models/semantic_ae/vq_vae_bev_sd_bev_codes_512/logs/training_logs

echo "🔧 Setting up environment"
source /home/$USER/.bashrc
conda activate $CONDA_ENV

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

echo "📋 Training Configuration:"
echo "   Base config: $BASE_CONFIG"
echo ""

echo "🚀 Starting training..."
echo "================================================"
echo ""

cd "$PROJECT_ROOT"

CUDA_LAUNCH_BLOCKING=1 python "$TRAIN_SCRIPT" --config "$BASE_CONFIG"

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Job Complete"
echo "================================================"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "================================================"

exit $EXIT_CODE

