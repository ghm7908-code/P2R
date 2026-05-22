#!/usr/bin/env bash
#SBATCH --job-name=point2roof_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=96:00:00
## Uncomment and edit these if your cluster requires them:
## #SBATCH --partition=gpu
## #SBATCH --account=your_account

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/geogfs1/home/u3666068/Point2Roof-master}"
RAW_ROOT="${RAW_ROOT:-/geogfs1/groups/hkurs/u3666068mgh/Tallin}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/geogfs1/groups/hkurs/u3666068mgh/Tallin/bwformer_trainval_256}"
SUBSET_COUNT="${SUBSET_COUNT:-4096}"
SUBSET_LIST="${SUBSET_LIST:-${RAW_ROOT}/train_list_subset_${SUBSET_COUNT}.txt}"
DATA_PATH="${DATA_PATH:-${SUBSET_LIST}}"
CFG_FILE="${CFG_FILE:-model_cfg.yaml}"
EXP_TAG="${EXP_TAG:-tallin_subset_${SUBSET_COUNT}}"
CONDA_ENV="${CONDA_ENV:-p2f}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-}"
CREATE_SUBSET="${CREATE_SUBSET:-1}"

cd "${PROJECT_DIR}"
mkdir -p logs "output/${EXP_TAG}"

echo "===== Point2Roof training job ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Project: ${PROJECT_DIR}"
echo "Data: ${DATA_PATH}"
echo "Raw root: ${RAW_ROOT}"
echo "Processed root: ${PROCESSED_ROOT}"
echo "Subset list: ${SUBSET_LIST}"
echo "Config: ${CFG_FILE}"
echo "Experiment tag: ${EXP_TAG}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES before train.py: ${CUDA_VISIBLE_DEVICES:-unset}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

set +u
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "WARNING: conda was not found; using the current Python environment."
fi
set -u

python -V

if [[ "${CREATE_SUBSET}" == "1" && ! -f "${SUBSET_LIST}" ]]; then
  echo "Subset list not found; creating ${SUBSET_LIST}"
  python -u create_bwformer_subset.py \
    --processed_root "${PROCESSED_ROOT}" \
    --raw_root "${RAW_ROOT}" \
    --raw_split train \
    --target_count "${SUBSET_COUNT}" \
    --output_list "${SUBSET_LIST}"
fi

cmd=(
  python -u train.py
  --cfg_file "${CFG_FILE}"
  --data_path "${DATA_PATH}"
  --extra_tag "${EXP_TAG}"
  --gpu "${GPU_ID}"
)

if [[ -n "${BATCH_SIZE}" ]]; then
  cmd+=(--batch_size "${BATCH_SIZE}")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"

echo "Training finished at $(date)"
