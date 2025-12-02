#!/bin/bash
#SBATCH -A shakeri_ds6050         ## Define allocation
#SBATCH --partition=gpu           ## Define GPU partition
#SBATCH --gres=gpu:1              ## FIXED: Changed from 4 to 1 (Your code is single-GPU)
#SBATCH --ntasks=1                ## Specify number of tasks 
#SBATCH --cpus-per-task=4         ## Match this to NUM_WORKERS in main.py
#SBATCH --mem=80G                 ## Memory
#SBATCH -t 1-00:00:00             ## Time limit
#SBATCH -J main_dino_knn          ## Job name
#SBATCH -o main_dino_knn-%A.out   ## Output file
#SBATCH -e main_dino_knn-%A.err   ## Error file
#SBATCH --mail-user=ezq9qu@virginia.edu
#SBATCH --mail-type=ALL

# 1. Load Modules
module purge
module load miniforge

# 2. Activate Environment
source activate /scratch/ezq9qu/llm_course

# 3. Debugging Info (Optional but helpful)
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Using Python: $(which python)"

# 4. Run Script
# FIXED: Removed the duplicate 'main.py'
python main.py \
    --backbone "dinov2" \
    --quadrat_dir /scratch/ezq9qu/quadrat/images \
    --single_plant_dir /scratch/ezq9qu/images_max_side_800 \
    --train_lora\
    --epochs 2