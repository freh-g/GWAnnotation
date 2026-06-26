#!/bin/bash
#SBATCH --job-name=xgblrrf
#SBATCH --output=xgblrrf_%j.out
#SBATCH --error=xgblrrf_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=20G


python /mnt/beegfs/scratch/francesco.gualdi/Projects/AI_PRS_CADD/XGB_Pipeline_v2/scripts/6_LOGOCV_xconditions_improved.py
