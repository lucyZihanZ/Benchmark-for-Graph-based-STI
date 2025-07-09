#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=result_aqi_D0.5.txt
#SBATCH --partition=gengpu
#SBATCH --constraint=sxm
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 
#SBATCH --mem=100G
#SBATCH --account=p32880
#SBATCH --output=/home/rmn3157/CSDI-main/logs/test.out
#SBATCH --error=/home/rmn3157/CSDI-main/logs/error.err

python exe_ssc.py --nsample 100