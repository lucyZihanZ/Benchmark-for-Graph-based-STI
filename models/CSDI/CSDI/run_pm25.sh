#!/bin/bash

#SBATCH --job-name=test_job

#SBATCH --partition=gengpu

#SBATCH --constraint=sxm

#SBATCH --gres=gpu:a100:1

#SBATCH -N 1

#SBATCH --cpus-per-task=10

#SBATCH --mem=50G

#SBATCH --time=24:00:00

#SBATCH --account=p32880

#SBATCH --output=/home/rmn3157/CSDI/logs/test.out

#SBATCH --error=/home/rmn3157/CSDI/logs/error.err

python exe_pm25.py --nsample 10