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

#SBATCH --output=/home/rmn3157/MTSCI-main/logs/test.out

#SBATCH --error=/home/rmn3157/MTSCI-main/logs/error.err

# Navigate to project root (MTSCI-main), not into src
cd ../src


python_script="main.py"

scratch=True
cuda='cuda:0'
dataset='ssc_pooled'
feature_num=20
seq_len=20
missing_pattern='point'
missing_ratio=0.2
val_missing_ratio=0.2
test_missing_ratio=0.2
dataset_path="../datasets/SSC/ssc/$dataset/"
checkpoint_path="../saved_models/${dataset}_${missing_pattern}_${missing_ratio}_model.pth"

# Keep everything in MTSCI-main root
if [ $scratch = True ]; then
    log_path="../logs/scratch"
else
    log_path="../logs/test"
fi

mkdir -p "$log_path"

for ((i=1; i<=5; i++))
do
    seed=$i
    echo "🔁 Running seed $seed on device $cuda..."

    log_file="$log_path/${dataset}_${missing_pattern}_ms${missing_ratio}_seed${seed}.log"

    if [ $scratch = True ]; then
        nohup python -u $python_script \
            --scratch \
            --device $cuda \
            --seed $seed \
            --dataset $dataset \
            --dataset_path $dataset_path \
            --seq_len $seq_len \
            --feature $feature_num \
            --missing_pattern $missing_pattern \
            --missing_ratio $missing_ratio \
            > "$log_file" 2>&1 &
    else
        nohup python -u $python_script \
            --device $cuda \
            --seed $seed \
            --dataset $dataset \
            --dataset_path $dataset_path \
            --seq_len $seq_len \
            --feature $feature_num \
            --missing_pattern $missing_pattern \
            --missing_ratio $missing_ratio \
            --val_missing_ratio $val_missing_ratio \
            --test_missing_ratio $test_missing_ratio \
            --checkpoint_path $checkpoint_path \
            --nsample 100 \
            > "$log_file" 2>&1 &
    fi

    wait
    echo ""
done
