#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load conda
conda activate monai

# Run a tutorial python script within the container. Modify the path to your container and your script.
python training.py -c ./config/config_infer_luna16_80g.yaml
