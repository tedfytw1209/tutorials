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

module load singularity

# For debug purpose, check if CUDA is currently available for torch. If available, will return `True`.
singularity exec --nv /blue/bianjiang/tienyuchang/monaicore1.3.0 python -c "import torch; print(torch.cuda.is_available())"

# Run a tutorial python script within the container. Modify the path to your container and your script.
singularity exec --nv /blue/bianjiang/tienyuchang/monaicore1.3.0 python training.py -e ./config/environment_mednist.json -c ./config/config_infer_vitconv_mednist_80g.json -d -m /blue/bianjiang/tienyuchang/basemodel/checkpoint_test.pth 
