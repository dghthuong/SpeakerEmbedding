#!/bin/bash

### Comment lines start with ## or #+space
### Slurm option lines start with #SBATCH

### Here are the SBATCH parameters that you should always consider:
##SBATCH --time=0-00:05:00   ## days-hours:minutes:seconds
#SBATCH --mem 3000M         ## 3GB ram (hardware ratio is < 4GB/core)
#SBATCH --ntasks=1          ## Not strictly necessary because default is 1
#SBATCH --cpus-per-task=1   ## Use greater than 1 for parallelized jobs
#SBATCH --gpus=1

### Here are other SBATCH parameters that you may benefit from using, currently commented out:
#SBATCH --job-name=bash ## job name
#SBATCH --output=job.out  ## standard out file

# source activate myenv

python get_speaker_embedding.py ./config/speaker_encoder.yaml

echo 'finished'
