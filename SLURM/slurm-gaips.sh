#!/bin/sh
# shellcheck disable=SC2206

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --time=320:00:00

###SBATCH --mincpus=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:0
#SBATCH --mem=50G

#SBATCH --job-name=forgot_name
#SBATCH --nodelist=nexus3


CUDA_VISIBLE_DEVICES=-1 python Run.py --training-preset 5 --name "not_a_name"
#CUDA_VISIBLE_DEVICES=-1 python Run.py --training-preset 1
#CUDA_VISIBLE_DEVICES=-1 python Run.py --testing-preset 3 



