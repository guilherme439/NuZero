#!/bin/sh
# shellcheck disable=SC2206

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00

#SBATCH --mincpus=18
###SBATCH --cpus-per-task=18
#SBATCH --gres=shard:12

#SBATCH --job-name=gaips


python Run.py --training-preset 5 --name gaips_big_net --log-driver






