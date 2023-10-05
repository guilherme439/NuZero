#!/bin/sh
# shellcheck disable=SC2206

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

#SBATCH --mincpus=18
#SBATCH --gres=shard:2

#SBATCH --job-name=gaips


python Run.py --training-preset 1 --name gaips_big_net --log-driver






