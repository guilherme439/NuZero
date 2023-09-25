# Taken from the ray documentation and modified
# Should be run from the project's root directory

import argparse
import subprocess
import sys
import time
from pathlib import Path

template_file = "SLURM/slurm-template.sh"
JOB_NAME = "${JOB_NAME}"
NUM_NODES = "${NUM_NODES}"
NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
PARTITION_OPTION = "${PARTITION_OPTION}"
GIVEN_NODE = "${GIVEN_NODE}"
LOAD_ENV = "${LOAD_ENV}"
TMP_DIR = "${TMP_DIR}"
NET_NAME = "${NET_NAME}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).",
    )
    parser.add_argument(
        "--num-nodes", "-n", type=int, default=1, help="Number of nodes to use."
    )
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
        "return of 'sinfo'. Default: ''.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use in each node. (Default: 0)",
    )
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
    )
    parser.add_argument(
        "--load-env",
        type=str,
        help="The script to load your environment ('module load cuda/10.1')",
        default="",
    )
    parser.add_argument(
        "--gaips",
        action='store_true'
    )
    parser.add_argument(
        "--rnl",
        action='store_true'
    )
    parser.add_argument(
        "--net-name",
        type=str,
        help="The name of the network you are training",
    )
    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(
        args.exp_name, time.strftime("%m%d-%H%M", time.localtime())
    )

    partition_option = (
        "#SBATCH --partition={}".format(args.partition) if args.partition else ""
    )

    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(PARTITION_OPTION, partition_option)
    text = text.replace(LOAD_ENV, str(args.load_env))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO " "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!",
    )
    text = text.replace(NET_NAME, str(args.net_name))

    if args.rnl:
        text = text.replace(TMP_DIR, "/mnt/cirrus/users/5/2/ist189452/TESE/ray_tmp")
    elif args.gaips:
        text = text.replace(TMP_DIR, "/home/users/gpalma/Desktop/ray_tmp")
    else:
        text = text.replace(TMP_DIR, "/tmp/ray")


    # ===== Save the script =====
    script_file = "SLURM/Scripts/" + str(args.num_nodes) + "_nodes-" + str(args.num_gpus) + "_gpus-" + str(args.exp_name) + ".sh"
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Starting to submit job!")
    subprocess.Popen(["sbatch", script_file])
    print(
        "Job submitted! Script file is at: <{}>. Log file is at: <{}>".format(
            script_file, "{}.log".format(job_name)
        )
    )
    sys.exit(0)