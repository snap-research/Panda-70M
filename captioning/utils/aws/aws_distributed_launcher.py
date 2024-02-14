import argparse
import glob
import os
from pathlib import Path
from subprocess import Popen, PIPE
import subprocess
import warnings
import multiprocessing as mp

from typing import List, Tuple, Dict


def do_worker(args):
    node_id, output_path, nodes_count, template = args

    current_formatting_dict = {
        "node_id": node_id,
        "nodes_count": nodes_count
    }

    # Creates and writes the file
    print(template)
    current_file_contents = template.format(**current_formatting_dict)
    current_file_path = os.path.join(output_path, f'{node_id:05d}.yaml')
    print(f" - Writing {current_file_path}")
    with open(current_file_path, 'w') as f:
        f.write(current_file_contents)

    commands = ['snap_rutils', 'cluster', 'run', current_file_path, '--yes', "-s"]
    print(commands)
    proc = Popen(commands, stdout=PIPE, stdin=PIPE, stderr=PIPE, universal_newlines=True, env=os.environ.copy())
    proc.stdin.write("yes\n")
    outputlog, errorlog = proc.communicate()
    print(outputlog)
    print(errorlog)


def launch_aws_distributed(template_filename: str, output_path: str, nodes_count: int, blacklisted_ids: List[int], workers: int):
    """
    Creates the configuration files for aws
    :param template_filename: the template filename
    :param output_path: the output path for the compiled template files
    :param nodes_count: the number of nodes to create
    :param blacklisted_ids: the ids of the nodes to blacklist. Useful to selectively rerun some nodes
    :param workers: the number of workers to use to spawn the jobs.
    """
    with open(template_filename, "r") as file:
        template = file.read()

    # Creates the output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Creates the workers in parallel
    pool = mp.Pool(workers)
    work_items = []
    for node_id in range(nodes_count):
        if node_id not in blacklisted_ids:
            work_items.append((node_id, output_path, nodes_count, template))
    pool.map(do_worker, work_items)
    pool.close()


def main():

    # python3 -m utils.aws.aws_distributed_launcher --template_filename configs/data_processing/aws/getty_images_summart_text_embeddings_t5-11b_template.yaml --nodes_count 60 --output_path configs/data_processing/aws/getty_images_summart_text_embeddings_t5-11b_template
    # python3 -m utils.aws.aws_distributed_launcher --template_filename configs/data_processing/aws/getty_images_summart_text_embeddings_clip_template.yaml --nodes_count 5 --output_path configs/data_processing/aws/getty_images_summart_text_embeddings_clip_template
    # python3 -m utils.aws.aws_distributed_launcher --template_filename configs/data_processing/aws/laion-5b-laion2B-en-aesthetic-5plus-data-512_summary_text_embeddings_t5-11b_template.yaml --nodes_count 5 --output_path configs/data_processing/aws/laion-5b-laion2B-en-aesthetic-5plus-data-512_summary_text_embeddings_t5-11b_template
    # python3 -m utils.aws.aws_distributed_launcher --template_filename configs/data_processing/aws/COYO-700M-512-min-image-size200_t5_summary_text_embeddings_t5-11b_template.yaml --nodes_count 30 --output_path configs/data_processing/aws/COYO-700M-512-min-image-size200_t5_summary_text_embeddings_t5-11b_template



    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_filename", type=str, required=True)
    parser.add_argument("--nodes_count", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--pods_blacklist", type=str, default="")
    parser.add_argument("--workers", type=int, default=4)

    arguments = parser.parse_args()

    template_filename = arguments.template_filename
    nodes_count = arguments.nodes_count
    output_path = arguments.output_path
    pods_blacklist = arguments.pods_blacklist
    workers = arguments.workers

    blacklisted_ids = []
    if pods_blacklist:
        with open(pods_blacklist, "r") as file:
            for line in file:
                blacklisted_id = int(line.strip().split("-")[-2])
                blacklisted_ids.append(blacklisted_id)

    launch_aws_distributed(template_filename, output_path, nodes_count, blacklisted_ids, workers)


if __name__ == '__main__':
    main()
