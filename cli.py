# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from copy import deepcopy
from functools import partial
# os.environ["FORCH_TORCHRUN"] = "1"


def main():
    import launcher
    from factory.chat.chat_model import run_chat
    from factory.eval.evaluator import run_eval
    from factory.extras import logging
    from factory.extras.env import VERSION, print_env
    from factory.extras.misc import find_available_port, get_device_count, is_env_enabled, use_ray
    from factory.train.tuner import export_model, run_exp

    logger = logging.get_logger(__name__)


    COMMAND_MAP = {
        "chat": run_chat,
        "env": print_env,
        "eval": run_eval,
        "export": export_model,
        "train": run_exp,
    }

    command = sys.argv.pop(1)
    if command == "train" and (is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not use_ray())):
        # launch distributed training
        nnodes = os.getenv("NNODES", "1")
        node_rank = os.getenv("NODE_RANK", "0")
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))
        logger.info_rank0(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")
        if int(nnodes) > 1:
            print(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")

        env = deepcopy(os.environ)
        if is_env_enabled("OPTIM_TORCH", "1"):
            # optimize DDP, see https://zhuanlan.zhihu.com/p/671834539
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # NOTE: DO NOT USE shell=True to avoid security risk
        process = subprocess.run(
            (
                "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
            )
            .format(
                nnodes=nnodes,
                node_rank=node_rank,
                nproc_per_node=nproc_per_node,
                master_addr=master_addr,
                master_port=master_port,
                file_name=launcher.__file__,
                args=" ".join(sys.argv[1:]),
            )
            .split(),
            env=env,
            check=True,
        )
        sys.exit(process.returncode)
    elif command in COMMAND_MAP:
        COMMAND_MAP[command]()
    else:
        print(f"Unknown command: {command}.\n")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
