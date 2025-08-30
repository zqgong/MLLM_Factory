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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GKDArguments:
    r"""Arguments pertaining to the GKD training."""

    teacher_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the teacher model used for the GKD training."},
    )
    teacher_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the teacher model."},
    )
    teacher_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the teacher model."},
    )


@dataclass
class GRPOArguments:
    r"""Arguments pertaining to the GRPO training."""

    grpo_beta: float = field(
        default=0.1,
        metadata={"help": "The KL regularization coefficient in GRPO training."},
    )
    grpo_temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature parameter for GRPO training."},
    )
    grpo_response_length: int = field(
        default=1024,
        metadata={"help": "The maximum response length for GRPO training."},
    )
    grpo_local_rollout_forward_batch_size: int = field(
        default=64,
        metadata={"help": "The forward batch size for local rollout in GRPO."},
    )
    grpo_num_ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of PPO epochs in GRPO training."},
    )
    grpo_num_mini_batches: int = field(
        default=1,
        metadata={"help": "The number of mini-batches in GRPO training."},
    )
    grpo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten rewards in GRPO training."},
    )