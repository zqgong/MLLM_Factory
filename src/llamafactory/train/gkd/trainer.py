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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Trainer
from typing_extensions import override

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin, Seq2SeqTrainingArguments

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class GKDTrainer(Trainer):
    r"""
    Inherits Trainer for Generalized Knowledge Distillation.
    """

    def __init__(
        self,
        student_model: "PreTrainedModel",
        teacher_model: "PreTrainedModel",
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        **kwargs,
    ):
        super().__init__(model=student_model, **kwargs)
        self.teacher_model = teacher_model
        self.finetuning_args = finetuning_args
        self.processor = processor
        
        # Move teacher model to device and set to eval mode
        if hasattr(self.args, "device"):
            self.teacher_model = self.teacher_model.to(self.args.device)
        self.teacher_model.eval()

    @override
    def compute_loss(
        self, 
        model: "PreTrainedModel", 
        inputs: Dict[str, Any], 
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        计算GKD损失，包括学生模型的标准损失和与教师模型的KL散度损失
        """
        # 获取学生模型的输出
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss if hasattr(student_outputs, "loss") else None
        student_logits = student_outputs.logits

        # 获取教师模型的输出
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # 计算KL散度损失
        kl_loss = self._compute_kl_loss(student_logits, teacher_logits)
        
        # 合并损失
        if student_loss is not None:
            # 如果有标准损失，结合两种损失
            total_loss = (1 - self.finetuning_args.gkd_beta) * student_loss + self.finetuning_args.gkd_beta * kl_loss
        else:
            # 如果没有标准损失，只使用KL损失
            total_loss = kl_loss

        return (total_loss, student_outputs) if return_outputs else total_loss

    def _compute_kl_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        计算学生模型和教师模型输出之间的KL散度损失
        """
        # 应用温度缩放
        student_probs = torch.softmax(student_logits / self.finetuning_args.gkd_temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / self.finetuning_args.gkd_temperature, dim=-1)
        
        # 计算KL散度
        kl_div = nn.functional.kl_div(
            torch.log(student_probs + 1e-8), 
            teacher_probs, 
            reduction="batchmean"
        )
        
        # 应用权重
        return self.finetuning_args.gkd_alpha * kl_div

    @override
    def prediction_step(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        执行预测步骤，只使用学生模型
        """
        model.eval()
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            
            if prediction_loss_only:
                return (loss, None, None)
                
            logits = outputs.logits
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
                
            labels = inputs.get("labels")
            if labels is not None:
                labels = labels.detach()
                
        return (loss, logits, labels)