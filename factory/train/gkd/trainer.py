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
from trl import GKDConfig
from .gdk_trainer import GKDTrainer as TRLGKDTrainer
from typing_extensions import override

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin, Seq2SeqTrainingArguments

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class GKDTrainer(TRLGKDTrainer):
    r"""
    Inherits TRL's GKDTrainer for Generalized Knowledge Distillation.
    """

    def __init__(
        self,
        student_model: "PreTrainedModel",
        teacher_model: "PreTrainedModel",
        args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        **kwargs,
    ):
        # 创建GKDConfig，继承原有的训练参数，并添加GKD特定参数
        gkd_config = GKDConfig(
            # 继承基础训练参数
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            bf16=args.bf16,
            fp16=args.fp16,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps if hasattr(args, 'eval_steps') else None,
            eval_strategy=args.eval_strategy,
            save_strategy=args.save_strategy,
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=args.remove_unused_columns,
            report_to=args.report_to,
            run_name=args.run_name if hasattr(args, 'run_name') else None,
            ddp_timeout=args.ddp_timeout if hasattr(args, 'ddp_timeout') else None,
            # 添加GKD特定参数
            # kl_alpha=finetuning_args.gkd_alpha,
            beta=finetuning_args.gkd_beta,
            temperature=finetuning_args.gkd_temperature,
        )
        
        # 提取TRL GKDTrainer需要的参数
        trl_kwargs = {}
        
        # 基本参数映射
        trl_kwargs["model"] = student_model
        trl_kwargs["teacher_model"] = teacher_model
        trl_kwargs["args"] = gkd_config
        
        # 处理器参数（tokenizer）
        if processor is not None:
            trl_kwargs["processing_class"] = processor
        elif "tokenizer" in kwargs:
            trl_kwargs["processing_class"] = kwargs.pop("tokenizer")
        
        # 数据集参数
        if "train_dataset" in kwargs:
            trl_kwargs["train_dataset"] = kwargs.pop("train_dataset")
        if "eval_dataset" in kwargs:
            trl_kwargs["eval_dataset"] = kwargs.pop("eval_dataset")
        
        # 数据整理器
        if "data_collator" in kwargs:
            trl_kwargs["data_collator"] = kwargs.pop("data_collator")
        
        # 回调函数
        if "callbacks" in kwargs:
            trl_kwargs["callbacks"] = kwargs.pop("callbacks")
        
        # 指标计算
        if "compute_metrics" in kwargs:
            trl_kwargs["compute_metrics"] = kwargs.pop("compute_metrics")
        
        if "preprocess_logits_for_metrics" in kwargs:
            trl_kwargs["preprocess_logits_for_metrics"] = kwargs.pop("preprocess_logits_for_metrics")
        
        # 清理不需要的参数
        kwargs.pop("processor", None)
        
        # 其他参数传递给父类
        trl_kwargs.update(kwargs)
        
        # 调用TRL GKDTrainer的构造函数
        super().__init__(**trl_kwargs)
        
        # 保存原始args和finetuning_args供后续使用
        self.original_args = args
        self.finetuning_args = finetuning_args

    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     """保存模型，使用原始的output_dir配置"""
    #     if output_dir is None:
    #         output_dir = self.original_args.output_dir
    #     super().save_model(output_dir, _internal_call)

    # def log_metrics(self, split: str, metrics: Dict[str, float]) -> None:
    #     """记录指标"""
    #     super().log_metrics(split, metrics)

    # def save_metrics(self, split: str, metrics: Dict[str, float], combined: bool = True) -> None:
    #     """保存指标"""
    #     super().save_metrics(split, metrics, combined)

    # def save_state(self) -> None:
    #     """保存训练状态"""
    #     super().save_state()

    # def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix: str = "eval", **gen_kwargs) -> Dict[str, float]:
    #     """评估模型"""
    #     return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

    # def predict(self, test_dataset, ignore_keys = None, metric_key_prefix: str = "test", **gen_kwargs):
    #     """预测"""
    #     return super().predict(test_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

    def save_predictions(self, dataset, predictions, skip_special_tokens: bool = True):
        """保存预测结果"""
        # 如果父类没有这个方法，我们需要实现一个简单版本
        if hasattr(super(), 'save_predictions'):
            return super().save_predictions(dataset, predictions, skip_special_tokens)
        else:
            # 简单的预测结果保存逻辑
            import json
            import os
            
            output_file = os.path.join(self.original_args.output_dir, "generated_predictions.jsonl")
            
            with open(output_file, "w", encoding="utf-8") as f:
                for i, prediction in enumerate(predictions.predictions):
                    if hasattr(self.processing_class, 'decode'):
                        decoded_prediction = self.processing_class.decode(prediction, skip_special_tokens=skip_special_tokens)
                    else:
                        decoded_prediction = str(prediction)
                    
                    result = {
                        "id": i,
                        "prediction": decoded_prediction
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            logger.info(f"Predictions saved to {output_file}")