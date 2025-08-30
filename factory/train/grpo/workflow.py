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

from typing import TYPE_CHECKING, Optional
from peft import LoraConfig

from ...data import get_trl_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import GRPOTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def reward_func(completions, **kwargs):
    """
    默认奖励函数：奖励生成长度接近目标长度的完成内容
    可以根据具体需求自定义奖励函数
    """
    target_length = kwargs.get("target_length", 100)
    return [-abs(target_length - len(completion)) for completion in completions]


def run_grpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    print(data_args)
    
    # 加载tokenizer和template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # 加载数据集 - 对于GRPO，使用sft数据集格式
    dataset_module = get_trl_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    # print(next(iter(dataset_module)))
    # exit()
    # 指标模块
    metric_module = {}

    # 生成参数（如果需要的话）
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # 准备LoRA配置（如果使用LoRA）
    peft_config = None
    if finetuning_args.finetuning_type == "lora":
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=finetuning_args.lora_rank,
            lora_alpha=finetuning_args.lora_alpha,
            lora_dropout=finetuning_args.lora_dropout,
            target_modules=finetuning_args.lora_target if finetuning_args.lora_target != ["all"] else None,
        )

    # 初始化GRPO训练器
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,  # 直接传入模型路径字符串
        args=training_args,
        finetuning_args=finetuning_args,
        reward_funcs=reward_func,  # 使用默认奖励函数
        callbacks=callbacks,
        processor=tokenizer_module.get("processor"),
        peft_config=peft_config,  # 传入LoRA配置
        train_dataset=dataset_module,
        **metric_module,
    )

    # 训练
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"  # 使用sft stage计算
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    # 评估
    if training_args.do_eval:
        eval_result = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)

    # 预测
    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict")
        if training_args.predict_with_generate:
            trainer.save_predictions(dataset_module["eval_dataset"], predict_results)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)

    # 创建模型卡片和推送
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)