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

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...data.collator import MultiModalDataCollatorForSeq2Seq
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from ..sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor

# Import the GOLD trainer
from .trainer import GOLDTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_gold(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    # Load tokenizer and template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Load dataset
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)

    # Load student model
    student_model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    # Load teacher model
    teacher_model_args = model_args.__class__(
        model_name_or_path=finetuning_args.teacher_model,
        adapter_name_or_path=finetuning_args.teacher_model_adapters,
        quantization_bit=finetuning_args.teacher_model_quantization_bit,
        **{k: v for k, v in model_args.__dict__.items() 
           if k not in ["model_name_or_path", "adapter_name_or_path", "quantization_bit", 'device_map', 'model_max_length', "block_diag_attn", "compute_dtype"]}
    )
    
    # Create a simple finetuning_args for teacher model, only for loading, not for training
    teacher_finetuning_args = finetuning_args.__class__(
        finetuning_type="full",  # Teacher model is always loaded as full model
    )
    
    teacher_model = load_model(tokenizer, teacher_model_args, teacher_finetuning_args, is_trainable=False)
    
    if getattr(student_model, "is_quantized", False) and not training_args.do_train:
        setattr(student_model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # Data collator
    label_pad_token_id = IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if not isinstance(label_pad_token_id, int):
        label_pad_token_id = IGNORE_INDEX
        
    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        model=student_model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=label_pad_token_id,
        **tokenizer_module,
    )

    # Metric module
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Generation kwargs
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    if tokenizer.additional_special_tokens_ids:
        eos_token_ids.extend(tokenizer.additional_special_tokens_ids)
    gen_kwargs["eos_token_id"] = eos_token_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize GOLD trainer
    trainer = GOLDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        processor=tokenizer_module.get("processor"),
        **dataset_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            # Use sft stage for calculate_tps since gold is similar to sft
            # Convert dataset to list format for calculate_tps
            train_dataset = dataset_module["train_dataset"]
            if train_dataset is not None:
                # Create a simple list with dummy data for calculate_tps
                train_dataset_list = [{"input_ids": []}]
            else:
                train_dataset_list = [{"input_ids": []}]
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                train_dataset_list, train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            eval_dataset = dataset_module.get("eval_dataset")
            if isinstance(eval_dataset, dict):
                try:
                    keys += [f"eval_{key}_loss" for key in eval_dataset.keys()]
                except AttributeError:
                    keys += ["eval_loss"]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir or "./", keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    # Convert Seq2SeqTrainingArguments to TrainingArguments for compatibility
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)