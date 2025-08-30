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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable
import os 
import torch
from trl import GRPOConfig #, GRPOTrainer as TRLGRPOTrainer
from .grpo_trainer import GRPOTrainer as TRLGRPOTrainer
from typing_extensions import override

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin, Seq2SeqTrainingArguments

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class GRPOTrainer(TRLGRPOTrainer):
    r"""
    Inherits TRL's GRPOTrainer for Group Relative Policy Optimization.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", str],
        args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        reward_funcs: Optional[Callable] = None,
        processor: Optional["ProcessorMixin"] = None,
        **kwargs,
    ):
        # Create GRPOConfig based on training arguments and finetuning arguments
        grpo_config = GRPOConfig(
            # Inherit base training parameters
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
            # report_to=args.report_to,
            run_name=args.run_name if hasattr(args, 'run_name') else None,
            # Add GRPO specific parameters
            max_completion_length=finetuning_args.grpo_response_length,
            temperature=finetuning_args.grpo_temperature,
        )
        
        # Extract TRL GRPOTrainer required parameters
        trl_kwargs = {}
        
        # Basic parameter mapping
        trl_kwargs["model"] = model  # 可以是字符串或模型对象
        if reward_funcs is not None:
            trl_kwargs["reward_funcs"] = reward_funcs
        trl_kwargs["args"] = grpo_config
        
        # Processor parameter (tokenizer)
        if processor is not None:
            trl_kwargs["processing_class"] = processor
        elif "tokenizer" in kwargs:
            trl_kwargs["processing_class"] = kwargs.pop("tokenizer")
        
        # Dataset parameters
        if "train_dataset" in kwargs:
            trl_kwargs["train_dataset"] = kwargs.pop("train_dataset")
        if "eval_dataset" in kwargs:
            trl_kwargs["eval_dataset"] = kwargs.pop("eval_dataset")
        
        # PEFT config
        if "peft_config" in kwargs:
            trl_kwargs["peft_config"] = kwargs.pop("peft_config")
        
        # Callback functions
        if "callbacks" in kwargs:
            trl_kwargs["callbacks"] = kwargs.pop("callbacks")
        
        # Metric computation
        if "compute_metrics" in kwargs:
            trl_kwargs["compute_metrics"] = kwargs.pop("compute_metrics")
        
        if "preprocess_logits_for_metrics" in kwargs:
            trl_kwargs["preprocess_logits_for_metrics"] = kwargs.pop("preprocess_logits_for_metrics")
        
        # Clean up unused parameters
        kwargs.pop("processor", None)
        
        # Pass other parameters to parent class
        trl_kwargs.update(kwargs)
        
        # Call TRL GRPOTrainer constructor
        super().__init__(**trl_kwargs)
        
        # Save original args and finetuning_args for later use
        self.original_args = args
        self.finetuning_args = finetuning_args
        
        logger.info(f"Initialized GRPO trainer with "
                   f"max_completion_length={finetuning_args.grpo_response_length}, "
                   f"temperature={finetuning_args.grpo_temperature}")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model and processor.
        """
        if hasattr(super(), 'save_model'):
            super().save_model(output_dir, _internal_call)
        else:
            # Fallback implementation if parent doesn't have save_model
            output_dir = output_dir or self.original_args.output_dir
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(output_dir)
            if hasattr(self, 'processing_class') and hasattr(self.processing_class, 'save_pretrained'):
                self.processing_class.save_pretrained(output_dir)

    def save_state(self):
        """
        Save the trainer state.
        """
        if hasattr(super(), 'save_state'):
            super().save_state()
        else:
            # Simple fallback implementation
            pass

    def log_metrics(self, split: str, metrics: Dict[str, float]) -> None:
        """
        Log metrics to all configured loggers.
        """
        if hasattr(super(), 'log_metrics'):
            super().log_metrics(split, metrics)
        else:
            # Simple logging fallback
            logger.info(f"{split} metrics: {metrics}")

    def save_metrics(self, split: str, metrics: Dict[str, float], combined: bool = True) -> None:
        """
        Save metrics to files.
        """
        if hasattr(super(), 'save_metrics'):
            super().save_metrics(split, metrics, combined)
        else:
            # Simple implementation
            import json
            import os
            
            output_file = os.path.join(self.original_args.output_dir, f"{split}_results.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

    def save_predictions(self, dataset, predictions, skip_special_tokens: bool = True):
        """Save prediction results"""
        # If parent class has this method, use it; otherwise implement a simple version
        if hasattr(super(), 'save_predictions'):
            return super().save_predictions(dataset, predictions, skip_special_tokens)
        else:
            # Simple prediction saving logic
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