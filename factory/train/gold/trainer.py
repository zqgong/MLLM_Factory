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
from typing_extensions import override

from ...extras.logging import get_logger
from .gold_trainer import GOLDTrainer as TRLGOLDTrainer


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin, Seq2SeqTrainingArguments

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class GOLDTrainer(TRLGOLDTrainer):
    r"""
    Inherits TRL's GOLDTrainer for Generalized On-policy Logit Distillation.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", str],
        teacher_model: Union["PreTrainedModel", str],
        args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        **kwargs,
    ):
        # Extract TRL GOLDTrainer required parameters
        trl_kwargs = {}
        
        # Basic parameter mapping
        trl_kwargs["model"] = model
        trl_kwargs["teacher_model"] = teacher_model
        trl_kwargs["args"] = args
        
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
        
        # Data collator
        if "data_collator" in kwargs:
            trl_kwargs["data_collator"] = kwargs.pop("data_collator")
        
        # Clean up unused parameters
        kwargs.pop("processor", None)
        
        # Pass other parameters to parent class
        trl_kwargs.update(kwargs)
        
        # Call TRL GOLDTrainer constructor
        super().__init__(**trl_kwargs)
        
        # Save original args and finetuning_args for later use
        self.original_args = args
        self.finetuning_args = finetuning_args
        
        logger.info("Initialized GOLD trainer")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model and processor.
        """
        if hasattr(super(), 'save_model'):
            super().save_model(output_dir, _internal_call)
        else:
            # Fallback implementation if parent doesn't have save_model
            output_dir = output_dir or getattr(self.original_args, 'output_dir', './output')
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
            
            output_dir = getattr(self.original_args, 'output_dir', './output')
            output_file = os.path.join(output_dir, f"{split}_results.json")
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
            
            output_dir = getattr(self.original_args, 'output_dir', './output')
            output_file = os.path.join(output_dir, "generated_predictions.jsonl")
            
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