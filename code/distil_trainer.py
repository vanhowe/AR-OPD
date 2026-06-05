# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import glob
import inspect
import json
import os
import shutil
import textwrap
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.base_trainer import BaseTrainer
from distil_config import DistilConfig
from cad_utils import science_verifier_score, tooluse_verifier_score
from accelerate.state import AcceleratorState
from transformers.trainer import OPTIMIZER_NAME, OPTIMIZER_NAME_BIN, PREFIX_CHECKPOINT_DIR, SCALER_NAME, SCHEDULER_NAME
from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)
from torch.nn.functional import log_softmax, kl_div


if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb


logger = logging.get_logger(__name__)


class MemoryEfficientSyncRefModelCallback(TrainerCallback):
    """
    Memory-efficient callback to synchronize the model with a reference model.

    Unlike the default SyncRefModelCallback, this version iterates through parameters
    one at a time instead of gathering all parameters at once. This reduces peak memory
    usage from O(full_model_size) to O(single_param_size), making it feasible to sync
    large models with DeepSpeed ZeRO-3.
    """

    def __init__(
        self,
        ref_model: Union[PreTrainedModel, nn.Module],
        accelerator: Optional[Any],
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_param(model_param, ref_param, alpha):
        """Sync a single parameter: ref = alpha * model + (1 - alpha) * ref"""
        ref_param.data.mul_(1.0 - alpha).add_(model_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model_memory_efficient(model, target_model, alpha):
        """
        Sync target_model to track model, gathering one parameter at a time.

        This is O(1) in memory overhead instead of O(N) where N is model size.
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        is_zero3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if is_zero3:
            import deepspeed

            # Iterate through parameters one at a time
            for (name, model_param), (_, ref_param) in zip(
                model.named_parameters(), target_model.named_parameters()
            ):
                # Gather only this pair of parameters
                with deepspeed.zero.GatheredParameters(
                    [model_param, ref_param], modifier_rank=0
                ):
                    if deepspeed.comm.get_rank() == 0:
                        MemoryEfficientSyncRefModelCallback._sync_param(
                            model_param, ref_param, alpha
                        )
        else:
            # Non-ZeRO-3: just iterate normally
            for model_param, ref_param in zip(model.parameters(), target_model.parameters()):
                MemoryEfficientSyncRefModelCallback._sync_param(model_param, ref_param, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model_memory_efficient(model, self.ref_model, args.ref_model_mixup_alpha)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class DistilTrainer(BaseTrainer):
    """
    Trainer for the Self-Distillation method.

    Example:

    ```python
    from datasets import load_dataset
    from trl import DistilTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")


    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]


    trainer = DistilTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`DistilConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "distil"]
    _name = "Distil"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Union[str, PreTrainedModel],
        base_model: Optional[Union[str, PreTrainedModel]] = None,
        args: Optional[DistilConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        base_kl_weight: float = 0.0,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DistilConfig(f"{model_name}-Distil")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str):  # it's a str, but not "auto"
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `DistilConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DistilConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path, truncation_side="left")

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        self.num_loss_tokens_to_skip = args.num_loss_tokens_to_skip
        self.cdsdft_enable = getattr(args, "cdsdft_enable", False)
        self.cdsdft_delta_metric = getattr(args, "cdsdft_delta_metric", "kl")
        self.cdsdft_delta_threshold = getattr(args, "cdsdft_delta_threshold", 0.02)
        self.cdsdft_retention_weight = getattr(args, "cdsdft_retention_weight", 0.02)
        self.cdsdft_gate_temperature = getattr(args, "cdsdft_gate_temperature", 10.0)
        self.cad_enable = getattr(args, "cad_enable", False)
        self.cad_delta_metric = getattr(args, "cad_delta_metric", "kl")
        self.cad_delta_threshold = getattr(args, "cad_delta_threshold", 0.02)
        self.cad_gate_temperature = getattr(args, "cad_gate_temperature", 10.0)
        self.cad_local_retention_weight = getattr(args, "cad_local_retention_weight", 0.02)
        self.cad_advantage_min = getattr(args, "cad_advantage_min", 0.0)
        self.cad_advantage_max = getattr(args, "cad_advantage_max", 2.0)
        self.cad_value_max = getattr(args, "cad_value_max", 2.0)
        self.osdft_enable = getattr(args, "osdft_enable", False)
        self.osdft_acquisition_weight = getattr(args, "osdft_acquisition_weight", 1.0)
        self.osdft_preservation_weight = getattr(args, "osdft_preservation_weight", 1.0)
        self.osdft_preservation_source = getattr(args, "osdft_preservation_source", "counterfactual")
        self.osdft_project_all = getattr(args, "osdft_project_all", False)
        self.dualsdft_enable = getattr(args, "dualsdft_enable", False)
        self.dual_delta_threshold = getattr(args, "dual_delta_threshold", 0.02)
        self.dual_gate_temperature = getattr(args, "dual_gate_temperature", 10.0)
        self.dual_alpha_floor = getattr(args, "dual_alpha_floor", 0.10)
        self.dual_alpha_cap = getattr(args, "dual_alpha_cap", 0.90)
        self.dual_confidence_power = getattr(args, "dual_confidence_power", 1.0)
        self.gdsdft_enable = getattr(args, "gdsdft_enable", False)
        self.gdsdft_lambda = getattr(args, "gdsdft_lambda", 1.0)
        self.gdsdft_residual_clip = getattr(args, "gdsdft_residual_clip", 5.0)
        self.dual_gopd_enable = getattr(args, "dual_gopd_enable", False)
        self.dual_gopd_lambda = getattr(args, "dual_gopd_lambda", 1.25)
        self.dual_gopd_advantage_threshold = getattr(args, "dual_gopd_advantage_threshold", 0.0)
        self.dual_gopd_residual_clip = getattr(args, "dual_gopd_residual_clip", 5.0)
        self.dual_teacher_cpu_offload = getattr(args, "dual_teacher_cpu_offload", False)
        self.dual_selected_logps_only = getattr(args, "dual_selected_logps_only", False)
        self.dual_exact_chunked_loss = getattr(args, "dual_exact_chunked_loss", False)
        self.dual_terminal_gate_enable = getattr(args, "dual_terminal_gate_enable", False)
        self.dual_terminal_confidence_temperature = getattr(args, "dual_terminal_confidence_temperature", 5.0)
        self.dual_terminal_advantage_threshold = getattr(args, "dual_terminal_advantage_threshold", 0.0)
        self.opd_metrics_enable = getattr(args, "opd_metrics_enable", True)
        self.opd_metrics_topk = getattr(args, "opd_metrics_topk", 16)

        enabled_special_modes = [self.cdsdft_enable, self.cad_enable, self.osdft_enable, self.dualsdft_enable]
        if sum(bool(flag) for flag in enabled_special_modes) > 1:
            raise ValueError("CAD, CD-SDFT, DualSDFT, and OVSDFT are mutually exclusive.")
        if self.cdsdft_enable and self.cdsdft_delta_metric != "kl":
            raise ValueError("Only cdsdft_delta_metric='kl' is supported.")
        if self.cad_enable and self.cad_delta_metric != "kl":
            raise ValueError("Only cad_delta_metric='kl' is supported.")
        if self.osdft_enable and self.osdft_preservation_source not in {"counterfactual", "base"}:
            raise ValueError("OVSDFT preservation source must be 'counterfactual' or 'base'.")
        if not 0.0 <= self.dual_alpha_floor <= 1.0:
            raise ValueError("dual_alpha_floor must be in [0, 1].")
        if not 0.0 <= self.dual_alpha_cap <= 1.0:
            raise ValueError("dual_alpha_cap must be in [0, 1].")
        if self.dual_alpha_floor > self.dual_alpha_cap:
            raise ValueError("dual_alpha_floor must be <= dual_alpha_cap.")
        if self.dual_confidence_power <= 0.0:
            raise ValueError("dual_confidence_power must be positive.")
        if self.gdsdft_lambda < 0.0:
            raise ValueError("gdsdft_lambda must be >= 0.0.")
        if self.gdsdft_residual_clip <= 0.0:
            raise ValueError("gdsdft_residual_clip must be positive.")
        if self.dual_gopd_lambda < 1.0:
            raise ValueError("dual_gopd_lambda must be >= 1.0.")
        if self.dual_gopd_residual_clip <= 0.0:
            raise ValueError("dual_gopd_residual_clip must be positive.")
        if self.gdsdft_enable and self.dual_gopd_enable:
            raise ValueError("gdsdft_enable and dual_gopd_enable cannot be enabled together.")
        if self.gdsdft_enable and self.dual_terminal_gate_enable:
            raise ValueError("gdsdft_enable and dual_terminal_gate_enable cannot be enabled together.")
        if self.dual_gopd_enable and self.dual_terminal_gate_enable:
            raise ValueError("dual_gopd_enable and dual_terminal_gate_enable cannot be enabled together.")
        if self.dual_terminal_confidence_temperature <= 0.0:
            raise ValueError("dual_terminal_confidence_temperature must be positive.")
        if self.opd_metrics_topk <= 0:
            raise ValueError("opd_metrics_topk must be positive.")
        if self.dual_selected_logps_only and not self.dualsdft_enable:
            raise ValueError("dual_selected_logps_only requires dualsdft_enable.")
        if self.dual_selected_logps_only and not self.gdsdft_enable:
            raise ValueError("dual_selected_logps_only currently supports GD-SDFT only.")
        if self.dual_selected_logps_only and self.dual_teacher_cpu_offload:
            raise ValueError("dual_selected_logps_only already avoids full-vocab teacher tensors; do not combine it with dual_teacher_cpu_offload.")
        if self.dual_selected_logps_only and self.dual_terminal_gate_enable:
            raise ValueError("dual_selected_logps_only is incompatible with dual_terminal_gate_enable.")
        if self.dual_exact_chunked_loss and not self.dualsdft_enable:
            raise ValueError("dual_exact_chunked_loss requires dualsdft_enable.")
        if self.dual_exact_chunked_loss and not self.gdsdft_enable:
            raise ValueError("dual_exact_chunked_loss currently supports GD-SDFT only.")
        if self.dual_exact_chunked_loss and self.dual_teacher_cpu_offload:
            raise ValueError("dual_exact_chunked_loss already avoids dual_teacher_cpu_offload.")
        if self.dual_exact_chunked_loss and self.dual_selected_logps_only:
            raise ValueError("dual_exact_chunked_loss and dual_selected_logps_only are mutually exclusive.")
        if self.dual_exact_chunked_loss and self.dual_terminal_gate_enable:
            raise ValueError("dual_exact_chunked_loss is incompatible with dual_terminal_gate_enable.")

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in DistilTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO-like algorithms, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in Distil
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        if self.osdft_enable and self.accelerator.num_processes > 1:
            logger.info(
                "OVSDFT is running with %s processes; gradient projection stats will be reduced across ranks.",
                self.accelerator.num_processes,
            )
        self._debug_optimizer_step_wrapped = False
        if os.environ.get("OVSDFT_DEBUG_FSDP", "0") == "1":
            original_clip_grad_norm = self.accelerator.clip_grad_norm_

            def debug_clip_grad_norm(*args, **kwargs):
                self._log_osdft_debug("before clip_grad_norm")
                result = original_clip_grad_norm(*args, **kwargs)
                self._log_osdft_debug(f"after clip_grad_norm; grad_norm={result}")
                return result

            self.accelerator.clip_grad_norm_ = debug_clip_grad_norm

            self._ensure_debug_wrapped_optimizer_step()

            import transformers.trainer as hf_trainer_module

            original_save_fsdp_model = hf_trainer_module.save_fsdp_model

            def debug_save_fsdp_model(*args, **kwargs):
                self._log_osdft_debug("before save_fsdp_model")
                result = original_save_fsdp_model(*args, **kwargs)
                self._log_osdft_debug("after save_fsdp_model")
                return result

            hf_trainer_module.save_fsdp_model = debug_save_fsdp_model

            original_save_fsdp_optimizer = hf_trainer_module.save_fsdp_optimizer

            def debug_save_fsdp_optimizer(*args, **kwargs):
                self._log_osdft_debug("before save_fsdp_optimizer")
                result = original_save_fsdp_optimizer(*args, **kwargs)
                self._log_osdft_debug("after save_fsdp_optimizer")
                return result

            hf_trainer_module.save_fsdp_optimizer = debug_save_fsdp_optimizer

        # Reference model
        self.beta = args.beta
        self.alpha = args.alpha
        self.generate_from_teacher = args.generate_from_teacher
        self.base_kl_weight = base_kl_weight
        if ref_model is not None:
            # If a reference model is provided, use it
            self.ref_model = ref_model
        elif self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)

        need_base_model = self.base_kl_weight > 0.0 or (
            self.osdft_enable and self.osdft_preservation_source == "base"
        )
        if need_base_model:
            if base_model is None:
                raise ValueError(
                    "`base_model` must be provided when `base_kl_weight` is greater than zero or OVSDFT uses base preservation."
                )
            self.base_model = base_model
        else:
            self.base_model = None
        if self.dual_selected_logps_only and self.base_model is not None:
            raise ValueError("dual_selected_logps_only is incompatible with base-model anchoring losses.")
        if self.dual_exact_chunked_loss and self.base_model is not None:
            raise ValueError("dual_exact_chunked_loss is incompatible with base-model anchoring losses.")

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
            if self.base_model is not None:
                disable_dropout_in_model(self.base_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._train_start_wall_time = time.time()
        self._last_log_wall_time = self._train_start_wall_time
        self._last_logged_num_tokens = 0.0
        self._metrics_history_path = Path(self.args.output_dir) / "metrics_history.jsonl"
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None
                # Use teacher model for vLLM when generate_from_teacher=True
                vllm_model_path = ref_model.name_or_path if self.generate_from_teacher and ref_model is not None else model.name_or_path
                logger.info(f"[DEBUG] Initializing vLLM with model: {vllm_model_path}, generate_from_teacher={self.generate_from_teacher}")
                self.llm = LLM(
                    model=vllm_model_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    # Important so temperature scaling/logit tweaking affects the TIS log probs
                    logprobs_mode="processed_logprobs",
                )
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if self.base_model is not None:
            if self.is_deepspeed_enabled:
                self.base_model = prepare_deepspeed(self.base_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.base_model = prepare_fsdp(self.base_model, self.accelerator)
            else:
                self.base_model = self.accelerator.prepare_model(self.base_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(MemoryEfficientSyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DistilTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "teacher_prompt",
                "counterfactual_prompt",
                "partial_teacher_prompt",
                "full_teacher_prompt",
                "golden_answer",
                "gold_answer",
                "gold_trace",
                "partial_trace",
                "problem",
                "subject",
                "level",
                "unique_id",
                "cad_answer",
                "image",
                "images",
            ]

    def _resolve_teacher_views(self, example):
        full_teacher_prompt = example.get("full_teacher_prompt", example.get("teacher_prompt", example["prompt"]))
        partial_teacher_prompt = example.get(
            "partial_teacher_prompt",
            example.get("counterfactual_prompt", example["prompt"]),
        )
        return full_teacher_prompt, partial_teacher_prompt

    def _find_last_subsequence(self, sequence, subsequence):
        if not subsequence or len(subsequence) > len(sequence):
            return -1
        last_start = len(sequence) - len(subsequence)
        for start in range(last_start, -1, -1):
            if sequence[start : start + len(subsequence)] == subsequence:
                return start
        return -1

    def _get_dual_terminal_anchor_text(self, example):
        gold_answer = example.get("gold_answer")
        if isinstance(gold_answer, str) and gold_answer.strip():
            return f"<answer>\n{gold_answer.strip()}\n</answer>\n\n"

        gold_trace = example.get("gold_trace")
        if isinstance(gold_trace, str) and gold_trace.strip():
            return gold_trace.strip()

        return None

    def _resolve_dual_terminal_anchor_positions(self, examples, teacher_prompt_ids_list):
        anchor_texts = [self._get_dual_terminal_anchor_text(example) or "" for example in examples]
        tokenized = self.processing_class(
            text=anchor_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )["input_ids"]
        closing_answer_ids = self.processing_class(
            text=["</answer>\n\n"],
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].tolist()

        anchor_positions = []
        for prompt_ids, anchor_ids_tensor in zip(teacher_prompt_ids_list, tokenized):
            prompt_len = len(prompt_ids)
            anchor_position = max(prompt_len - 1, 0)
            anchor_ids = [token_id for token_id in anchor_ids_tensor.tolist() if token_id != self.pad_token_id]
            anchor_start = self._find_last_subsequence(prompt_ids, anchor_ids)
            if anchor_start >= 0:
                anchor_position = anchor_start + len(anchor_ids) - 1
            else:
                closing_start = self._find_last_subsequence(prompt_ids, closing_answer_ids)
                if closing_start >= 0:
                    anchor_position = closing_start + len(closing_answer_ids) - 1
            anchor_positions.append(anchor_position)

        return torch.tensor(anchor_positions, dtype=torch.long, device=self.accelerator.device)

    def _compute_dual_full_confidence(self, full_teacher_all_logps):
        full_probs = full_teacher_all_logps.exp()
        topk = torch.topk(full_probs, k=min(2, full_probs.size(-1)), dim=-1).values
        if topk.size(-1) == 1:
            full_confidence = topk[..., 0]
        else:
            full_confidence = (topk[..., 0] - topk[..., 1]).clamp(min=0.0, max=1.0)
        return full_confidence

    def _compute_dual_alpha(self, full_teacher_all_logps, partial_teacher_all_logps):
        dual_delta = kl_div(partial_teacher_all_logps, full_teacher_all_logps, reduction="none", log_target=True).sum(-1)
        delta_gate = torch.sigmoid(self.dual_gate_temperature * (dual_delta - self.dual_delta_threshold))

        full_confidence = self._compute_dual_full_confidence(full_teacher_all_logps)
        confidence_term = full_confidence.pow(self.dual_confidence_power)

        alpha_span = self.dual_alpha_cap - self.dual_alpha_floor
        dual_alpha = self.dual_alpha_floor + alpha_span * delta_gate * confidence_term
        dual_alpha = dual_alpha.clamp(min=self.dual_alpha_floor, max=self.dual_alpha_cap)
        return dual_delta, delta_gate, full_confidence, dual_alpha

    def _compute_gdsdft_target(self, full_teacher_all_logps, partial_teacher_all_logps):
        dual_delta = kl_div(partial_teacher_all_logps, full_teacher_all_logps, reduction="none", log_target=True).sum(-1)
        view_residual = (full_teacher_all_logps - partial_teacher_all_logps).clamp(
            min=-self.gdsdft_residual_clip,
            max=self.gdsdft_residual_clip,
        )
        mixed_target_logps = torch.log_softmax(
            partial_teacher_all_logps + self.gdsdft_lambda * view_residual,
            dim=-1,
        )
        lambda_tensor = torch.full_like(dual_delta, self.gdsdft_lambda)
        return dual_delta, lambda_tensor, mixed_target_logps

    def _compute_opd_overlap_metrics(
        self,
        student_all_logps: Optional[torch.Tensor],
        teacher_all_logps: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            not self.opd_metrics_enable
            or student_all_logps is None
            or teacher_all_logps is None
        ):
            return None, None

        topk = min(self.opd_metrics_topk, student_all_logps.size(-1), teacher_all_logps.size(-1))
        if topk <= 0:
            return None, None

        student_topk = torch.topk(student_all_logps, k=topk, dim=-1)
        teacher_topk_indices = torch.topk(teacher_all_logps, k=topk, dim=-1).indices

        student_topk_indices = student_topk.indices
        student_topk_logps = student_topk.values.float()
        teacher_topk_logps_on_student = teacher_all_logps.gather(-1, student_topk_indices).float()

        overlap_pairs = student_topk_indices.unsqueeze(-1) == teacher_topk_indices.unsqueeze(-2)
        overlap_mask = overlap_pairs.any(dim=-1)
        overlap_count = overlap_mask.sum(dim=-1)
        overlap_ratio = overlap_count.to(student_topk_logps.dtype) / float(topk)

        neg_inf = float("-inf")
        masked_student_overlap = student_topk_logps.masked_fill(~overlap_mask, neg_inf)
        masked_teacher_overlap = teacher_topk_logps_on_student.masked_fill(~overlap_mask, neg_inf)

        student_overlap_log_mass = torch.logsumexp(masked_student_overlap, dim=-1)
        teacher_overlap_log_mass = torch.logsumexp(masked_teacher_overlap, dim=-1)
        student_overlap_log_mass = torch.where(
            overlap_count > 0,
            student_overlap_log_mass,
            torch.zeros_like(student_overlap_log_mass),
        )
        teacher_overlap_log_mass = torch.where(
            overlap_count > 0,
            teacher_overlap_log_mass,
            torch.zeros_like(teacher_overlap_log_mass),
        )

        renorm_student_logps = masked_student_overlap - student_overlap_log_mass.unsqueeze(-1)
        renorm_teacher_logps = masked_teacher_overlap - teacher_overlap_log_mass.unsqueeze(-1)

        overlap_advantage = renorm_student_logps.exp() * (renorm_teacher_logps - renorm_student_logps)
        overlap_advantage = overlap_advantage.masked_fill(~overlap_mask, 0.0)
        overlap_token_advantage = overlap_advantage.sum(dim=-1) / overlap_count.clamp(min=1).to(
            overlap_advantage.dtype
        )
        overlap_token_advantage = torch.where(
            overlap_count > 0,
            overlap_token_advantage,
            torch.zeros_like(overlap_token_advantage),
        )

        return overlap_ratio, overlap_token_advantage

    def _compute_dual_gopd_target(
        self,
        full_teacher_all_logps,
        partial_teacher_all_logps,
        full_teacher_per_token_logps,
        partial_teacher_per_token_logps,
    ):
        dual_delta = kl_div(partial_teacher_all_logps, full_teacher_all_logps, reduction="none", log_target=True).sum(-1)
        dual_advantage = full_teacher_per_token_logps - partial_teacher_per_token_logps
        dual_gate = torch.sigmoid(
            self.dual_gate_temperature * (dual_advantage - self.dual_gopd_advantage_threshold)
        )

        full_confidence = self._compute_dual_full_confidence(full_teacher_all_logps)
        confidence_term = full_confidence.pow(self.dual_confidence_power)

        alpha_span = self.dual_alpha_cap - self.dual_alpha_floor
        dual_alpha = self.dual_alpha_floor + alpha_span * dual_gate * confidence_term
        dual_alpha = dual_alpha.clamp(min=self.dual_alpha_floor, max=self.dual_alpha_cap)

        view_residual = (full_teacher_all_logps - partial_teacher_all_logps).clamp(
            min=-self.dual_gopd_residual_clip,
            max=self.dual_gopd_residual_clip,
        )
        extrapolation = (self.dual_gopd_lambda * dual_alpha).unsqueeze(-1) * view_residual
        mixed_target_logps = torch.log_softmax(partial_teacher_all_logps + extrapolation, dim=-1)

        return dual_delta, dual_advantage, dual_gate, full_confidence, dual_alpha, mixed_target_logps

    def _compute_dual_terminal_weights(
        self,
        full_completion_hidden_states,
        partial_completion_hidden_states,
        terminal_anchor_states,
        completion_mask,
    ):
        anchor_states = torch.nn.functional.normalize(terminal_anchor_states.float(), dim=-1)
        full_states = torch.nn.functional.normalize(full_completion_hidden_states.float(), dim=-1)
        partial_states = torch.nn.functional.normalize(partial_completion_hidden_states.float(), dim=-1)

        phi_full = (full_states * anchor_states.unsqueeze(1)).sum(-1)
        phi_partial = (partial_states * anchor_states.unsqueeze(1)).sum(-1)
        dual_advantage = phi_full - phi_partial

        valid_mask = completion_mask.to(phi_full.dtype)
        valid_count = valid_mask.sum().clamp(min=1.0)
        phi_full_mean = (phi_full * valid_mask).sum() / valid_count
        centered_phi_full = phi_full - phi_full_mean
        phi_full_std = torch.sqrt((centered_phi_full.square() * valid_mask).sum() / valid_count + 1e-6)
        phi_full_z = centered_phi_full / phi_full_std

        confidence = torch.sigmoid(self.dual_terminal_confidence_temperature * phi_full_z)
        alpha = torch.sigmoid(
            self.dual_gate_temperature * (dual_advantage - self.dual_terminal_advantage_threshold)
        )
        dual_weight = (confidence * alpha).to(dtype=full_completion_hidden_states.dtype)

        return (
            phi_full.to(dtype=full_completion_hidden_states.dtype),
            phi_partial.to(dtype=full_completion_hidden_states.dtype),
            dual_advantage.to(dtype=full_completion_hidden_states.dtype),
            confidence.to(dtype=full_completion_hidden_states.dtype),
            alpha.to(dtype=full_completion_hidden_states.dtype),
            dual_weight,
        )

    def _compute_distillation_kl(self, model_logps, target_logps):
        if self.alpha == 0:
            return kl_div(model_logps, target_logps, reduction="none", log_target=True)
        if self.alpha == 1:
            return kl_div(target_logps, model_logps, reduction="none", log_target=True)

        alpha = torch.tensor(self.alpha, dtype=model_logps.dtype, device=model_logps.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([model_logps + torch.log(1 - alpha), target_logps + torch.log(alpha)]),
            dim=0,
        )
        kl_target = kl_div(mixture_log_probs, target_logps, reduction="none", log_target=True)
        kl_student = kl_div(mixture_log_probs, model_logps, reduction="none", log_target=True)
        return alpha * kl_target + (1 - alpha) * kl_student

    def _compute_selected_logp_distillation(self, model_selected_logps: torch.Tensor, target_selected_logps: torch.Tensor):
        # Scalar KL-style surrogate on the realized completion tokens only. This avoids materializing
        # full-vocab teacher log-probs in the single-GPU survival path.
        return torch.exp(target_selected_logps - model_selected_logps) - (
            target_selected_logps - model_selected_logps
        ) - 1

    def _compute_gdsdft_selected_token_target(
        self,
        completion_ids: torch.Tensor,
        full_teacher_logits: torch.Tensor,
        partial_teacher_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_logsumexp = torch.logsumexp(full_teacher_logits.float(), dim=-1)
        partial_logsumexp = torch.logsumexp(partial_teacher_logits.float(), dim=-1)
        logprob_residual = (
            full_teacher_logits.float()
            - partial_teacher_logits.float()
            - (full_logsumexp - partial_logsumexp).unsqueeze(-1)
        ).clamp(
            min=-self.gdsdft_residual_clip,
            max=self.gdsdft_residual_clip,
        )
        target_logits = partial_teacher_logits.float() + self.gdsdft_lambda * logprob_residual
        target_per_token_logps = selective_log_softmax(
            target_logits.to(dtype=full_teacher_logits.dtype),
            completion_ids,
        )
        full_teacher_per_token_logps = selective_log_softmax(full_teacher_logits, completion_ids)
        partial_teacher_per_token_logps = selective_log_softmax(partial_teacher_logits, completion_ids)
        insight_delta = full_teacher_per_token_logps - partial_teacher_per_token_logps
        lambda_tensor = torch.full_like(target_per_token_logps, self.gdsdft_lambda)
        return insight_delta, lambda_tensor, target_per_token_logps

    def _compute_gdsdft_losses_with_chunked_logits(
        self,
        student_logits: torch.Tensor,
        full_teacher_logits_cpu: torch.Tensor,
        partial_teacher_logits_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Exact GD-SDFT objective, but streamed chunk-by-chunk to avoid materializing student/teacher
        # full-vocab log-prob tensors simultaneously on GPU.
        device = student_logits.device
        dtype = student_logits.dtype
        chunk_size = 8192

        student_logits_f = student_logits.float()
        student_logsumexp = torch.logsumexp(student_logits_f, dim=-1)

        full_cpu = full_teacher_logits_cpu.float()
        partial_cpu = partial_teacher_logits_cpu.float()
        full_logsumexp_cpu = torch.logsumexp(full_cpu, dim=-1)
        partial_logsumexp_cpu = torch.logsumexp(partial_cpu, dim=-1)

        acquisition_per_token_loss = torch.zeros_like(student_logsumexp, device=device, dtype=torch.float32)
        partial_target_per_token_loss = torch.zeros_like(student_logsumexp, device=device, dtype=torch.float32)
        per_token_loss = torch.zeros_like(student_logsumexp, device=device, dtype=torch.float32)
        dual_delta_cpu = torch.zeros_like(full_logsumexp_cpu, device=full_logsumexp_cpu.device, dtype=torch.float32)
        mixed_target_logsumexp_cpu = torch.full_like(full_logsumexp_cpu, -torch.inf, dtype=torch.float32)

        vocab_size = student_logits.size(-1)
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)

            student_log_chunk = student_logits_f[..., start:end] - student_logsumexp.unsqueeze(-1)
            full_log_chunk_cpu = full_cpu[..., start:end] - full_logsumexp_cpu.unsqueeze(-1)
            partial_log_chunk_cpu = partial_cpu[..., start:end] - partial_logsumexp_cpu.unsqueeze(-1)

            full_log_chunk = full_log_chunk_cpu.to(device=device, dtype=torch.float32)
            partial_log_chunk = partial_log_chunk_cpu.to(device=device, dtype=torch.float32)

            full_probs_chunk = torch.exp(full_log_chunk)
            partial_probs_chunk = torch.exp(partial_log_chunk)

            acquisition_per_token_loss += (full_probs_chunk * (full_log_chunk - student_log_chunk)).sum(dim=-1)
            partial_target_per_token_loss += (partial_probs_chunk * (partial_log_chunk - student_log_chunk)).sum(dim=-1)

            dual_delta_cpu += (
                torch.exp(partial_log_chunk_cpu) * (partial_log_chunk_cpu - full_log_chunk_cpu)
            ).sum(dim=-1)

            mixed_unnorm_chunk_cpu = partial_log_chunk_cpu + self.gdsdft_lambda * (
                full_log_chunk_cpu - partial_log_chunk_cpu
            ).clamp(
                min=-self.gdsdft_residual_clip,
                max=self.gdsdft_residual_clip,
            )
            mixed_target_logsumexp_cpu = torch.logaddexp(
                mixed_target_logsumexp_cpu,
                torch.logsumexp(mixed_unnorm_chunk_cpu, dim=-1),
            )

        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)

            student_log_chunk = student_logits_f[..., start:end] - student_logsumexp.unsqueeze(-1)
            full_log_chunk_cpu = full_cpu[..., start:end] - full_logsumexp_cpu.unsqueeze(-1)
            partial_log_chunk_cpu = partial_cpu[..., start:end] - partial_logsumexp_cpu.unsqueeze(-1)
            mixed_unnorm_chunk_cpu = partial_log_chunk_cpu + self.gdsdft_lambda * (
                full_log_chunk_cpu - partial_log_chunk_cpu
            ).clamp(
                min=-self.gdsdft_residual_clip,
                max=self.gdsdft_residual_clip,
            )
            mixed_log_chunk = (
                mixed_unnorm_chunk_cpu - mixed_target_logsumexp_cpu.unsqueeze(-1)
            ).to(device=device, dtype=torch.float32)
            mixed_probs_chunk = torch.exp(mixed_log_chunk)
            per_token_loss += (mixed_probs_chunk * (mixed_log_chunk - student_log_chunk)).sum(dim=-1)

        lambda_tensor = torch.full_like(acquisition_per_token_loss, self.gdsdft_lambda, dtype=torch.float32)
        return (
            acquisition_per_token_loss.to(dtype=dtype),
            partial_target_per_token_loss.to(dtype=dtype),
            per_token_loss.to(dtype=dtype),
            dual_delta_cpu.to(device=device, dtype=dtype),
            lambda_tensor.to(device=device, dtype=dtype),
            lambda_tensor.to(device=device, dtype=dtype),
        )

    def _offload_logps_to_cpu(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.detach().to(device="cpu", copy=True).contiguous()

    def _compute_dual_losses_with_cpu_offload(
        self,
        all_logps: torch.Tensor,
        full_teacher_all_logps_cpu: torch.Tensor,
        partial_teacher_all_logps_cpu: torch.Tensor,
        full_teacher_per_token_logps: torch.Tensor,
        partial_teacher_per_token_logps: torch.Tensor,
        full_completion_hidden_states: Optional[torch.Tensor] = None,
        partial_completion_hidden_states: Optional[torch.Tensor] = None,
        terminal_anchor_states: Optional[torch.Tensor] = None,
        completion_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            full_cpu = full_teacher_all_logps_cpu.float()
            partial_cpu = partial_teacher_all_logps_cpu.float()
            if self.dual_terminal_gate_enable:
                if (
                    full_completion_hidden_states is None
                    or partial_completion_hidden_states is None
                    or terminal_anchor_states is None
                    or completion_mask is None
                ):
                    raise ValueError("Terminal-gated DualSDFT requires teacher hidden states, anchors, and a completion mask.")
                (
                    dual_delta_cpu,
                    dual_secondary_cpu,
                    dual_advantage_cpu,
                    full_confidence_cpu,
                    dual_gate_cpu,
                    dual_alpha_cpu,
                ) = self._compute_dual_terminal_weights(
                    full_completion_hidden_states,
                    partial_completion_hidden_states,
                    terminal_anchor_states,
                    completion_mask,
                )
                mixed_target_cpu = None
            elif self.gdsdft_enable:
                dual_delta_cpu, dual_lambda_cpu, mixed_target_cpu = self._compute_gdsdft_target(full_cpu, partial_cpu)
                dual_secondary_cpu = None
                dual_advantage_cpu = None
                dual_gate_cpu = dual_lambda_cpu
                full_confidence_cpu = dual_lambda_cpu
                dual_alpha_cpu = dual_lambda_cpu
            elif self.dual_gopd_enable:
                (
                    dual_delta_cpu,
                    dual_advantage_cpu,
                    dual_gate_cpu,
                    full_confidence_cpu,
                    dual_alpha_cpu,
                    mixed_target_cpu,
                ) = self._compute_dual_gopd_target(
                    full_cpu,
                    partial_cpu,
                    full_teacher_per_token_logps.detach().to(device="cpu", dtype=torch.float32, copy=True),
                    partial_teacher_per_token_logps.detach().to(device="cpu", dtype=torch.float32, copy=True),
                )
                dual_secondary_cpu = None
            else:
                dual_delta_cpu, dual_gate_cpu, full_confidence_cpu, dual_alpha_cpu = self._compute_dual_alpha(
                    full_cpu,
                    partial_cpu,
                )
                dual_secondary_cpu = None
                dual_advantage_cpu = None
                mixed_target_cpu = torch.log_softmax(
                    (1.0 - dual_alpha_cpu).unsqueeze(-1) * partial_cpu
                    + dual_alpha_cpu.unsqueeze(-1) * full_cpu,
                    dim=-1,
                )

        device = all_logps.device
        dtype = all_logps.dtype

        full_teacher_all_logps = full_teacher_all_logps_cpu.to(device=device, dtype=dtype)
        acquisition_per_token_loss = self._compute_distillation_kl(all_logps, full_teacher_all_logps).sum(-1)
        del full_teacher_all_logps

        partial_teacher_all_logps = partial_teacher_all_logps_cpu.to(device=device, dtype=dtype)
        partial_target_per_token_loss = self._compute_distillation_kl(all_logps, partial_teacher_all_logps).sum(-1)
        del partial_teacher_all_logps

        if self.dual_terminal_gate_enable:
            per_token_loss = (
                dual_alpha_cpu.to(device=device, dtype=dtype) * acquisition_per_token_loss
                + (1.0 - dual_alpha_cpu.to(device=device, dtype=dtype)) * partial_target_per_token_loss
            )
        else:
            mixed_target_logps = mixed_target_cpu.to(device=device, dtype=dtype)
            per_token_loss = self._compute_distillation_kl(all_logps, mixed_target_logps).sum(-1)
            del mixed_target_logps

        dual_delta = dual_delta_cpu.to(device=device, dtype=dtype)
        dual_secondary = dual_secondary_cpu.to(device=device, dtype=dtype) if dual_secondary_cpu is not None else None
        dual_advantage = dual_advantage_cpu.to(device=device, dtype=dtype) if dual_advantage_cpu is not None else None
        dual_gate = dual_gate_cpu.to(device=device, dtype=dtype)
        full_confidence = full_confidence_cpu.to(device=device, dtype=dtype)
        dual_alpha = dual_alpha_cpu.to(device=device, dtype=dtype)

        return (
            acquisition_per_token_loss,
            partial_target_per_token_loss,
            per_token_loss,
            dual_delta,
            dual_secondary,
            dual_advantage,
            dual_gate,
            full_confidence,
            dual_alpha,
        )

    def _compute_cad_values(self, inputs, completions_text):
        if not self.cad_enable:
            return None

        values = []
        for example, completion_text in zip(inputs, completions_text):
            if "golden_answer" in example and example["golden_answer"] is not None:
                values.append(tooluse_verifier_score(completion_text, example["golden_answer"]))
            elif "gold_answer" in example and example["gold_answer"] is not None:
                values.append(tooluse_verifier_score(completion_text, example["gold_answer"]))
            elif "cad_answer" in example and example["cad_answer"]:
                question_text = None
                prompt_messages = example.get("prompt")
                if isinstance(prompt_messages, list) and len(prompt_messages) > 1 and isinstance(prompt_messages[1], dict):
                    question_text = prompt_messages[1].get("content")
                values.append(
                    science_verifier_score(
                        completion_text,
                        example["cad_answer"],
                        question_text=question_text,
                        question_bucket=example.get("question_bucket"),
                    )
                )
            else:
                values.append(1.0)
        return torch.tensor(values, dtype=torch.float32, device=self.accelerator.device)

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to Distil, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        compute_all_logps=True,
        return_completion_hidden_states=False,
        anchor_positions=None,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_selected_logps = []
        all_logps = []
        all_entropies = []
        all_completion_hidden_states = []
        all_anchor_hidden_states = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings
            if return_completion_hidden_states or anchor_positions is not None:
                model_inputs["output_hidden_states"] = True

            outputs = model(**model_inputs)
            logits = outputs.logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            selected_logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            if compute_all_logps:
                logps = log_softmax(logits, dim=-1)
            else:
                logps = None
            all_selected_logps.append(selected_logps)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

            if return_completion_hidden_states or anchor_positions is not None:
                last_hidden_states = outputs.hidden_states[-1]
                if return_completion_hidden_states:
                    completion_hidden_states = last_hidden_states[:, :-1, :]
                    completion_hidden_states = completion_hidden_states[:, -logits_to_keep:, :]
                    all_completion_hidden_states.append(completion_hidden_states)
                if anchor_positions is not None:
                    batch_anchor_positions = anchor_positions[start : start + input_ids_batch.size(0)].to(last_hidden_states.device)
                    batch_indices = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
                    anchor_hidden_states = last_hidden_states[batch_indices, batch_anchor_positions]
                    all_anchor_hidden_states.append(anchor_hidden_states)

        selected_logps = torch.cat(all_selected_logps, dim=0)
        if compute_all_logps:
            logps = torch.cat(all_logps, dim=0)
        else:
            logps = None
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        completion_hidden_states = (
            torch.cat(all_completion_hidden_states, dim=0) if return_completion_hidden_states else None
        )
        anchor_hidden_states = torch.cat(all_anchor_hidden_states, dim=0) if anchor_positions is not None else None
        return selected_logps, logps, entropies, completion_hidden_states, anchor_hidden_states

    @profiling_decorator
    def _get_per_token_logps_and_logits(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        return_completion_hidden_states=False,
        anchor_positions=None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # This helper is used only in the selected-token single-GPU path, where we need the transient
        # teacher logits to reconstruct a closer approximation of the original GD-SDFT mixed target.
        batch_size = batch_size or input_ids.size(0)
        all_selected_logps = []
        all_logits = []
        all_entropies = []
        all_completion_hidden_states = []
        all_anchor_hidden_states = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False
            if return_completion_hidden_states or anchor_positions is not None:
                model_inputs["output_hidden_states"] = True

            outputs = model(**model_inputs)
            logits = outputs.logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            selected_logps = selective_log_softmax(logits, completion_ids)
            all_selected_logps.append(selected_logps)
            all_logits.append(logits)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

            if return_completion_hidden_states or anchor_positions is not None:
                last_hidden_states = outputs.hidden_states[-1]
                if return_completion_hidden_states:
                    completion_hidden_states = last_hidden_states[:, :-1, :]
                    completion_hidden_states = completion_hidden_states[:, -logits_to_keep:, :]
                    all_completion_hidden_states.append(completion_hidden_states)
                if anchor_positions is not None:
                    batch_anchor_positions = anchor_positions[start : start + input_ids_batch.size(0)].to(last_hidden_states.device)
                    batch_indices = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
                    anchor_hidden_states = last_hidden_states[batch_indices, batch_anchor_positions]
                    all_anchor_hidden_states.append(anchor_hidden_states)

        selected_logps = torch.cat(all_selected_logps, dim=0)
        logits = torch.cat(all_logits, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        completion_hidden_states = (
            torch.cat(all_completion_hidden_states, dim=0) if return_completion_hidden_states else None
        )
        anchor_hidden_states = torch.cat(all_anchor_hidden_states, dim=0) if anchor_positions is not None else None
        return selected_logps, logits, entropies, completion_hidden_states, anchor_hidden_states

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Gather full FSDP1 params once at the root before syncing them into vLLM."""
        if isinstance(module, FSDP):
            offload_to_cpu = os.environ.get("OVSDFT_FSDP_SYNC_OFFLOAD_CPU") == "1"
            self._log_osdft_debug(f"enter _sync_fsdp1_params_to_vllm prefix={prefix or '<root>'}")
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
                self._log_osdft_debug(
                    f"_sync_fsdp1_params_to_vllm before summon_full_params offload_to_cpu={offload_to_cpu} "
                    f"free_gb={free_bytes / (1024 ** 3):.2f} total_gb={total_bytes / (1024 ** 3):.2f}"
                )
            with FSDP.summon_full_params(module, recurse=True, writeback=False, offload_to_cpu=offload_to_cpu):
                self._log_osdft_debug(f"_sync_fsdp1_params_to_vllm entered summon_full_params prefix={prefix or '<root>'}")
                loaded_param_count = 0
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])
                    param_data = param.data.cpu() if offload_to_cpu and param.is_cuda else param.data
                    if loaded_param_count == 0:
                        self._log_osdft_debug(
                            f"_sync_fsdp1_params_to_vllm first_param name={full_name} "
                            f"device={param_data.device} shape={tuple(param_data.shape)}"
                        )

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param_data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param_data)])
                    loaded_param_count += 1
                self._log_osdft_debug(
                    f"_sync_fsdp1_params_to_vllm finished param loop prefix={prefix or '<root>'} count={loaded_param_count}"
                )
            self._log_osdft_debug(f"exit _sync_fsdp1_params_to_vllm prefix={prefix or '<root>'}")
            return

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        # Select which model to sync to vLLM: teacher (ref_model) or student (model)
        # When generate_from_teacher=True, sync the teacher model since vLLM was initialized with teacher weights
        self._log_osdft_debug(
            f"enter _move_model_to_vllm step={self.state.global_step} generate_from_teacher={self.generate_from_teacher}"
        )
        model_to_sync = self.ref_model if self.generate_from_teacher else self.model

        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            if self.generate_from_teacher:
                raise ValueError("PEFT model handling only applies when syncing student model (teacher is typically not PEFT)")
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(model_to_sync)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(model_to_sync)
            else:
                for name, param in model_to_sync.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
        self._log_osdft_debug(f"exit _move_model_to_vllm step={self.state.global_step}")

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        self._log_osdft_debug(
            f"enter _prepare_inputs mode={mode} step_counter={self._step} global_step={self.state.global_step}"
        )
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                self._log_osdft_debug(
                    f"_prepare_inputs generating completions step_counter={self._step} generate_every={generate_every}"
                )
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        self._log_osdft_debug(
            f"exit _prepare_inputs mode={mode} step_counter={self._step} global_step={self.state.global_step}"
        )
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _generate_single_turn(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device
        self._log_osdft_debug(
            f"enter _generate_single_turn prompts={len(prompts)} global_step={self.state.global_step}"
        )

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):  # i.e., when using conversational data
                    prepare_multimodal_messages(prompt, num_images=len(image_list))

        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in prompts
        ]

        if images is not None:
            prompt_inputs = self.processing_class(text=prompts_text, padding=True, return_tensors="pt", **kwargs)
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # Generate completions using either vLLM or regular generation
        # Note: When generate_from_teacher=True, vLLM is initialized with teacher weights
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self._log_osdft_debug("before llm.wake_up")
                self.llm.wake_up()
                self._log_osdft_debug("after llm.wake_up")

            # First, update the vLLM weights if needed
            # When generate_from_teacher=True and sync_ref_model=False, teacher is static so no sync needed
            # (vLLM already loaded teacher weights at initialization)
            should_sync = self.state.global_step != self._last_loaded_step
            if self.generate_from_teacher and not self.args.sync_ref_model:
                should_sync = False  # Teacher is static, no need to sync
            self._log_osdft_debug(
                f"_generate_single_turn should_sync={should_sync} last_loaded_step={self._last_loaded_step} global_step={self.state.global_step}"
            )
            if should_sync:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if images is not None:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if images is not None:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            truncate_prompt_tokens=self.max_prompt_length,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"])
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs = obj_list[0]

                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "truncate_prompt_tokens": self.max_prompt_length,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if images is not None:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images

                if images is not None and all_images:
                    vllm_inputs = []
                    for prompt, image_list in zip(all_prompts_text, all_images):
                        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image_list}})

                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    self._log_osdft_debug(f"before colocate vLLM.generate prompts={len(vllm_inputs)}")
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
                    self._log_osdft_debug(f"after colocate vLLM.generate prompts={len(vllm_inputs)}")

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs

                if self.args.vllm_enable_sleep_mode:
                    self._log_osdft_debug("before llm.sleep")
                    self.llm.sleep(level=1)
                    self._log_osdft_debug("after llm.sleep")

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = paged_prompt_inputs.input_ids
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
            logprobs = None  # not used in this case

        else:
            # Regular generation path
            generate_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                **kwargs,
            )
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
            logprobs = None  # not used in this case

        self._log_osdft_debug(
            f"exit _generate_single_turn prompts={len(prompts)} completions={len(completion_ids)} global_step={self.state.global_step}"
        )
        return prompt_ids, completion_ids, logprobs, forward_kwargs

    def _generate(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        self._log_osdft_debug(f"enter _generate mode={mode} prompts={len(prompts)} global_step={self.state.global_step}")

        prompt_ids, completion_ids, logprobs, forward_kwargs = self._generate_single_turn(prompts, images)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        self._log_osdft_debug(
            f"exit _generate mode={mode} prompts={len(prompts)} total_completion_tokens={int(total_completion_tokens.item()) if torch.is_tensor(total_completion_tokens) else total_completion_tokens}"
        )
        return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        self._log_osdft_debug(
            f"enter _generate_and_score_completions mode={mode} batch={len(inputs)} global_step={self.state.global_step}"
        )

        if self.dualsdft_enable:
            missing_full = not all(
                ("full_teacher_prompt" in example) or ("teacher_prompt" in example) for example in inputs
            )
            missing_partial = not all(
                ("partial_teacher_prompt" in example) or ("counterfactual_prompt" in example) for example in inputs
            )
            if missing_full or missing_partial:
                raise ValueError(
                    "DualSDFT requires both full and partial teacher views via "
                    "`full_teacher_prompt`/`teacher_prompt` and `partial_teacher_prompt`/`counterfactual_prompt`."
                )

        prompts = [x["prompt"] for x in inputs]
        teacher_views = [self._resolve_teacher_views(example) for example in inputs]
        teacher_prompts = [full_teacher_prompt for full_teacher_prompt, _ in teacher_views]
        counterfactual_prompts = [partial_teacher_prompt for _, partial_teacher_prompt in teacher_views]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # Decide whether to generate from teacher (with context) or student (without context)
        generation_prompts = teacher_prompts if self.generate_from_teacher else prompts

        (
            _generation_prompt_ids_list,  # Discard - we'll compute student/teacher prompt IDs separately
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate(generation_prompts, images)

        # Process student prompts (always used for student training, regardless of generation source)
        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in prompts
        ]
        if self.use_vllm:
            self.processing_class.truncation_side = "left"
        student_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        student_inputs = super()._prepare_inputs(student_inputs)
        student_prompt_ids, student_prompt_mask = student_inputs["input_ids"], student_inputs["attention_mask"]
        prompt_ids_list = [p[m].tolist() for p, m in zip(student_prompt_ids, student_prompt_mask.bool())]

        # Process teacher prompts (always used for teacher, regardless of generation source)
        teacher_prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in teacher_prompts
        ]
        teacher_inputs = self.processing_class(
            text=teacher_prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        teacher_inputs = super()._prepare_inputs(teacher_inputs)
        if self.use_vllm:
            self.processing_class.truncation_side = "right"
        teacher_prompt_ids, teacher_prompt_mask = teacher_inputs["input_ids"], teacher_inputs["attention_mask"]
        teacher_prompt_ids_list = [p[m].tolist() for p, m in zip(teacher_prompt_ids, teacher_prompt_mask.bool())]
        dual_terminal_anchor_positions = None
        if self.dualsdft_enable and self.dual_terminal_gate_enable:
            dual_terminal_anchor_positions = self._resolve_dual_terminal_anchor_positions(inputs, teacher_prompt_ids_list)

        counterfactual_prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in counterfactual_prompts
        ]
        counterfactual_inputs = self.processing_class(
            text=counterfactual_prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        counterfactual_inputs = super()._prepare_inputs(counterfactual_inputs)
        counterfactual_prompt_ids, counterfactual_prompt_mask = counterfactual_inputs["input_ids"], counterfactual_inputs["attention_mask"]
        counterfactual_prompt_ids_list = [p[m].tolist() for p, m in zip(counterfactual_prompt_ids, counterfactual_prompt_mask.bool())]

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        teacher_prompt_ids = [torch.tensor(ids, device=device) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in teacher_prompt_ids]
        teacher_prompt_ids = pad(teacher_prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        teacher_prompt_mask = pad(teacher_prompt_mask, padding_value=0, padding_side="left")
        counterfactual_prompt_ids = [torch.tensor(ids, device=device) for ids in counterfactual_prompt_ids_list]
        counterfactual_prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in counterfactual_prompt_ids]
        counterfactual_prompt_ids = pad(counterfactual_prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        counterfactual_prompt_mask = pad(counterfactual_prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        teacher_prompt_completion_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)  # (B, P+C)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            # Skip when generate_from_teacher=True since importance sampling is not used in that case.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if not self.generate_from_teacher and (
                self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction)):
                old_per_token_logps, _, _, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    compute_all_logps=False,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            # Skip when generate_from_teacher=True since vLLM has teacher weights (no mismatch to correct)
            if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )
            else:
                importance_sampling_ratio = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _, _, _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        compute_all_logps=False,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _, _, _, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            compute_all_logps=False,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        cad_values = self._compute_cad_values(inputs, completions_text)

        # Not really necessary, but keeping for now
        rewards = torch.zeros_like(completion_ids, dtype=torch.float32)
        advantages = rewards

        # Keep a copy for logging (data is already local to each process, no slicing needed)
        all_process_advantages = advantages.clone()

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["rewards"]["main"].extend(gather_object(rewards.mean(dim=-1).tolist()))
        self._logs["advantages"].extend(gather_object(all_process_advantages.mean(dim=-1).tolist()))
        reward_to_log = rewards.clone()
        reward_to_log = reward_to_log[completion_mask.bool()]
        mean_reward = torch.mean(reward_to_log) if reward_to_log.numel() > 0 else torch.tensor(0.0, device=device)
        self._metrics[mode]["rewards"].append(self.accelerator.gather(mean_reward).mean().item())
        if cad_values is not None:
            mean_cad_value = torch.mean(cad_values) if cad_values.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["cad_value"].append(self.accelerator.gather(mean_cad_value).nanmean().item())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if importance_sampling_ratio is not None:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "counterfactual_prompt_ids": counterfactual_prompt_ids,
            "counterfactual_prompt_mask": counterfactual_prompt_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "sample_weights": torch.tensor(
                [float(example.get("sample_weight", 1.0)) for example in inputs],
                dtype=torch.float32,
                device=device,
            ),
        }
        if dual_terminal_anchor_positions is not None:
            output["dual_terminal_anchor_positions"] = dual_terminal_anchor_positions
        if cad_values is not None:
            output["cad_values"] = cad_values
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        self._log_osdft_debug(
            f"exit _generate_and_score_completions mode={mode} batch={len(inputs)} global_step={self.state.global_step}"
        )
        return output


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The DistilTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    def _reduce_sequence_loss(self, per_token_loss, loss_completion_mask, sample_weights):
        sample_loss = (per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
        if sample_weights is not None:
            normalized_weights = sample_weights.to(sample_loss.dtype).clamp(min=0.0)
            weight_sum = normalized_weights.sum().clamp(min=torch.finfo(sample_loss.dtype).eps)
            return (sample_loss * normalized_weights).sum() / weight_sum
        return sample_loss.mean()

    def _get_trainable_parameters(self, model):
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    def _clear_parameter_grads(self, parameters):
        for parameter in parameters:
            parameter.grad = None

    def _clone_parameter_grads(self, parameters):
        offload_to_cpu = os.environ.get("OVSDFT_OFFLOAD_GRAD_SNAPSHOTS_CPU") == "1"
        grads = []
        for parameter in parameters:
            if parameter.grad is None:
                grads.append(None)
            else:
                grad = parameter.grad.detach()
                if offload_to_cpu:
                    grads.append(grad.to(device="cpu", copy=True))
                else:
                    grads.append(grad.clone())
        return grads

    def _collect_debug_parameter_samples(self, require_grad=True, max_parameters=3, sample_size=8):
        samples = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.numel() == 0:
                continue
            grad_sample = None
            grad_max_abs = None
            if parameter.grad is not None and parameter.grad.numel() > 0:
                flat_grad = parameter.grad.detach().reshape(-1)
                grad_sample = flat_grad[:sample_size].float().cpu().tolist()
                grad_max_abs = float(flat_grad.abs().max().detach().cpu())
            elif require_grad:
                continue

            flat_parameter = parameter.detach().reshape(-1)
            samples.append(
                {
                    "name": name,
                    "param_sample": flat_parameter[:sample_size].float().cpu().tolist(),
                    "param_max_abs": float(flat_parameter.abs().max().detach().cpu()),
                    "grad_sample": grad_sample,
                    "grad_max_abs": grad_max_abs,
                }
            )
            if len(samples) >= max_parameters:
                break
        return samples

    def _format_debug_parameter_samples(self, samples, baseline=None, include_deltas=False):
        if not samples:
            return "[]"

        baseline_map = {}
        if baseline is not None:
            baseline_map = {item["name"]: item for item in baseline}

        formatted = []
        for sample in samples:
            parts = [sample["name"]]
            parts.append(f"param0={sample['param_sample'][:2]}")
            if sample["grad_sample"] is not None:
                parts.append(f"grad0={sample['grad_sample'][:2]}")
                parts.append(f"grad_max={sample['grad_max_abs']:.6e}")
            if include_deltas:
                previous = baseline_map.get(sample["name"])
                if previous is None:
                    parts.append("delta_max=NA")
                else:
                    delta = [
                        abs(float(current) - float(before))
                        for current, before in zip(sample["param_sample"], previous["param_sample"])
                    ]
                    delta_max = max(delta) if delta else 0.0
                    parts.append(f"delta_max={delta_max:.6e}")
            formatted.append("{" + ", ".join(parts) + "}")
        return "[" + ", ".join(formatted) + "]"

    def _ensure_debug_wrapped_optimizer_step(self):
        if os.environ.get("OVSDFT_DEBUG_FSDP", "0") != "1":
            return
        if self._debug_optimizer_step_wrapped or self.optimizer is None:
            return

        self._log_debug_optimizer_param_mapping()
        original_optimizer_step = self.optimizer.step

        def debug_optimizer_step(*args, **kwargs):
            tracked_before = self._collect_debug_parameter_samples(require_grad=True)
            lrs = [group.get("lr", None) for group in self.optimizer.param_groups]
            self._log_osdft_debug(
                f"before optimizer.step; lrs={lrs}; tracked="
                + self._format_debug_parameter_samples(tracked_before, include_deltas=False)
            )
            result = original_optimizer_step(*args, **kwargs)
            tracked_after = self._collect_debug_parameter_samples(require_grad=False)
            self._log_osdft_debug(
                "after optimizer.step; tracked="
                + self._format_debug_parameter_samples(
                    tracked_after,
                    baseline=tracked_before,
                    include_deltas=True,
                )
            )
            return result

        self.optimizer.step = debug_optimizer_step
        self._debug_optimizer_step_wrapped = True
        self._log_osdft_debug("wrapped optimizer.step for debug")

    def _log_debug_optimizer_param_mapping(self, max_parameters=5):
        model_parameter_names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        descriptions = []
        seen = 0
        for group_index, group in enumerate(self.optimizer.param_groups):
            for parameter in group["params"]:
                matched_name = model_parameter_names.get(id(parameter), "<unmatched>")
                shape = tuple(parameter.shape)
                descriptions.append(f"group={group_index}, name={matched_name}, shape={shape}, type={type(parameter).__name__}")
                seen += 1
                if seen >= max_parameters:
                    self._log_osdft_debug("optimizer param mapping: [" + "; ".join(descriptions) + "]")
                    return
        self._log_osdft_debug("optimizer param mapping: [" + "; ".join(descriptions) + "]")

    def _log_osdft_debug(self, message: str):
        if os.environ.get("OVSDFT_DEBUG_FSDP", "0") != "1":
            return
        rank = self.accelerator.process_index if getattr(self, "accelerator", None) is not None else -1
        print(f"[OVSDFT_DEBUG][rank={rank}] {message}", flush=True)
        logger.warning("[OVSDFT_DEBUG][rank=%s] %s", rank, message)

    def _compute_osdft_gradient_stats(self, acquisition_grads, preservation_grads):
        device = self.accelerator.device
        dot = torch.zeros((), device=device)
        acquisition_norm_sq = torch.zeros((), device=device)
        preservation_norm_sq = torch.zeros((), device=device)

        for acquisition_grad, preservation_grad in zip(acquisition_grads, preservation_grads):
            if acquisition_grad is None and preservation_grad is None:
                continue

            if acquisition_grad is not None:
                acquisition_norm_sq = acquisition_norm_sq + acquisition_grad.float().pow(2).sum().to(device)
            if preservation_grad is not None:
                preservation_norm_sq = preservation_norm_sq + preservation_grad.float().pow(2).sum().to(device)
            if acquisition_grad is not None and preservation_grad is not None:
                dot = dot + (acquisition_grad.float() * preservation_grad.float()).sum().to(device)

        if self.accelerator.num_processes > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(dot, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(acquisition_norm_sq, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(preservation_norm_sq, op=torch.distributed.ReduceOp.SUM)

        eps = torch.finfo(torch.float32).eps
        acquisition_norm = torch.sqrt(acquisition_norm_sq.clamp(min=0.0))
        preservation_norm = torch.sqrt(preservation_norm_sq.clamp(min=0.0))
        denom = (acquisition_norm * preservation_norm).clamp(min=eps)
        cosine = dot / denom
        should_project = preservation_norm_sq > eps and (self.osdft_project_all or dot < 0)
        coefficient = dot / preservation_norm_sq.clamp(min=eps) if bool(should_project.item()) else dot.new_zeros(())
        parallel_norm = torch.abs(dot) / preservation_norm.clamp(min=eps) if preservation_norm.item() > 0 else dot.new_zeros(())
        removed_fraction = parallel_norm / acquisition_norm.clamp(min=eps) if acquisition_norm.item() > 0 else dot.new_zeros(())
        return {
            "dot": dot.detach(),
            "cosine": cosine.detach(),
            "coefficient": coefficient.detach(),
            "removed_fraction": removed_fraction.detach(),
            "should_project": should_project.detach().to(dtype=torch.float32),
        }

    def _apply_osdft_gradients(self, parameters, acquisition_grads, preservation_grads, stats):
        coefficient = stats["coefficient"]
        should_project = bool(stats["should_project"].item() > 0.5)
        for parameter, acquisition_grad, preservation_grad in zip(parameters, acquisition_grads, preservation_grads):
            if acquisition_grad is None and preservation_grad is None:
                continue
            target_device = parameter.device
            preservation_component = preservation_grad.detach() if preservation_grad is not None else None
            acquisition_component = acquisition_grad.detach() if acquisition_grad is not None else None
            if preservation_component is not None and preservation_component.device != target_device:
                preservation_component = preservation_component.to(target_device)
            if acquisition_component is not None and acquisition_component.device != target_device:
                acquisition_component = acquisition_component.to(target_device)

            if acquisition_component is not None and should_project and preservation_component is not None:
                acquisition_component = acquisition_component - coefficient.to(acquisition_component.dtype) * preservation_component

            if preservation_component is not None:
                if parameter.grad is None:
                    parameter.grad = (self.osdft_preservation_weight * preservation_component).clone()
                else:
                    parameter.grad.mul_(self.osdft_preservation_weight)
            else:
                parameter.grad = None

            if acquisition_component is not None:
                acquisition_term = self.osdft_acquisition_weight * acquisition_component
                if parameter.grad is None:
                    parameter.grad = acquisition_term.clone()
                else:
                    parameter.grad.add_(acquisition_term)

    def training_step(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        if not self.osdft_enable:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        self._ensure_debug_wrapped_optimizer_step()
        model.train()
        inputs = self._prepare_inputs(inputs)
        if self.is_fsdp_enabled:
            # FSDP relies on backward hooks on the wrapped parameters. Collect each OVSDFT branch with a real
            # backward pass, but use a fresh forward for the second branch because FSDP may reshard/free views
            # after the first backward even when the autograd graph is retained.
            self._log_osdft_debug("enter fsdp training_step")
            with self.compute_loss_context_manager():
                loss, breakdown = self._compute_loss(model, inputs, return_loss_breakdown=True)

            trainable_parameters = self._get_trainable_parameters(model)
            acquisition_loss = breakdown["osdft_acquisition_loss"]
            self._log_osdft_debug(f"computed first forward; acquisition_loss={float(acquisition_loss.detach().cpu())}")
            self._clear_parameter_grads(trainable_parameters)
            self._log_osdft_debug("before acquisition backward")
            self.accelerator.backward(acquisition_loss)
            self._log_osdft_debug("after acquisition backward")
            acquisition_grads = self._clone_parameter_grads(trainable_parameters)
            self._log_osdft_debug("cloned acquisition grads")

            self._clear_parameter_grads(trainable_parameters)
            self._log_osdft_debug("before preservation forward")
            with self.compute_loss_context_manager():
                _, preservation_breakdown = self._compute_loss(
                    model,
                    inputs,
                    return_loss_breakdown=True,
                    log_metrics=False,
                )
            preservation_loss = preservation_breakdown["osdft_preservation_loss"]
            self._log_osdft_debug(f"computed preservation forward; preservation_loss={float(preservation_loss.detach().cpu())}")
            self._log_osdft_debug("before preservation backward")
            self.accelerator.backward(preservation_loss)
            self._log_osdft_debug("after preservation backward")
            trainable_parameters = self._get_trainable_parameters(model)
            preservation_grads = self._clone_parameter_grads(trainable_parameters)
            self._log_osdft_debug("cloned preservation grads")
        else:
            with self.compute_loss_context_manager():
                loss, breakdown = self._compute_loss(model, inputs, return_loss_breakdown=True)

            trainable_parameters = self._get_trainable_parameters(model)
            acquisition_loss = breakdown["osdft_acquisition_loss"]
            preservation_loss = breakdown["osdft_preservation_loss"]
            acquisition_grads = torch.autograd.grad(
                acquisition_loss,
                trainable_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            preservation_grads = torch.autograd.grad(
                preservation_loss,
                trainable_parameters,
                allow_unused=True,
            )
        self._log_osdft_debug("before gradient stats")
        stats = self._compute_osdft_gradient_stats(acquisition_grads, preservation_grads)
        self._log_osdft_debug("after gradient stats")
        self._apply_osdft_gradients(trainable_parameters, acquisition_grads, preservation_grads, stats)
        self._log_osdft_debug("after apply gradients")

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["osdft_grad_cosine"].append(
            self.accelerator.gather(stats["cosine"]).nanmean().item()
        )
        self._metrics[mode]["osdft_projection_coef"].append(
            self.accelerator.gather(stats["coefficient"]).nanmean().item()
        )
        self._metrics[mode]["osdft_removed_fraction"].append(
            self.accelerator.gather(stats["removed_fraction"]).nanmean().item()
        )
        self._metrics[mode]["osdft_projection_applied"].append(
            self.accelerator.gather(stats["should_project"]).nanmean().item()
        )

        return loss.detach()

    def _compute_loss(self, model, inputs, return_loss_breakdown=False, log_metrics=True):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        teacher_prompt_ids, teacher_prompt_mask = inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"]
        counterfactual_prompt_ids = inputs.get("counterfactual_prompt_ids", prompt_ids)
        counterfactual_prompt_mask = inputs.get("counterfactual_prompt_mask", prompt_mask)
        dual_terminal_anchor_positions = inputs.get("dual_terminal_anchor_positions")
        cad_values = inputs.get("cad_values")
        sample_weights = inputs.get("sample_weights")

        # Create a separate mask for loss computation that skips the first N tokens
        # Note: completion_mask is used for both attention (forward pass) and loss computation
        # We need to keep the original for attention, but create a modified one for loss
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            # Create a mask that is 0 for the first num_loss_tokens_to_skip tokens and 1 elsewhere
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            # Apply the skip mask (only mask tokens that were originally unmasked)
            loss_completion_mask = completion_mask * skip_mask

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        counterfactual_input_ids = torch.cat([counterfactual_prompt_ids, completion_ids], dim=1)
        counterfactual_attention_mask = torch.cat([counterfactual_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        use_selected_only_dual_path = self.dualsdft_enable and self.dual_selected_logps_only
        use_chunked_exact_dual_path = self.dualsdft_enable and self.dual_exact_chunked_loss

        # Compute the per_token_logps and the entropy at each position in the completion
        if use_chunked_exact_dual_path:
            per_token_logps, student_logits, entropies, _, _ = self._get_per_token_logps_and_logits(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=True,
                return_completion_hidden_states=False,
                anchor_positions=None,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
            )
            all_logps = None
        else:
            per_token_logps, all_logps, entropies, _, _ = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=True,
                compute_all_logps=not use_selected_only_dual_path,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
            )

        teacher_all_logps_cpu = None
        counterfactual_all_logps_cpu = None
        teacher_logits_cpu = None
        counterfactual_logits_cpu = None
        teacher_completion_hidden_states = None
        teacher_anchor_hidden_states = None
        counterfactual_completion_hidden_states = None
        counterfactual_per_token_logps = None
        teacher_logits = None
        counterfactual_logits = None
        overlap_ratio = None
        overlap_token_advantage = None
        with torch.inference_mode():
            if (use_selected_only_dual_path or use_chunked_exact_dual_path) and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if use_selected_only_dual_path or use_chunked_exact_dual_path:
                (
                    teacher_per_token_logps,
                    teacher_logits,
                    teacher_entropies,
                    teacher_completion_hidden_states,
                    teacher_anchor_hidden_states,
                ) = self._get_per_token_logps_and_logits(
                    self.ref_model,
                    teacher_input_ids,
                    teacher_attention_mask,
                    logits_to_keep,
                    compute_entropy=True,
                    return_completion_hidden_states=False,
                    anchor_positions=None,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    num_images=inputs.get("num_images"),
                    pixel_attention_mask=inputs.get("pixel_attention_mask"),
                    image_sizes=inputs.get("image_sizes"),
                    token_type_ids=inputs.get("token_type_ids"),
                )
                teacher_all_logps = None
            else:
                (
                    teacher_per_token_logps,
                    teacher_all_logps,
                    teacher_entropies,
                    teacher_completion_hidden_states,
                    teacher_anchor_hidden_states,
                ) = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    teacher_input_ids,
                    teacher_attention_mask,
                    logits_to_keep,
                    compute_entropy=True,
                    compute_all_logps=True,
                    return_completion_hidden_states=self.dualsdft_enable and self.dual_terminal_gate_enable,
                    anchor_positions=dual_terminal_anchor_positions if self.dualsdft_enable and self.dual_terminal_gate_enable else None,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    num_images=inputs.get("num_images"),
                    pixel_attention_mask=inputs.get("pixel_attention_mask"),
                    image_sizes=inputs.get("image_sizes"),
                    token_type_ids=inputs.get("token_type_ids"),
                )
                overlap_ratio, overlap_token_advantage = self._compute_opd_overlap_metrics(
                    all_logps,
                    teacher_all_logps,
                )
            if self.dualsdft_enable and self.dual_teacher_cpu_offload:
                teacher_all_logps_cpu = self._offload_logps_to_cpu(teacher_all_logps)
                del teacher_all_logps
                teacher_all_logps = None
            if self.dualsdft_enable or self.cdsdft_enable or self.cad_enable or (
                self.osdft_enable and self.osdft_preservation_source == "counterfactual"
            ):
                if (use_selected_only_dual_path or use_chunked_exact_dual_path) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if use_selected_only_dual_path or use_chunked_exact_dual_path:
                    (
                        counterfactual_per_token_logps,
                        counterfactual_logits,
                        _,
                        counterfactual_completion_hidden_states,
                        _,
                    ) = self._get_per_token_logps_and_logits(
                        self.ref_model,
                        counterfactual_input_ids,
                        counterfactual_attention_mask,
                        logits_to_keep,
                        compute_entropy=False,
                        return_completion_hidden_states=False,
                        anchor_positions=None,
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        num_images=inputs.get("num_images"),
                        pixel_attention_mask=inputs.get("pixel_attention_mask"),
                        image_sizes=inputs.get("image_sizes"),
                        token_type_ids=inputs.get("token_type_ids"),
                    )
                    counterfactual_all_logps = None
                else:
                    (
                        counterfactual_per_token_logps,
                        counterfactual_all_logps,
                        _,
                        counterfactual_completion_hidden_states,
                        _,
                    ) = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        counterfactual_input_ids,
                        counterfactual_attention_mask,
                        logits_to_keep,
                        compute_entropy=False,
                        compute_all_logps=True,
                        return_completion_hidden_states=self.dualsdft_enable and self.dual_terminal_gate_enable,
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        num_images=inputs.get("num_images"),
                        pixel_attention_mask=inputs.get("pixel_attention_mask"),
                        image_sizes=inputs.get("image_sizes"),
                        token_type_ids=inputs.get("token_type_ids"),
                    )
                if self.dualsdft_enable and self.dual_teacher_cpu_offload:
                    counterfactual_all_logps_cpu = self._offload_logps_to_cpu(counterfactual_all_logps)
                    del counterfactual_all_logps
                    counterfactual_all_logps = None
            if use_chunked_exact_dual_path:
                teacher_logits_cpu = self._offload_logps_to_cpu(teacher_logits)
                counterfactual_logits_cpu = self._offload_logps_to_cpu(counterfactual_logits)
                del teacher_logits
                del counterfactual_logits
                counterfactual_all_logps = None
            if self.base_model is not None:
                _, base_all_logps, _, _, _ = self._get_per_token_logps_and_entropies(
                    self.base_model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    compute_entropy=False,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    num_images=inputs.get("num_images"),
                    pixel_attention_mask=inputs.get("pixel_attention_mask"),
                    image_sizes=inputs.get("image_sizes"),
                    token_type_ids=inputs.get("token_type_ids"),
                )
            else:
                base_all_logps = None

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, loss_completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        dual_delta = None
        dual_gate = None
        full_confidence = None
        dual_alpha = None
        dual_alpha_unscaled = None
        gdsdft_lambda_tensor = None
        full_target_per_token_loss = None
        partial_target_per_token_loss = None
        dual_terminal_phi_partial = None
        dual_terminal_advantage = None
        cad_alpha = None
        cad_advantage = None
        if self.dualsdft_enable:
            if use_chunked_exact_dual_path:
                if teacher_logits_cpu is None or counterfactual_logits_cpu is None:
                    raise ValueError("dual_exact_chunked_loss requires full and partial teacher logits.")
                (
                    acquisition_per_token_loss,
                    partial_target_per_token_loss,
                    per_token_loss,
                    dual_delta,
                    dual_gate,
                    full_confidence,
                ) = self._compute_gdsdft_losses_with_chunked_logits(
                    student_logits,
                    teacher_logits_cpu,
                    counterfactual_logits_cpu,
                )
                gdsdft_lambda_tensor = dual_gate
                dual_alpha = dual_gate
                del student_logits
            elif use_selected_only_dual_path:
                if counterfactual_per_token_logps is None or teacher_logits is None or counterfactual_logits is None:
                    raise ValueError("dual_selected_logps_only requires both full and partial teacher logits.")
                acquisition_per_token_loss = self._compute_selected_logp_distillation(
                    per_token_logps, teacher_per_token_logps
                )
                partial_target_per_token_loss = self._compute_selected_logp_distillation(
                    per_token_logps, counterfactual_per_token_logps
                )
                dual_delta, gdsdft_lambda_tensor, target_per_token_logps = self._compute_gdsdft_selected_token_target(
                    completion_ids,
                    teacher_logits,
                    counterfactual_logits,
                )
                dual_gate = gdsdft_lambda_tensor
                full_confidence = gdsdft_lambda_tensor
                dual_alpha = gdsdft_lambda_tensor
                per_token_loss = self._compute_selected_logp_distillation(
                    per_token_logps,
                    target_per_token_logps,
                )
                del teacher_logits
                del counterfactual_logits
            elif self.dual_teacher_cpu_offload:
                if teacher_all_logps_cpu is None or counterfactual_all_logps_cpu is None:
                    raise ValueError("DualSDFT CPU offload path requires both full and partial teacher log-probs.")
                (
                    acquisition_per_token_loss,
                    partial_target_per_token_loss,
                    per_token_loss,
                    dual_delta,
                    dual_terminal_phi_partial,
                    dual_terminal_advantage,
                    dual_gate,
                    full_confidence,
                    dual_alpha,
                ) = self._compute_dual_losses_with_cpu_offload(
                    all_logps,
                    teacher_all_logps_cpu,
                    counterfactual_all_logps_cpu,
                    teacher_per_token_logps,
                    counterfactual_per_token_logps,
                    full_completion_hidden_states=teacher_completion_hidden_states,
                    partial_completion_hidden_states=counterfactual_completion_hidden_states,
                    terminal_anchor_states=teacher_anchor_hidden_states,
                    completion_mask=loss_completion_mask,
                )
            else:
                if counterfactual_all_logps is None:
                    raise ValueError("DualSDFT requires partial/full teacher prompts.")
                acquisition_per_token_loss = self._compute_distillation_kl(all_logps, teacher_all_logps).sum(-1)
                partial_target_per_token_loss = self._compute_distillation_kl(all_logps, counterfactual_all_logps).sum(-1)
                if self.dual_terminal_gate_enable:
                    if (
                        teacher_completion_hidden_states is None
                        or counterfactual_completion_hidden_states is None
                        or teacher_anchor_hidden_states is None
                    ):
                        raise ValueError("Terminal-gated DualSDFT requires teacher hidden states and answer anchors.")
                    (
                        dual_delta,
                        dual_terminal_phi_partial,
                        dual_terminal_advantage,
                        full_confidence,
                        dual_gate,
                        dual_alpha,
                    ) = self._compute_dual_terminal_weights(
                        teacher_completion_hidden_states,
                        counterfactual_completion_hidden_states,
                        teacher_anchor_hidden_states,
                        loss_completion_mask,
                    )
                    per_token_loss = dual_alpha * acquisition_per_token_loss + (1.0 - dual_alpha) * partial_target_per_token_loss
                elif self.gdsdft_enable:
                    dual_delta, gdsdft_lambda_tensor, mixed_target_logps = self._compute_gdsdft_target(
                        teacher_all_logps,
                        counterfactual_all_logps,
                    )
                    dual_gate = gdsdft_lambda_tensor
                    full_confidence = gdsdft_lambda_tensor
                    dual_alpha = gdsdft_lambda_tensor
                    per_token_loss = self._compute_distillation_kl(all_logps, mixed_target_logps).sum(-1)
                elif self.dual_gopd_enable:
                    (
                        dual_delta,
                        dual_terminal_advantage,
                        dual_gate,
                        full_confidence,
                        dual_alpha,
                        mixed_target_logps,
                    ) = self._compute_dual_gopd_target(
                        teacher_all_logps,
                        counterfactual_all_logps,
                        teacher_per_token_logps,
                        counterfactual_per_token_logps,
                    )
                    per_token_loss = self._compute_distillation_kl(all_logps, mixed_target_logps).sum(-1)
                else:
                    dual_delta, dual_gate, full_confidence, dual_alpha = self._compute_dual_alpha(
                        teacher_all_logps,
                        counterfactual_all_logps,
                    )
                    mixed_target_logps = torch.log_softmax(
                        (1.0 - dual_alpha).unsqueeze(-1) * counterfactual_all_logps
                        + dual_alpha.unsqueeze(-1) * teacher_all_logps,
                        dim=-1,
                    )
                    per_token_loss = self._compute_distillation_kl(all_logps, mixed_target_logps).sum(-1)
            dual_alpha_unscaled = None if self.gdsdft_enable else dual_alpha
            if self.gdsdft_enable and self.dual_teacher_cpu_offload:
                gdsdft_lambda_tensor = dual_alpha
            full_target_per_token_loss = acquisition_per_token_loss
        elif self.cad_enable:
            acquisition_per_token_loss = self._compute_distillation_kl(all_logps, teacher_all_logps).sum(-1)
            if cad_values is None:
                cad_values = torch.ones(all_logps.size(0), dtype=torch.float32, device=all_logps.device)
            delta_kl = kl_div(counterfactual_all_logps, teacher_all_logps, reduction="none", log_target=True).sum(-1)
            delta_gate = torch.sigmoid(
                self.cad_gate_temperature * (delta_kl - self.cad_delta_threshold)
            )
            cad_value_scale = (cad_values.to(all_logps.dtype) / max(self.cad_value_max, 1e-6)).unsqueeze(-1)
            cad_alpha = torch.clamp(cad_value_scale * delta_gate, min=0.0, max=1.0)
            cad_advantage = torch.clamp(
                cad_values.to(all_logps.dtype).unsqueeze(-1) * delta_gate,
                min=self.cad_advantage_min,
                max=self.cad_advantage_max,
            )
            eps = torch.finfo(all_logps.dtype).tiny
            cad_target_logps = torch.logaddexp(
                teacher_all_logps + torch.log(cad_alpha.clamp(min=eps)).unsqueeze(-1),
                counterfactual_all_logps + torch.log((1.0 - cad_alpha).clamp(min=eps)).unsqueeze(-1),
            )
            cad_acquisition_per_token_loss = self._compute_distillation_kl(all_logps, cad_target_logps).sum(-1)
            cf_retention_kl = kl_div(all_logps, counterfactual_all_logps, reduction="none", log_target=True).sum(-1)
            per_token_loss = cad_advantage * cad_acquisition_per_token_loss + self.cad_local_retention_weight * (1.0 - cad_alpha) * cf_retention_kl
        elif self.cdsdft_enable:
            acquisition_per_token_loss = self._compute_distillation_kl(all_logps, teacher_all_logps).sum(-1)
            delta_kl = kl_div(counterfactual_all_logps, teacher_all_logps, reduction="none", log_target=True).sum(-1)
            delta_gate = torch.sigmoid(
                self.cdsdft_gate_temperature * (delta_kl - self.cdsdft_delta_threshold)
            )
            cf_retention_kl = kl_div(all_logps, counterfactual_all_logps, reduction="none", log_target=True).sum(-1)
            per_token_loss = delta_gate * acquisition_per_token_loss + self.cdsdft_retention_weight * cf_retention_kl
        else:
            acquisition_per_token_loss = self._compute_distillation_kl(all_logps, teacher_all_logps).sum(-1)
            delta_kl = None
            delta_gate = None
            cf_retention_kl = None
            per_token_loss = acquisition_per_token_loss
        osdft_acquisition_per_token_loss = acquisition_per_token_loss
        if self.osdft_enable:
            if self.osdft_preservation_source == "counterfactual":
                if counterfactual_all_logps is None:
                    raise ValueError("OVSDFT with counterfactual preservation requires counterfactual logits.")
                osdft_preservation_per_token_loss = kl_div(
                    all_logps, counterfactual_all_logps, reduction="none", log_target=True
                ).sum(-1)
            else:
                if base_all_logps is None:
                    raise ValueError("OVSDFT with base preservation requires `base_model` to be provided.")
                osdft_preservation_per_token_loss = kl_div(
                    all_logps, base_all_logps, reduction="none", log_target=True
                ).sum(-1)
        else:
            osdft_preservation_per_token_loss = None

        if base_all_logps is not None and self.base_kl_weight > 0.0 and not self.osdft_enable:
            base_kl_loss = kl_div(all_logps, base_all_logps, reduction="none", log_target=True)
            base_per_token_loss = base_kl_loss.sum(-1)
            per_token_loss = per_token_loss + self.base_kl_weight * base_per_token_loss
        else:
            base_per_token_loss = None

        if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
            ratio = inputs["importance_sampling_ratio"]
            importance_weights = (ratio * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
            importance_weights = importance_weights.unsqueeze(-1)
            per_token_loss = per_token_loss * importance_weights
            if self.dualsdft_enable:
                full_target_per_token_loss = full_target_per_token_loss * importance_weights
                partial_target_per_token_loss = partial_target_per_token_loss * importance_weights
                dual_delta = dual_delta * importance_weights
                if gdsdft_lambda_tensor is not None:
                    gdsdft_lambda_tensor = gdsdft_lambda_tensor * importance_weights
                if dual_terminal_phi_partial is not None:
                    dual_terminal_phi_partial = dual_terminal_phi_partial * importance_weights
                if dual_terminal_advantage is not None:
                    dual_terminal_advantage = dual_terminal_advantage * importance_weights
                dual_gate = dual_gate * importance_weights
                full_confidence = full_confidence * importance_weights
                dual_alpha = dual_alpha * importance_weights
            if self.osdft_enable:
                osdft_acquisition_per_token_loss = osdft_acquisition_per_token_loss * importance_weights
                osdft_preservation_per_token_loss = osdft_preservation_per_token_loss * importance_weights

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
            if self.dualsdft_enable:
                full_target_per_token_loss = full_target_per_token_loss * entropy_mask
                partial_target_per_token_loss = partial_target_per_token_loss * entropy_mask
                dual_delta = dual_delta * entropy_mask
                if gdsdft_lambda_tensor is not None:
                    gdsdft_lambda_tensor = gdsdft_lambda_tensor * entropy_mask
                if dual_terminal_phi_partial is not None:
                    dual_terminal_phi_partial = dual_terminal_phi_partial * entropy_mask
                if dual_terminal_advantage is not None:
                    dual_terminal_advantage = dual_terminal_advantage * entropy_mask
                dual_gate = dual_gate * entropy_mask
                full_confidence = full_confidence * entropy_mask
                dual_alpha = dual_alpha * entropy_mask
            if self.osdft_enable:
                osdft_acquisition_per_token_loss = osdft_acquisition_per_token_loss * entropy_mask
                osdft_preservation_per_token_loss = osdft_preservation_per_token_loss * entropy_mask

        if self.osdft_enable:
            acquisition_loss = self._reduce_sequence_loss(
                osdft_acquisition_per_token_loss, loss_completion_mask, sample_weights
            )
            preservation_loss = self._reduce_sequence_loss(
                osdft_preservation_per_token_loss, loss_completion_mask, sample_weights
            )
            loss = (
                self.osdft_acquisition_weight * acquisition_loss
                + self.osdft_preservation_weight * preservation_loss
            )
        else:
            loss = self._reduce_sequence_loss(per_token_loss, loss_completion_mask, sample_weights)
        loss = loss / self.current_gradient_accumulation_steps
        if self.osdft_enable:
            acquisition_loss = acquisition_loss / self.current_gradient_accumulation_steps
            preservation_loss = preservation_loss / self.current_gradient_accumulation_steps

        if log_metrics:
            mode = "train" if self.model.training else "eval"
            prompt_lengths = prompt_mask.sum(-1)
            completion_lengths = completion_mask.sum(-1)
            teacher_prompt_lengths = teacher_prompt_mask.sum(-1)
            counterfactual_prompt_lengths = counterfactual_prompt_mask.sum(-1)
            privileged_extra_lengths = (teacher_prompt_lengths - prompt_lengths).clamp(min=0)
            residual_extra_lengths = (teacher_prompt_lengths - counterfactual_prompt_lengths).clamp(min=0)
            loss_token_counts = loss_completion_mask.sum(-1)

            self._append_length_stats(mode, "prompt", prompt_lengths, include_buckets=True)
            self._append_length_stats(mode, "completion", completion_lengths, include_buckets=True)
            self._append_length_stats(mode, "teacher_prompt", teacher_prompt_lengths)
            self._append_length_stats(mode, "counterfactual_prompt", counterfactual_prompt_lengths)
            self._append_length_stats(mode, "privileged_extra", privileged_extra_lengths)
            self._append_length_stats(mode, "residual_extra", residual_extra_lengths)
            self._append_length_stats(mode, "loss_completion", loss_token_counts)
            self._append_scalar_metric(
                mode,
                "loss_token_fraction",
                loss_token_counts.float().sum() / completion_mask.sum().clamp(min=1).float(),
            )
            if entropy_mask is not None:
                self._append_scalar_metric(
                    mode,
                    "entropy_mask_fraction",
                    entropy_mask.float().sum() / loss_completion_mask.sum().clamp(min=1).float(),
                )

            with torch.no_grad():
                kl_approx = (per_token_logps - teacher_per_token_logps) + torch.exp(teacher_per_token_logps - per_token_logps) - 1
                kl_approx_mean = (kl_approx * loss_completion_mask).sum() / loss_completion_mask.sum()
            self._metrics[mode]["kl_approx"].append(self.accelerator.gather(kl_approx_mean).nanmean().item())

            loss_completion_token_count = loss_completion_mask.sum().clamp(min=1.0)

            def masked_batch_mean(x):
                if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                    return x.mean()
                else:
                    return (x * loss_completion_mask).sum() / loss_completion_token_count

            def masked_batch_mean_with_mask(x, mask):
                if x.shape[1] == 1:
                    return x.mean()
                mask_token_count = mask.sum().clamp(min=1.0)
                return (x * mask).sum() / mask_token_count

            if self.beta != 0.0:
                mean_kl = masked_batch_mean(per_token_kl)
                self._metrics[mode]["kl_to_base_model"].append(self.accelerator.gather(mean_kl).nanmean().item())

            if base_per_token_loss is not None:
                mean_base_anchor_kl = masked_batch_mean(base_per_token_loss)
                self._metrics[mode]["base_anchor_kl"].append(
                    self.accelerator.gather(mean_base_anchor_kl).nanmean().item()
                )

            if self.dualsdft_enable:
                mixed_mean = masked_batch_mean(per_token_loss)
                full_mean = masked_batch_mean(full_target_per_token_loss)
                partial_mean = masked_batch_mean(partial_target_per_token_loss)
                gathered_mixed_mean = self.accelerator.gather(mixed_mean).nanmean().item()
                gathered_full_mean = self.accelerator.gather(full_mean).nanmean().item()
                gathered_partial_mean = self.accelerator.gather(partial_mean).nanmean().item()
                self._metrics[mode]["dual_mix_kl"].append(gathered_mixed_mean)
                self._metrics[mode]["dual_full_kl"].append(gathered_full_mean)
                self._metrics[mode]["dual_partial_kl"].append(gathered_partial_mean)
                self._metrics[mode]["anchor_drift_to_partial"].append(gathered_partial_mean)
                if self.dual_terminal_gate_enable:
                    self._metrics[mode]["dual_terminal_phi_full"].append(
                        self.accelerator.gather(masked_batch_mean(dual_delta)).nanmean().item()
                    )
                    if dual_terminal_phi_partial is not None:
                        self._metrics[mode]["dual_terminal_phi_partial"].append(
                            self.accelerator.gather(masked_batch_mean(dual_terminal_phi_partial)).nanmean().item()
                        )
                    if dual_terminal_advantage is not None:
                        self._metrics[mode]["dual_terminal_advantage"].append(
                            self.accelerator.gather(masked_batch_mean(dual_terminal_advantage)).nanmean().item()
                        )
                elif self.dual_gopd_enable:
                    self._metrics[mode]["dual_delta"].append(
                        self.accelerator.gather(masked_batch_mean(dual_delta)).nanmean().item()
                    )
                    if dual_terminal_advantage is not None:
                        self._metrics[mode]["dual_advantage"].append(
                            self.accelerator.gather(masked_batch_mean(dual_terminal_advantage)).nanmean().item()
                        )
                elif self.gdsdft_enable:
                    self._metrics[mode]["dual_delta"].append(
                        self.accelerator.gather(masked_batch_mean(dual_delta)).nanmean().item()
                    )
                    if gdsdft_lambda_tensor is not None:
                        self._metrics[mode]["gdsdft_lambda"].append(
                            self.accelerator.gather(masked_batch_mean(gdsdft_lambda_tensor)).nanmean().item()
                        )
                else:
                    self._metrics[mode]["dual_delta"].append(
                        self.accelerator.gather(masked_batch_mean(dual_delta)).nanmean().item()
                    )
                if dual_delta is not None:
                    gathered_dual_delta = self.accelerator.gather(masked_batch_mean(dual_delta)).nanmean().item()
                    residual_uptake = gathered_partial_mean - gathered_full_mean
                    residual_capture_ratio = residual_uptake / max(gathered_dual_delta, 1e-8)
                    residual_transfer_efficiency = residual_uptake / max(gathered_partial_mean, 1e-8)
                    self._metrics[mode]["privileged_belief_shift"].append(gathered_dual_delta)
                    self._metrics[mode]["residual_uptake"].append(residual_uptake)
                    self._metrics[mode]["residual_capture_ratio"].append(residual_capture_ratio)
                    self._metrics[mode]["residual_transfer_efficiency"].append(residual_transfer_efficiency)
                if not self.gdsdft_enable:
                    self._metrics[mode]["dual_gate"].append(
                        self.accelerator.gather(masked_batch_mean(dual_gate)).nanmean().item()
                    )
                    self._metrics[mode]["dual_confidence"].append(
                        self.accelerator.gather(masked_batch_mean(full_confidence)).nanmean().item()
                    )
                    self._metrics[mode]["dual_alpha"].append(
                        self.accelerator.gather(masked_batch_mean(dual_alpha)).nanmean().item()
                    )
                if dual_alpha_unscaled is not None:
                    dual_choice_mask = loss_completion_mask
                    if entropy_mask is not None:
                        dual_choice_mask = dual_choice_mask * entropy_mask
                    dual_prefers_full = (dual_alpha_unscaled >= 0.5).to(dual_alpha_unscaled.dtype)
                    dual_prefers_partial = 1.0 - dual_prefers_full
                    token_full_rate = masked_batch_mean_with_mask(dual_prefers_full, dual_choice_mask)
                    token_partial_rate = masked_batch_mean_with_mask(dual_prefers_partial, dual_choice_mask)
                    per_sequence_alpha = (
                        (dual_alpha_unscaled * dual_choice_mask).sum(-1)
                        / dual_choice_mask.sum(-1).clamp(min=1.0)
                    )
                    sequence_prefers_full = (per_sequence_alpha >= 0.5).to(per_sequence_alpha.dtype).mean()
                    sequence_prefers_partial = 1.0 - sequence_prefers_full
                    self._metrics[mode]["dual_prefers_full_rate"].append(
                        self.accelerator.gather(token_full_rate).nanmean().item()
                    )
                    self._metrics[mode]["dual_prefers_partial_rate"].append(
                        self.accelerator.gather(token_partial_rate).nanmean().item()
                    )
                    self._metrics[mode]["dual_sequence_prefers_full_rate"].append(
                        self.accelerator.gather(sequence_prefers_full).nanmean().item()
                    )
                    self._metrics[mode]["dual_sequence_prefers_partial_rate"].append(
                        self.accelerator.gather(sequence_prefers_partial).nanmean().item()
                    )
            elif self.cad_enable:
                mean_delta = masked_batch_mean(delta_kl)
                mean_gate = masked_batch_mean(delta_gate)
                mean_cf_ret = masked_batch_mean(cf_retention_kl)
                mean_alpha = masked_batch_mean(cad_alpha)
                mean_advantage = masked_batch_mean(cad_advantage)
                self._metrics[mode]["cad_delta"].append(self.accelerator.gather(mean_delta).nanmean().item())
                self._metrics[mode]["cad_gate"].append(self.accelerator.gather(mean_gate).nanmean().item())
                self._metrics[mode]["cad_alpha"].append(self.accelerator.gather(mean_alpha).nanmean().item())
                self._metrics[mode]["cad_advantage"].append(self.accelerator.gather(mean_advantage).nanmean().item())
                self._metrics[mode]["cad_local_ret_kl"].append(self.accelerator.gather(mean_cf_ret).nanmean().item())
            elif self.cdsdft_enable:
                mean_delta = masked_batch_mean(delta_kl)
                mean_gate = masked_batch_mean(delta_gate)
                mean_cf_ret = masked_batch_mean(cf_retention_kl)
                self._metrics[mode]["cdsdft_delta"].append(self.accelerator.gather(mean_delta).nanmean().item())
                self._metrics[mode]["cdsdft_gate"].append(self.accelerator.gather(mean_gate).nanmean().item())
                self._metrics[mode]["cdsdft_cf_ret_kl"].append(self.accelerator.gather(mean_cf_ret).nanmean().item())
            elif self.osdft_enable:
                mean_pres = masked_batch_mean(osdft_preservation_per_token_loss)
                self._metrics[mode]["osdft_acquisition_kl"].append(
                    self.accelerator.gather(acquisition_loss.detach()).nanmean().item()
                )
                self._metrics[mode]["osdft_preservation_kl"].append(
                    self.accelerator.gather(preservation_loss.detach()).nanmean().item()
                )
                self._metrics[mode]["osdft_preservation_token_kl"].append(
                    self.accelerator.gather(mean_pres).nanmean().item()
                )

            mean_entropy = masked_batch_mean(entropies)
            self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
            if teacher_entropies is not None:
                mean_teacher_entropy = masked_batch_mean(teacher_entropies)
                entropy_gap = (teacher_entropies - entropies).abs()
                mean_entropy_gap = masked_batch_mean(entropy_gap)
                self._metrics[mode]["teacher_entropy"].append(
                    self.accelerator.gather(mean_teacher_entropy).nanmean().item()
                )
                self._metrics[mode]["entropy_gap"].append(
                    self.accelerator.gather(mean_entropy_gap).nanmean().item()
                )
            if overlap_ratio is not None:
                self._metrics[mode]["overlap_ratio"].append(
                    self.accelerator.gather(masked_batch_mean(overlap_ratio)).nanmean().item()
                )
            if overlap_token_advantage is not None:
                self._metrics[mode]["overlap_token_advantage"].append(
                    self.accelerator.gather(masked_batch_mean(overlap_token_advantage)).nanmean().item()
                )
            if sample_weights is not None:
                mean_sample_weight = sample_weights.mean()
                self._metrics[mode]["sample_weight"].append(
                    self.accelerator.gather(mean_sample_weight).nanmean().item()
                )

        if return_loss_breakdown:
            return loss, {
                "osdft_acquisition_loss": acquisition_loss,
                "osdft_preservation_loss": preservation_loss,
            }
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().float().cpu().item())
            return float(value.detach().float().mean().cpu().item())
        return float(value)

    def _append_scalar_metric(self, mode: str, key: str, value: Any) -> None:
        self._metrics[mode][key].append(self._to_float(value))

    def _append_length_stats(
        self,
        mode: str,
        prefix: str,
        lengths: torch.Tensor,
        include_buckets: bool = False,
    ) -> None:
        if lengths.numel() == 0:
            return
        gathered = self.accelerator.gather(lengths.detach()).float()
        if gathered.numel() == 0:
            return

        self._append_scalar_metric(mode, f"{prefix}/mean_length", gathered.mean())
        self._append_scalar_metric(mode, f"{prefix}/min_length", gathered.min())
        self._append_scalar_metric(mode, f"{prefix}/max_length", gathered.max())
        for quantile, suffix in ((0.5, "p50_length"), (0.9, "p90_length"), (0.99, "p99_length")):
            self._append_scalar_metric(mode, f"{prefix}/{suffix}", torch.quantile(gathered, quantile))

        if include_buckets:
            bucket_edges = [256, 512, 1024, 2048, 4096]
            previous_edge = None
            for edge in bucket_edges:
                if previous_edge is None:
                    mask = gathered <= edge
                    bucket_name = f"le_{edge}_rate"
                else:
                    mask = (gathered > previous_edge) & (gathered <= edge)
                    bucket_name = f"{previous_edge + 1}_{edge}_rate"
                self._append_scalar_metric(mode, f"{prefix}/{bucket_name}", mask.float().mean())
                previous_edge = edge
            self._append_scalar_metric(mode, f"{prefix}/gt_4096_rate", (gathered > 4096).float().mean())

    def _append_metrics_history(self, mode: str, logs: dict[str, float]) -> None:
        if not self.accelerator.is_main_process:
            return
        payload = {"mode": mode, "step": self.state.global_step}
        if getattr(self.state, "epoch", None) is not None:
            payload["epoch"] = self.state.epoch
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                payload[key] = float(value.detach().float().cpu().item())
            elif isinstance(value, (int, float)):
                payload[key] = float(value)
            else:
                try:
                    payload[key] = float(value)
                except (TypeError, ValueError):
                    payload[key] = value
        self._metrics_history_path.parent.mkdir(parents=True, exist_ok=True)
        with self._metrics_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        now = time.time()
        if mode == "train":
            elapsed = now - self._train_start_wall_time
            since_last_log = max(now - self._last_log_wall_time, 1e-6)
            num_tokens = float(logs.get("num_tokens", self._last_logged_num_tokens))
            token_delta = max(num_tokens - self._last_logged_num_tokens, 0.0)
            logs["wall_time_sec"] = elapsed
            logs["log_interval_sec"] = since_last_log
            logs["tokens_per_sec"] = token_delta / since_last_log if token_delta > 0 else 0.0
            self._last_logged_num_tokens = num_tokens
            self._last_log_wall_time = now
        super().log(logs, start_time)
        self._append_metrics_history(mode, logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }

                if self._logs["images"]:
                    table["images"] = []
                    for image_list in self._logs["images"]:
                        # Convert images to wandb Image objects for proper visualization
                        table["images"].append([wandb.Image(image) for image in image_list])

                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def _strip_checkpoint_training_state(self, checkpoint_dir: Path) -> None:
        removable_patterns = [
            OPTIMIZER_NAME,
            OPTIMIZER_NAME_BIN,
            SCHEDULER_NAME,
            SCALER_NAME,
            "rng_state*.pth",
            f"rank*-of-*-{OPTIMIZER_NAME}",
        ]
        for pattern in removable_patterns:
            for path_str in glob.glob(str(checkpoint_dir / pattern)):
                path = Path(path_str)
                if not path.exists():
                    continue
                if path.is_dir():
                    for child in path.iterdir():
                        if child.is_file():
                            child.unlink()
                    path.rmdir()
                else:
                    path.unlink()

    def _prune_old_checkpoint_training_state(self, run_dir: Path, latest_checkpoint_dir: Path) -> None:
        for checkpoint_dir in run_dir.glob(f"{PREFIX_CHECKPOINT_DIR}-*"):
            if checkpoint_dir == latest_checkpoint_dir or not checkpoint_dir.is_dir():
                continue
            self._strip_checkpoint_training_state(checkpoint_dir)

    def _fsdp_optimizer_save_guard(self, output_dir: str) -> tuple[bool, dict[str, float]]:
        if not self.is_fsdp_enabled:
            return False, {}
        if os.environ.get("OVSDFT_FSDP_ENFORCE_OPTIMIZER_DISK_GUARD", "1") != "1":
            return False, {}

        free_bytes = shutil.disk_usage(output_dir).free
        min_free_gib = float(os.environ.get("OVSDFT_FSDP_MIN_FREE_GB_FOR_OPTIMIZER_SAVE", "40"))
        min_free_bytes = int(min_free_gib * (1 << 30))
        if free_bytes >= min_free_bytes:
            return False, {
                "free_gib": free_bytes / (1 << 30),
                "min_free_gib": min_free_gib,
            }

        return True, {
            "free_gib": free_bytes / (1 << 30),
            "min_free_gib": min_free_gib,
        }

    def _use_custom_fsdp_sharded_save(self) -> bool:
        if not self.is_fsdp_enabled:
            return False
        if os.environ.get("OVSDFT_FSDP_SHARDED_SAFE_SAVE", "1") != "1":
            return False
        fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        if fsdp_plugin is None:
            return False
        return "SHARDED_STATE_DICT" in str(fsdp_plugin.state_dict_type)

    def _save_fsdp_sharded_state(self, output_dir: str, include_optimizer: bool) -> None:
        import warnings

        import torch.distributed.checkpoint as dist_cp
        from accelerate.utils.fsdp_utils import FSDP_MODEL_NAME, _get_model_state_dict, _prepare_sd_options
        from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
        from torch.distributed.checkpoint.filesystem import FileSystemWriter, SerializationFormat
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
        from transformers.trainer import reissue_pt_warnings

        fsdp_plugin = self.accelerator.state.fsdp_plugin
        include_optimizer = include_optimizer and os.environ.get("OVSDFT_FSDP_SKIP_OPTIMIZER_SAVE", "0") != "1"
        writer_kwargs = {
            "serialization_format": SerializationFormat.SAFETENSORS,
            "thread_count": 1,
        }
        if os.environ.get("OVSDFT_FSDP_SINGLE_FILE_PER_RANK", "1") != "1":
            writer_kwargs["single_file_per_rank"] = False

        ctx = (
            FSDP.state_dict_type(
                self.model,
                fsdp_plugin.state_dict_type,
                fsdp_plugin.state_dict_config,
                fsdp_plugin.optim_state_dict_config,
            )
            if fsdp_plugin.fsdp_version == 1
            else nullcontext()
        )
        sd_options = _prepare_sd_options(fsdp_plugin)

        self.accelerator.wait_for_everyone()
        with ctx:
            model_state = _get_model_state_dict(self.model, adapter_only=False, sd_options=sd_options)
            model_dir = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_0")
            os.makedirs(model_dir, exist_ok=True)
            self._log_osdft_debug(
                f"custom sharded save model -> {model_dir} format=safetensors include_optimizer={include_optimizer}"
            )
            dist_cp.save(
                state_dict={"model": model_state},
                storage_writer=dist_cp.FileSystemWriter(model_dir, **writer_kwargs),
                planner=DefaultSavePlanner(),
            )

            if include_optimizer:
                if fsdp_plugin.fsdp_version == 2:
                    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

                    optimizer_state = get_optimizer_state_dict(self.model, self.optimizer, options=sd_options)
                else:
                    optimizer_state = FSDP.optim_state_dict(self.model, self.optimizer)

                optimizer_dir = os.path.join(output_dir, "optimizer_0")
                os.makedirs(optimizer_dir, exist_ok=True)
                self._log_osdft_debug(
                    f"custom sharded save optimizer -> {optimizer_dir} format=safetensors"
                )
                dist_cp.save(
                    state_dict={"optimizer": optimizer_state},
                    storage_writer=dist_cp.FileSystemWriter(optimizer_dir, **writer_kwargs),
                    planner=DefaultSavePlanner(),
                )

        if self.args.should_save:
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
        self.accelerator.wait_for_everyone()

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        custom_fsdp_sharded_save = self._use_custom_fsdp_sharded_save()
        if (
            self.is_fsdp_enabled
            and (
                custom_fsdp_sharded_save
                or os.environ.get("OVSDFT_FSDP_CONFIG_ONLY_SAVE", "0") == "1"
            )
        ):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            self._log_osdft_debug(
                f"enter save_model config_only output_dir={output_dir} internal_call={_internal_call}"
            )
            if self.args.should_save:
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
                self._log_osdft_debug(f"config_only save config -> {output_dir}")
                model_to_save.config.save_pretrained(output_dir)
                if getattr(model_to_save, "can_generate", None) is not None and model_to_save.can_generate():
                    self._log_osdft_debug(f"config_only save generation_config -> {output_dir}")
                    model_to_save.generation_config.save_pretrained(output_dir)
                if self.processing_class is not None:
                    tokenizer_source_candidates = []
                    processing_tokenizer = (
                        self.processing_class.tokenizer
                        if isinstance(self.processing_class, ProcessorMixin)
                        and getattr(self.processing_class, "tokenizer", None) is not None
                        else self.processing_class
                    )

                    for candidate in [
                        getattr(processing_tokenizer, "name_or_path", None),
                        getattr(getattr(processing_tokenizer, "init_kwargs", {}), "get", lambda *_: None)(
                            "name_or_path"
                        ),
                        getattr(model_to_save, "name_or_path", None),
                        getattr(model_to_save.config, "_name_or_path", None),
                    ]:
                        if candidate and candidate not in tokenizer_source_candidates:
                            tokenizer_source_candidates.append(candidate)

                    self._log_osdft_debug(
                        f"config_only tokenizer source candidates={tokenizer_source_candidates}"
                    )

                    tokenizer_source_dir = next(
                        (candidate for candidate in tokenizer_source_candidates if os.path.isdir(candidate)),
                        None,
                    )
                    if tokenizer_source_dir is not None:
                        self._log_osdft_debug(
                            f"config_only copy tokenizer assets from {tokenizer_source_dir} -> {output_dir}"
                        )
                        tokenizer_patterns = [
                            "tokenizer*",
                            "special_tokens_map.json",
                            "added_tokens.json",
                            "merges.txt",
                            "vocab.json",
                            "chat_template.jinja",
                        ]
                        copied_files = []
                        for pattern in tokenizer_patterns:
                            for src_path in glob.glob(os.path.join(tokenizer_source_dir, pattern)):
                                if os.path.isfile(src_path):
                                    dst_path = os.path.join(output_dir, os.path.basename(src_path))
                                    shutil.copy2(src_path, dst_path)
                                    copied_files.append(os.path.basename(dst_path))
                        self._log_osdft_debug(
                            f"config_only copied tokenizer assets={sorted(set(copied_files))}"
                        )
                    else:
                        self._log_osdft_debug(f"config_only save processing_class -> {output_dir}")
                        self.processing_class.save_pretrained(output_dir)
                elif (
                    self.data_collator is not None
                    and hasattr(self.data_collator, "tokenizer")
                    and self.data_collator.tokenizer is not None
                ):
                    self._log_osdft_debug(f"config_only save data_collator.tokenizer -> {output_dir}")
                    self.data_collator.tokenizer.save_pretrained(output_dir)
            if custom_fsdp_sharded_save and self.args.save_only_model:
                self._save_fsdp_sharded_state(output_dir, include_optimizer=False)
            self._log_osdft_debug(
                f"exit save_model config_only output_dir={output_dir} internal_call={_internal_call}"
            )
            return
        self._log_osdft_debug(f"enter save_model output_dir={output_dir}")
        result = super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        self._log_osdft_debug(f"exit save_model output_dir={output_dir}")
        return result

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict] = None) -> None:
        debug_save = os.environ.get("OVSDFT_DEBUG_SAVE_SHARDS", "0") == "1"
        if not debug_save:
            return super()._save(output_dir=output_dir, state_dict=state_dict)

        import transformers.modeling_utils as modeling_utils

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self._log_osdft_debug(
            f"enter _save output_dir={output_dir} state_dict_keys={len(state_dict) if state_dict is not None else 'None'}"
        )

        original_safe_save_file = modeling_utils.safe_save_file

        def logged_safe_save_file(shard_state_dict, filename, metadata=None):
            self._log_osdft_debug(
                f"before safe_save_file filename={filename} tensors={len(shard_state_dict)}"
            )
            result = original_safe_save_file(shard_state_dict, filename, metadata=metadata)
            self._log_osdft_debug(
                f"after safe_save_file filename={filename} tensors={len(shard_state_dict)}"
            )
            return result

        modeling_utils.safe_save_file = logged_safe_save_file
        try:
            result = super()._save(output_dir=output_dir, state_dict=state_dict)
            self._log_osdft_debug(f"exit _save output_dir={output_dir}")
            return result
        finally:
            modeling_utils.safe_save_file = original_safe_save_file

    def _save_optimizer_and_scheduler(self, output_dir):
        self._log_osdft_debug(f"enter _save_optimizer_and_scheduler output_dir={output_dir}")
        should_skip_optimizer_state, guard_info = self._fsdp_optimizer_save_guard(output_dir)
        if should_skip_optimizer_state:
            marker = {
                "reason": "low_disk_headroom",
                "output_dir": output_dir,
                "free_gib": round(guard_info["free_gib"], 3),
                "min_free_gib": round(guard_info["min_free_gib"], 3),
                "timestamp": int(time.time()),
            }
            if self.args.should_save:
                marker_path = Path(output_dir) / "optimizer_state_skipped.json"
                marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
            logger.warning(
                "Skipping FSDP optimizer/scheduler checkpoint save for %s because free disk is %.2f GiB, below the %.2f GiB guard.",
                output_dir,
                guard_info["free_gib"],
                guard_info["min_free_gib"],
            )
            self._log_osdft_debug(
                "skip _save_optimizer_and_scheduler output_dir="
                f"{output_dir} reason=low_disk_headroom free_gib={guard_info['free_gib']:.3f} "
                f"min_free_gib={guard_info['min_free_gib']:.3f}"
            )
            return
        if self._use_custom_fsdp_sharded_save():
            self._save_fsdp_sharded_state(output_dir, include_optimizer=True)
            self._log_osdft_debug(
                f"exit _save_optimizer_and_scheduler output_dir={output_dir} (custom sharded safetensors)"
            )
            return
        if (
            self.is_fsdp_enabled
            and os.environ.get("OVSDFT_FSDP_CONFIG_ONLY_SAVE", "0") == "1"
        ):
            import transformers.trainer as hf_trainer_module

            # Rank 0 may still be copying config/tokenizer side files in `save_model()`.
            # Synchronize before entering the collective FSDP state-dict save so nonzero ranks
            # do not block indefinitely while rank 0 is still outside the save_fsdp_model path.
            self.accelerator.wait_for_everyone()
            fsdp_ckpt_kwargs = (
                hf_trainer_module._get_fsdp_ckpt_kwargs()
                if hasattr(hf_trainer_module, "_get_fsdp_ckpt_kwargs")
                else {}
            )
            self._log_osdft_debug(
                f"config_only save FSDP full-state model only output_dir={output_dir} kwargs={fsdp_ckpt_kwargs}"
            )
            hf_trainer_module.save_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model,
                output_dir,
                **fsdp_ckpt_kwargs,
            )
            self._log_osdft_debug(
                f"exit _save_optimizer_and_scheduler output_dir={output_dir} (config_only model-only)"
            )
            return
        result = super()._save_optimizer_and_scheduler(output_dir)
        self._log_osdft_debug(f"exit _save_optimizer_and_scheduler output_dir={output_dir}")
        return result

    def _save_rng_state(self, output_dir):
        if (
            self.is_fsdp_enabled
            and os.environ.get("OVSDFT_FSDP_CONFIG_ONLY_SAVE", "0") == "1"
        ):
            self._log_osdft_debug(f"skip _save_rng_state output_dir={output_dir} (config_only)")
            return
        return super()._save_rng_state(output_dir)

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        self._log_osdft_debug(f"enter _save_checkpoint at step={self.state.global_step}")
        if os.environ.get("OVSDFT_SKIP_SAVE", "0") == "1":
            self._log_osdft_debug(f"skip _save_checkpoint at step={self.state.global_step}")
            return
        if os.environ.get("OVSDFT_PREDELETE_OLD_CHECKPOINTS", "0") == "1":
            run_dir = Path(self._get_output_dir(trial=trial))
            for checkpoint_dir in run_dir.glob(f"{PREFIX_CHECKPOINT_DIR}-*"):
                if checkpoint_dir.is_dir():
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        self._log_osdft_debug(f"before save_model at step={self.state.global_step}")
        super()._save_checkpoint(model, trial)
        self._log_osdft_debug(f"exit _save_checkpoint at step={self.state.global_step}")
        if not getattr(self.args, "latest_checkpoint_keep_optimizer", False):
            return
        if getattr(self.args, "keep_optimizer_for_all_checkpoints", False):
            return
        run_dir = Path(self._get_output_dir(trial=trial))
        latest_checkpoint_dir = run_dir / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        if latest_checkpoint_dir.is_dir():
            self._prune_old_checkpoint_training_state(run_dir, latest_checkpoint_dir)

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        self._log_osdft_debug(
            f"enter _maybe_log_save_evaluate step={self.state.global_step} should_log={self.control.should_log} should_save={self.control.should_save}"
        )
        if self.control.should_log and grad_norm is not None:
            self._append_scalar_metric("train", "grad_norm", grad_norm)
        result = super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )
        self._log_osdft_debug(f"exit _maybe_log_save_evaluate step={self.state.global_step}")
        return result
