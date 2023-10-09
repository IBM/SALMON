# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import os
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import logging

import torch
import transformers
import argparse
from transformers import set_seed

try:
    from transformers import LlamaTokenizerFast as LlamaTokenizer

    print("Using fast tokenizer")
except:
    from transformers import LlamaTokenizer

    print("Using slow tokenizer")

from transformers import AutoTokenizer, AutoModelForCausalLM

from qlora_utils import (
    SavePeftModelCallback,
    print_trainable_parameters,
    get_last_checkpoint,
    DEFAULT_PAD_TOKEN,
)
from data_utils.data_utils_rm import make_binary_reward_modeling_data_module
from models.reward_model import (
    RewardConfig,
    RewardModel,
    RewardModelTrainer as Trainer,
    compute_reward_modeling_metrics,
)

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default=None, metadata={"help": "Dataset name"})
    eval_dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    eval_dataset_name: str = field(default="alpaca_human_preference")
    eval_size: int = field(
        default=500,
        metadata={
            "help": "Number of examples to split out from training to use for evaluation."
        },
    )
    meta_prompt_pattern: Optional[str] = field(
        default=None, metadata={"help": "Which meta prompt pattern to use."}
    )
    principle_collection_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the principle collection."}
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # From AlpacaFarm
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    query_len: int = field(default=None, metadata={"help": "Length of the query."})
    response_len: int = field(
        default=None, metadata={"help": "Length of the response."}
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    end_sequence_with_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to end sequences with EOS. "
            "Ending with EOS might help the reward model realize it's time to predict."
        },
    )
    # From QLoRA
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Which modules to use LoRA on. If None, will use all linear layers."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoint to resume."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_training: bool = field(
        default=False, metadata={"help": "Resume from training"}
    )


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    if args.resume_dir is not None:
        checkpoint_dir, completed_training = args.resume_dir, False
    else:
        checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)

    if completed_training:
        rank0_print("Detected that training was already completed!")

    if checkpoint_dir is None:
        rank0_print("Training from scratch.")
    else:
        rank0_print("Loading from checkpoint:", checkpoint_dir)
        if args.resume_from_training:
            rank0_print("Resuming from training not supported yet. Exiting.")
            exit(1)

    use_llama_base_model = (
        "dromedary" in args.model_name_or_path.lower()
        or "llama" in args.model_name_or_path.lower()
        or "vicuna" in args.model_name_or_path.lower()
    )

    if use_llama_base_model:
        tokenizer_model_name = (
            "TheBloke/dromedary-65b-lora-HF"  # TODO(zhiqings): hacking
        )
        TokenizerClass = LlamaTokenizer
    else:
        tokenizer_model_name = args.model_name_or_path
        TokenizerClass = AutoTokenizer

    # Tokenizer
    tokenizer = TokenizerClass.from_pretrained(
        tokenizer_model_name,
        cache_dir=args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
    )

    if use_llama_base_model:
        if tokenizer._pad_token is None:
            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
    else:
        raise NotImplementedError

    data_module = make_binary_reward_modeling_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    if args.do_train:
        training_data = data_module["train_dataset"]
        rank0_print("Training data size:", len(training_data))
        rank0_print("Training data example:")
        for i in range(min(3, len(training_data))):
            ex_input_ids_0 = training_data[i]["input_ids"][0]
            rank0_print(tokenizer.decode(ex_input_ids_0, skip_special_tokens=True))
            rank0_print("=" * 20)
            ex_input_ids_1 = training_data[i]["input_ids"][1]
            rank0_print(tokenizer.decode(ex_input_ids_1, skip_special_tokens=True))
            rank0_print("=" * 20)
            rank0_print("=" * 20)

    config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)

    model = RewardModel(
        args=args,
        config=config,
        qlora=True,
        checkpoint_dir=checkpoint_dir,
    )

    model.backbone_model.config.use_cache = False
    print_trainable_parameters(args, model)
    print("loaded model")
    set_seed(args.seed)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_reward_modeling_metrics,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if args.do_train or args.do_eval:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
