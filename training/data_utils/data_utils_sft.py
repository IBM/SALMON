# Copyright 2023 The Self-Align Team
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

import copy

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import pandas as pd

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset

import data_utils.common_utils as utils


IGNORE_INDEX = -100


@dataclass
class DataCollatorForCausalLM(object):
    left_truncated_tokenizer: transformers.PreTrainedTokenizer
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool
    add_eos_to_target: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [example["input"] for example in instances]
        if self.add_eos_to_target:
            targets = [
                f"\n{example['output']}{self.tokenizer.eos_token}"
                for example in instances
            ]
        else:
            targets = [f"\n{example['output']}" for example in instances]

        begin_padding_len = self.tokenizer(
            ["\n"], return_tensors="pt", add_special_tokens=False
        ).input_ids.shape[1]

        # Tokenize
        tokenized_sources_with_prompt = self.left_truncated_tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            # add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len + begin_padding_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            tokenized_target = tokenized_target[begin_padding_len:]
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(
                        torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                    )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        "input": [],
        "output": [],
    }
    for example_instances in examples["instances"]:
        for instance in example_instances:
            out["input"].append(instance["instruction_with_input"])
            out["output"].append(instance["output"])
    if extract_reformulations:
        for example_reformulations in examples["reformulations"]:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out["input"].append(instance["instruction_with_input"])
                    out["output"].append(instance["output"])
    return out


DROMEDARY_PROMPT_DICT = {
    "prompt_input": (
        "{meta_prompt}\n" "{instruction}\n\n" "{input}\n\n" "### Dromedary"
    ),
    "prompt_no_input": ("{meta_prompt}\n" "{instruction}\n\n" "### Dromedary"),
}


def extract_dromedary_dataset(example, meta_prompts):
    assert "example_id" in example
    total_meta_prompt = len(meta_prompts)
    meta_prompt = meta_prompts[int(example["example_id"]) % total_meta_prompt]

    if example.get("input", "") != "":
        prompt_format = DROMEDARY_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = DROMEDARY_PROMPT_DICT["prompt_no_input"]

    return {
        "input": prompt_format.format(meta_prompt=meta_prompt, **example),
        "output": "\n" + example["output"],
    }


def local_dataset(dataset_name):
    if dataset_name.endswith(".json"):
        full_dataset = load_dataset("json", data_files=dataset_name)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    return full_dataset


def make_sft_data_module(
    left_truncated_tokenizer: transformers.PreTrainedTokenizer,
    tokenizer: transformers.PreTrainedTokenizer,
    args,
) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }
    """

    def load_data(dataset_name):
        if dataset_name == "alpaca":
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == "alpaca-clean":
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == "chip2":
            return load_dataset("laion/OIG", data_files="unified_chip2.jsonl")
        elif dataset_name == "self-instruct":
            return load_dataset("yizhongw/self_instruct", name="self_instruct")
        elif dataset_name == "hh-rlhf":
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == "longform":
            return load_dataset("akoksal/LongForm")
        elif dataset_name == "oasst1":
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == "vicuna":
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = (
                        args.dataset_format if args.dataset_format else "alpaca"
                    )
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(
                    f"Dataset {dataset_name} not implemented yet."
                )

    def format_dataset(dataset, dataset_format, meta_prompt_pattern):
        if dataset_format == "dromedary":
            assert meta_prompt_pattern is not None

            meta_prompts = utils.make_meta_prompts(meta_prompt_pattern)

            dataset = dataset.map(
                lambda ex: extract_dromedary_dataset(ex, meta_prompts=meta_prompts),
                remove_columns=["instruction", "example_id"],
            )
        elif dataset_format == "chip2" or (
            dataset_format is None and args.dataset == "chip2"
        ):
            dataset = dataset.map(
                lambda x: {
                    "input": x["text"].split("\n<bot>: ")[0].replace("<human>: ", ""),
                    "output": x["text"].split("\n<bot>: ")[1],
                }
            )
        elif dataset_format == "self-instruct" or (
            dataset_format is None and args.dataset == "self-instruct"
        ):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == "hh-rlhf" or (
            dataset_format is None and args.dataset == "hh-rlhf"
        ):
            dataset = dataset.map(lambda x: {"input": "", "output": x["chosen"]})
        elif dataset_format == "oasst1" or (
            dataset_format is None and args.dataset == "oasst1"
        ):
            dataset = dataset.map(
                lambda x: {
                    "input": "",
                    "output": x["text"],
                }
            )
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names["train"]
                if col not in ["input", "output"]
            ]
        )
        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format, args.meta_prompt_pattern)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        else:
            print(
                "Splitting train dataset in train and validation according to `eval_dataset_size`"
            )
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset["test"]
        if (
            args.max_eval_samples is not None
            and len(eval_dataset) > args.max_eval_samples
        ):
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )
    if args.do_train:
        train_dataset = dataset["train"]
        if (
            args.max_train_samples is not None
            and len(train_dataset) > args.max_train_samples
        ):
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )

    data_collator = DataCollatorForCausalLM(
        left_truncated_tokenizer=left_truncated_tokenizer,
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
        add_eos_to_target=args.add_eos_to_target,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )
