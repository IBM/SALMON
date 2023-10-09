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

from dataclasses import dataclass
import json
import logging
from typing import Callable, Optional, Dict, Sequence, List, Tuple, Union

import tqdm

import einops
import pandas as pd
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

import data_utils.common_utils as utils

IGNORE_INDEX = -100


logger = logging.getLogger(__name__)


BASE_PROMPT_DICT = {
    "prompt_input": "{instruction}\n\n{input}",
    "prompt_no_input": "{instruction}",
}


def format_prompt(
    example: Dict[str, str],
    meta_prompts: List[str],
) -> str:
    assert (
        "instruction" in example and "input" in example
    ), "Internal error: example missing required keys."

    if "example_id" in example:
        total_meta_prompt = len(meta_prompts)
        meta_prompt = meta_prompts[int(example["example_id"]) % total_meta_prompt]
    else:
        meta_prompt = meta_prompts[0]

    if example.get("input", "") != "":
        prompt_format = BASE_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = BASE_PROMPT_DICT["prompt_no_input"]

    formatted_input = prompt_format.format(**example)
    meta_prompt = meta_prompt.split("{Output}")[0]

    formatted_prompt = meta_prompt.format(Input=formatted_input)
    return formatted_prompt


def format_output(
    example: dict,
    meta_prompts: List[str],
    eos_token: Optional[str] = None,
    output_key="output",
) -> str:
    if eos_token is None:
        eos_token = ""

    if "example_id" in example:
        total_meta_prompt = len(meta_prompts)
        meta_prompt = meta_prompts[int(example["example_id"]) % total_meta_prompt]
    else:
        meta_prompt = meta_prompts[0]

    meta_prompt = "{Output}" + meta_prompt.split("{Output}")[1]
    formatted_output = meta_prompt.format(Output=example[output_key]) + eos_token
    return formatted_output


def format_full_prompt(
    example: Dict[str, str],
    meta_prompts: List[str],
    principle_collection: Dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    eos_token: Optional[str] = None,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    output_key: str = "output",
    max_principles: Optional[int] = 5,
) -> str:
    if eos_token is None:
        eos_token = ""
    assert (
        "instruction" in example and "input" in example
    ), "Internal error: example missing required keys."

    if "example_id" in example:
        total_meta_prompt = len(meta_prompts)
        meta_prompt = meta_prompts[int(example["example_id"]) % total_meta_prompt]
    else:
        meta_prompt = meta_prompts[0]

    if "{Dimensions}" in meta_prompt:
        assert principle_collection is not None

        if "dimensions" in example:
            dimensions = eval(example["dimensions"])
        else:
            dimensions = list(principle_collection.keys())[:max_principles]

        if "negative_definitions" in example:
            negative_definitions = eval(example["negative_definitions"])
        else:
            negative_definitions = [0] * len(dimensions)

        principle_str = ""
        for neg_flag, dim in zip(negative_definitions, dimensions):
            if neg_flag:
                principle_str += f"- {principle_collection[dim + '_neg']}\n"
            else:
                principle_str += f"- {principle_collection[dim]}\n"
        principle_str = principle_str.strip()
        meta_prompt = meta_prompt.replace("{Dimensions}", principle_str)

    if example.get("input", "") != "":
        prompt_format = BASE_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = BASE_PROMPT_DICT["prompt_no_input"]

    formatted_input = prompt_format.format(**example)

    if query_len is not None:
        ori_truncation_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        formatted_input_tokens = tokenizer(
            text=formatted_input,
            max_length=query_len,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        formatted_input = tokenizer.batch_decode(
            formatted_input_tokens["input_ids"], skip_special_tokens=True
        )[0].strip()
        tokenizer.truncation_side = ori_truncation_side

    if response_len is not None:
        ori_truncation_side = tokenizer.truncation_side
        tokenizer.truncation_side = "right"
        formatted_output_tokens = tokenizer(
            text=example[output_key],
            max_length=response_len,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        formatted_output = tokenizer.batch_decode(
            formatted_output_tokens["input_ids"], skip_special_tokens=True
        )[0].strip()
        tokenizer.truncation_side = ori_truncation_side
    else:
        formatted_output = example[output_key]

    formatted_prompt = (
        meta_prompt.format(
            Input=formatted_input,
            Output=formatted_output,
        )
        + eos_token
    )

    return formatted_prompt


def format_prompt_with_data_frame(
    df: pd.DataFrame,
    meta_prompts: dict,
    df_postprocessor: Optional[Callable] = None,
    return_dict=False,
):
    if df_postprocessor is not None:
        df = df_postprocessor(df)
    list_dict_data = df.to_dict(orient="records")

    prompts = [format_prompt(example, meta_prompts) for example in list_dict_data]
    metadata = {"prompt_dict": meta_prompts}

    if return_dict:
        return dict(prompts=prompts, list_dict_data=list_dict_data, metadata=metadata)
    return prompts, list_dict_data, metadata


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    use_data_frame: bool = True,
) -> dict:
    """Tokenize a list of strings and return the tokenized content as well metadata (e.g., truncation statistics)."""
    padding = getattr(tokenizer, "padding", "max_length")
    return_overflowing_tokens = False

    if use_data_frame:
        strings_ds = HFDataset.from_dict({"full_prompt": strings})
    else:
        strings_ds = strings

    def _tokenize(x):
        tokenized_text = tokenizer(
            x["full_prompt"],
            return_tensors="pt",
            padding=padding,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=return_overflowing_tokens,
        )
        return {
            "input_ids": tokenized_text.input_ids,
            "attention_mask": tokenized_text.attention_mask,
            "input_length": tokenized_text.attention_mask.sum(),
        }

    tokenized_list = strings_ds.map(
        lambda x: _tokenize(x),
        num_proc=16,
    )
    # logger.warning(f"Example of tokenized text: {tokenized_list[0]['tokenized_text']}")

    tokenized_list.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "input_length"],
    )
    if padding == "max_length":
        input_ids = labels = torch.cat(
            [
                tokenized["input_ids"]
                for tokenized in tqdm.tqdm(
                    tokenized_list,
                    desc="Concatenating input_ids",
                )
            ]
        )
    else:  # "longest"
        input_ids = labels = [
            tokenized["input_ids"][0]
            for tokenized in tqdm.tqdm(
                tokenized_list,
                desc="Concatenating input_ids",
            )
        ]

    logger.warning(
        "You are using a `transformers` version that does not support `return_overflowing_tokens=True`. "
        "The tokenization metadata will not be recorded."
        "In order to see truncation statistics, please downgrade to `transformers<=4.26.1`."
    )
    input_ids_lens = labels_lens = [
        tokenized["input_length"].item()
        for tokenized in tqdm.tqdm(tokenized_list, desc="Computing lengths")
    ]
    num_truncated_tokens = num_truncated_examples = -1

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        tokenization_metadata=dict(
            num_examples=len(tokenized_list),
            num_truncated_tokens=num_truncated_tokens,
            num_truncated_examples=num_truncated_examples,
            input_ids_avg_len=utils.mean(input_ids_lens),
            input_ids_max_len=max(input_ids_lens),
            input_ids_min_len=min(input_ids_lens),
            labels_avg_len=utils.mean(labels_lens),
            labels_max_len=max(labels_lens),
            labels_min_len=min(labels_lens),
            model_max_length=tokenizer.model_max_length,
        ),
    )


def preprocess_for_reward_modeling(
    data: Sequence[dict],
    meta_prompts: List[str],
    principle_collection: Dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    df_postprocessor: Optional[Callable] = None,
    end_sequence_with_eos: bool = False,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    use_data_frame: bool = True,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    if use_data_frame:
        df = pd.DataFrame(data)
        if df_postprocessor is not None:
            df = df_postprocessor(df)
        list_dict_data = df.to_dict(orient="records")
    else:
        list_dict_data = data

    index_0, index_1 = tuple(
        torch.full(
            size=(len(list_dict_data), 1), fill_value=fill_value, dtype=torch.long
        )
        for fill_value in (0, 1)
    )

    def _get_numeric_preference(example: dict):
        # 1 vs 2 is stored in table, but for modeling we use 0 vs 1; remap here.
        return {1: 0, 2: 1}[example["preference"]]

    choice = torch.tensor(
        [[_get_numeric_preference(dict_data)] for dict_data in list_dict_data]
    )

    def _get_text(example: dict, output_key: str):
        full_prompt = format_full_prompt(
            example,
            meta_prompts=meta_prompts,
            principle_collection=principle_collection,
            tokenizer=tokenizer,
            eos_token=tokenizer.eos_token if end_sequence_with_eos else None,
            output_key=output_key,
            query_len=query_len,
            response_len=response_len,
        )
        return full_prompt

    if use_data_frame:
        text_list_0, text_list_1 = tuple(
            [_get_text(dict_data, key) for dict_data in list_dict_data]
            for key in ("output_1", "output_2")
        )
    else:
        text_list_0 = list_dict_data.map(
            lambda example: {
                "full_prompt": _get_text(example, "output_1"),
            },
            num_proc=16,
        )
        text_list_1 = list_dict_data.map(
            lambda example: {
                "full_prompt": _get_text(example, "output_2"),
            },
            num_proc=16,
        )

    def _merge_tokenization_metadata(metadata_list: Sequence[dict]) -> dict:
        num_examples = sum(metadata["num_examples"] for metadata in metadata_list)
        num_truncated_tokens = sum(
            metadata["num_truncated_tokens"] for metadata in metadata_list
        )
        num_truncated_examples = sum(
            metadata["num_truncated_examples"] for metadata in metadata_list
        )
        input_ids_avg_lens = (
            sum(
                [
                    metadata["input_ids_avg_len"] * metadata["num_examples"]
                    for metadata in metadata_list
                ]
            )
            / num_examples
        )
        input_ids_max_len = max(
            metadata["input_ids_max_len"] for metadata in metadata_list
        )
        input_ids_min_len = min(
            metadata["input_ids_min_len"] for metadata in metadata_list
        )
        labels_avg_lens = (
            sum(
                [
                    metadata["labels_avg_len"] * metadata["num_examples"]
                    for metadata in metadata_list
                ]
            )
            / num_examples
        )
        labels_max_len = max(metadata["labels_max_len"] for metadata in metadata_list)
        labels_min_len = min(metadata["labels_min_len"] for metadata in metadata_list)
        return dict(
            num_examples=num_examples,
            num_truncated_tokens=num_truncated_tokens,
            num_truncated_examples=num_truncated_examples,
            input_ids_avg_len=input_ids_avg_lens,
            input_ids_max_len=input_ids_max_len,
            input_ids_min_len=input_ids_min_len,
            labels_avg_len=labels_avg_lens,
            labels_max_len=labels_max_len,
            labels_min_len=labels_min_len,
        )

    logger.warning(f"Tokenizing {len(list_dict_data)} pairs...")
    tokenized_0, tokenized_1 = tuple(
        _tokenize_fn(text_list, tokenizer, use_data_frame)
        for text_list in (text_list_0, text_list_1)
    )
    # "size" (bsz, 2, seq_len)
    input_ids = [
        list(pair)
        for pair in utils.zip_(tokenized_0["input_ids"], tokenized_1["input_ids"])
    ]
    labels = [
        list(pair) for pair in utils.zip_(tokenized_0["labels"], tokenized_1["labels"])
    ]
    tokenization_metadata = _merge_tokenization_metadata(
        [tokenized_0["tokenization_metadata"], tokenized_1["tokenization_metadata"]]
    )

    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        index_0=index_0,
        index_1=index_1,
        choice=choice,
        tokenization_metadata=tokenization_metadata,
        metadata=dict(mean_choice=choice.float().mean().item()),
    )
    if verbose:
        logger.warning(
            f"Tokenization metadata:\n{json.dumps(packaged_data['tokenization_metadata'])}"
        )

    return packaged_data


class BinaryRewardModelingDataset(Dataset):
    def __init__(
        self,
        data: Sequence[dict],
        meta_prompts: List[str],
        principle_collection: Dict[str, str],
        tokenizer: transformers.PreTrainedTokenizer,
        df_postprocessor: Optional[Callable] = None,
        end_sequence_with_eos: bool = False,
        query_len: Optional[int] = None,
        response_len: Optional[int] = None,
        use_data_frame: bool = True,
    ):
        super(BinaryRewardModelingDataset, self).__init__()
        data_dict = preprocess_for_reward_modeling(
            data=data,
            meta_prompts=meta_prompts,
            principle_collection=principle_collection,
            tokenizer=tokenizer,
            df_postprocessor=df_postprocessor,
            end_sequence_with_eos=end_sequence_with_eos,
            query_len=query_len,
            response_len=response_len,
            use_data_frame=use_data_frame,
        )
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.index_0 = data_dict["index_0"]
        self.index_1 = data_dict["index_1"]
        self.choice = data_dict["choice"]
        self.metadata = data_dict["metadata"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            index_0=self.index_0[i],
            index_1=self.index_1[i],
            choice=self.choice[i],
        )


def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def split_train_into_train_and_eval(
    train_dataset: Dataset, eval_size: int, seed: int
) -> Tuple[Dataset, Dataset]:
    assert eval_size < len(
        train_dataset  # noqa
    ), "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size  # noqa
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset


def pad_sequence_from_left(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    )  # noqa
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence


@dataclass
class DataCollatorForBinaryRewardModelingDataset(object):
    """
    This collation assumes data preprocessing converts text into *padded* tensors of the same length.
    For autoregressive models like OPT and GPT2, `input_ids` alone is sufficient to produce the rewards.
    For enc-dec models like T5, we need `labels`.

    `input_ids` and `labels` are tensors of size (bsz, num_candidates, max_seq_len), i.e., each batch instance has
    `num_candidates` generations/completions.
    `index_0` and `index_1` are tensors of size (bsz, num_pairs), and are used to index into `input_ids` and
    `labels` to find the first and second sequences in the pair.
    `choice` is a binary int/long tensor of size (bsz, num_pairs) indicating which sequence in the pair is better,
    i.e., 0 means the first sequence is preferred, and 1 means otherwise.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def _left_pad_helper(self, instances: Sequence[dict], key: str):
        # TODO(lxuechen): Potentially replace with `transformers.PretrainedTokenizerBase.prepare_for_model`.
        # `instances` is a list of dicts, each dict has key whose value is a list of tensors, possibly of unequal length.
        input_ids = [seq for instance in instances for seq in instance[key]]  # Flatten.
        input_ids = pad_sequence_from_left(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = einops.rearrange(
            input_ids,
            "(bsz num_candidates) max_seq_len -> bsz num_candidates max_seq_len",
            num_candidates=len(instances[0][key]),
        )
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        index_0, index_1, choice = tuple(
            torch.stack([instance[key] for instance in instances])
            for key in ("index_0", "index_1", "choice")
        )
        input_ids = self._left_pad_helper(instances, "input_ids")
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            index_0=index_0,
            index_1=index_1,
            choice=choice,
        )


def make_binary_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    meta_prompts = utils.make_meta_prompts(data_args.meta_prompt_pattern)

    principle_collection = None
    if data_args.principle_collection_path:
        principle_collection = {}
        with open(data_args.principle_collection_path, "r", encoding="utf-8") as f:
            principle_collection_list = json.load(f)
            for item in principle_collection_list:
                principle_collection[item["dimension"]] = item["definition"]
                if "negative_definition" in item:
                    principle_collection[item["dimension"] + "_neg"] = item[
                        "negative_definition"
                    ]

    if "alpaca_farm" in data_args.dataset_path:
        train_preference = load_dataset(data_args.dataset_path, data_args.dataset_name)[
            "preference"
        ]
        use_data_frame = True
    elif data_args.dataset_path.endswith("json"):
        train_preference = load_dataset("json", data_files=data_args.dataset_path)[
            "train"
        ]
        use_data_frame = False
    else:
        raise ValueError(
            f"Unsupported dataset_path: {data_args.dataset_path}."
            "Only json and alpaca_farm datasets are supported."
        )

    train_dataset = BinaryRewardModelingDataset(
        data=train_preference,
        meta_prompts=meta_prompts,
        principle_collection=principle_collection,
        tokenizer=tokenizer,
        end_sequence_with_eos=training_args.end_sequence_with_eos,
        query_len=training_args.query_len,
        response_len=training_args.response_len,
        use_data_frame=use_data_frame,
    )

    if (
        data_args.dataset_path == data_args.eval_dataset_path
        and data_args.dataset_name == data_args.eval_dataset_name
    ):
        train_dataset, eval_dataset = split_train_into_train_and_eval(
            train_dataset=train_dataset,
            eval_size=data_args.eval_size,
            seed=training_args.seed,
        )

    else:
        alpaca_human_preference = load_dataset(
            data_args.eval_dataset_path, data_args.eval_dataset_name
        )["preference"]
        eval_dataset = BinaryRewardModelingDataset(
            data=alpaca_human_preference,
            meta_prompts=meta_prompts,
            principle_collection=principle_collection,
            tokenizer=tokenizer,
            end_sequence_with_eos=training_args.end_sequence_with_eos,
            query_len=training_args.query_len,
            response_len=training_args.response_len,
        )
        _, eval_dataset = split_train_into_train_and_eval(
            train_dataset=eval_dataset,
            eval_size=data_args.eval_size,
            seed=training_args.seed,
        )

    data_collator = DataCollatorForBinaryRewardModelingDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
