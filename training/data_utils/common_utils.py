# Copyright 2023 The Alpaca Team
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

import argparse
import glob
import os
import random
from typing import (
    Callable,
    Dict,
    Optional,
    Sequence,
    Union,
    Mapping,
    Any,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

Numeric = Union[int, float]


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)


def mean(*seqs: Sequence[Numeric]) -> Union[Numeric, Sequence[Numeric]]:
    singleton = len(seqs) == 1
    means = [float(np.mean(seq)) for seq in seqs]
    return means[0] if singleton else means


def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.

    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.

    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def flatten_dict(nested, sep=".", postprocess_fn=lambda *args: args):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):  # collections.Mapping fails in py3.10.
                rec(v, prefix + k + sep, into)
            else:
                v = postprocess_fn(v)
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def unpack_dict(
    d: Dict, keys: Sequence[str], return_type: type = tuple
) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def merge_dict(dicts: Sequence[dict], merge_fn: Callable = lambda *args: args) -> dict:
    """Merge a sequence of dicts (with the same set of keys) into a single dict."""
    if len(dicts) == 0:
        return dict()
    return {key: merge_fn([dict_[key] for dict_ in dicts]) for key in dicts[0].keys()}


def prepare_inputs(
    data: Union[torch.Tensor, Any], device: Union[str, int, torch.device]
) -> Union[torch.Tensor, Any]:
    if isinstance(data, Mapping):
        return type(data)(
            {k: prepare_inputs(v, device) for k, v in data.items()}
        )  # noqa
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_inputs(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)  # This can break with deepspeed.
    return data


def compute_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """Compute per-token logprobs, zeroing out places with ignore_index (padding)."""
    return -F.cross_entropy(
        logits.permute(0, 2, 1), labels, reduction="none", ignore_index=ignore_index
    )


def pad(
    inputs: torch.Tensor,
    target_size: Union[torch.Size, Sequence[int]],
    value=0.0,
    left=True,
):
    current_size = inputs.size()
    diffs = tuple(ti - ci for ti, ci in zip_(target_size, current_size))
    pad_params = []
    for diff in diffs:
        pad_params = ([diff, 0] if left else [0, diff]) + pad_params
    res = F.pad(inputs, pad=pad_params, value=value)
    return res


def left_pad(
    inputs: torch.Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0
):
    return pad(inputs=inputs, target_size=target_size, value=value, left=True)


def right_pad(
    inputs: torch.Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0
):
    return pad(inputs=inputs, target_size=target_size, value=value, left=False)


def manual_seed(args_or_seed: Union[int, argparse.Namespace], fix_cudnn=False):
    if hasattr(args_or_seed, "seed"):
        args_or_seed = args_or_seed.seed
    random.seed(args_or_seed)
    np.random.seed(args_or_seed)
    torch.manual_seed(args_or_seed)
    torch.cuda.manual_seed_all(args_or_seed)
    os.environ["PYTHONHASHSEED"] = str(args_or_seed)
    if fix_cudnn:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


def make_meta_prompts(meta_prompt_pattern: str):
    meta_prompt_files = glob.glob(meta_prompt_pattern)
    print(f"Found {len(meta_prompt_files)} meta prompts: {meta_prompt_files}")

    meta_prompts = []
    for meta_prompt_file in meta_prompt_files:
        with open(meta_prompt_file, "r", encoding="utf-8") as f:
            meta_prompt = f.readlines()
        meta_prompt = "".join(meta_prompt).strip()
        meta_prompts.append(meta_prompt)
    return meta_prompts


class InfiniteLoader(object):
    """Wraps an existing loader so that it outputs stuff indefinitely; useful for semi-supervised learning."""

    def __init__(self, loader: DataLoader):
        super(InfiniteLoader, self).__init__()
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)
