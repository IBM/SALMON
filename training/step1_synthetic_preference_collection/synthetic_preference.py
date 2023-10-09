import fcntl
import glob
import json
import os
import random

import numpy as np
import torch
import tqdm
import fire

from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    LlamaTokenizerFast,
)
from datasets import load_dataset


def main(
    model_name: str,
    adapters_name: str,
    preferece_prompt: str,
    rm_principles: str,
    response_pattern: str,
    output_file: str,
    tokenizer_name: str = "TheBloke/dromedary-65b-lora-HF",  # a random llama-based model
):
    with open(preferece_prompt, "r") as f:
        PREFERENCE_META_PROMPT = f.read().strip()

    with open(rm_principles, "r") as f:
        principle_definitions = json.load(f)

    data_files = glob.glob(response_pattern)
    cache = ""
    raw_data = {}
    for data_file in data_files:
        type = data_file.split("/")[-1].split("_")[-1].split(".")[0]
        raw_data[type] = []
        with open(data_file, "r") as f:
            for line in f:
                try:
                    raw_data[type].append(json.loads(cache + line))
                    cache = ""
                except json.decoder.JSONDecodeError:
                    cache += line.strip()
        raw_data[type] = sorted(raw_data[type], key=lambda x: x["instruction"])

    print(raw_data.keys())

    preference_dataset = []

    for type_idx_a in range(len(raw_data.keys())):
        for type_idx_b in range(type_idx_a + 1, len(raw_data.keys())):
            type_a = list(raw_data.keys())[type_idx_a]
            type_b = list(raw_data.keys())[type_idx_b]
            print(
                f"len {type_a} and {type_b}",
                len(raw_data[type_a]),
                len(raw_data[type_b]),
            )

            for idx in range(len(raw_data[type_a])):
                data_a = raw_data[type_a][idx]
                data_b = raw_data[type_b][idx]

                if data_a["instruction"] != data_b["instruction"]:
                    print("Instruction mismatch!")
                    print(data_a["instruction"])
                    print(data_b["instruction"])
                    exit(0)

                if (
                    len(data_a["output"]["generation"]) < 16
                    or len(data_b["output"]["generation"]) < 16
                ):
                    continue

                preference_dataset.append(
                    {
                        "instruction": data_a["instruction"],
                        "input": data_a["input"],
                        "output_1": data_a["output"]["generation"]
                        .split("###")[0]
                        .strip(),
                        "output_2": data_b["output"]["generation"]
                        .split("###")[0]
                        .strip(),
                        "preference": 0,
                    }
                )

    random.Random(42).shuffle(preference_dataset)

    print(f"Starting to load the model {model_name} into memory")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)

    print(preference_dataset[0 + rank])

    batch_size = 4  # For a single 80GB GPU
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    m = PeftModel.from_pretrained(
        m,
        adapters_name,
        adapter_name="default",
        is_trainable=False,
    )
    tok = LlamaTokenizerFast.from_pretrained(
        tokenizer_name,
        padding_side="left",
        truncation_side="left",
        model_max_length=1536,
    )
    tok.pad_token_id = 0

    print(f"Successfully loaded the model {model_name} into memory")
    print_flag = True

    idxs = []
    dimensions = []
    prompts = []
    preferences = []

    for idx in tqdm.tqdm(range(len(preference_dataset))):
        if idx % world_size != rank:
            continue

        for principle_definition in principle_definitions:
            dimension = principle_definition["dimension"]
            definition = principle_definition["definition"]

            data = preference_dataset[idx]
            instruction = data["instruction"]
            instruction_input = data["input"]

            if instruction_input:
                instruction += "\n\n" + instruction_input

            output_1 = data["output_1"]
            output_2 = data["output_2"]
            preference = data["preference"]
            preferences.append(preference)
            idxs.append(idx)
            dimensions.append(dimension)

            for a, b in [(output_1, output_2), (output_2, output_1)]:
                prompt_for_score = PREFERENCE_META_PROMPT.format(
                    UserInstruction=instruction,
                    OutputA=a,
                    OutputB=b,
                    Dimension=dimension,
                    Definition=definition,
                )

                if print_flag:
                    print_flag = False
                    print(prompt_for_score)

                prompts.append(tok.bos_token + prompt_for_score)

            if len(prompts) == batch_size * 2:
                tokenized_input = tok(
                    prompts,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding="max_length",
                    return_attention_mask=True,
                    truncation=True,
                )

                input_ids = tokenized_input["input_ids"].to(m.device)
                attention_mask = tokenized_input["attention_mask"].to(m.device)

                with torch.inference_mode():
                    output = m(
                        input_ids,
                        attention_mask=attention_mask,
                    )

                logits = output.logits

                token_id_a = tok.encode("\n (a", add_special_tokens=False)[-1]
                token_id_b = tok.encode("\n (b", add_special_tokens=False)[-1]

                relative_scores = []
                prompt_preferences = []

                for ex_idx in range(batch_size):
                    score_a_for_1_2 = logits[ex_idx * 2 + 0, -1, token_id_a]
                    score_b_for_1_2 = logits[ex_idx * 2 + 0, -1, token_id_b]
                    score_a_for_2_1 = logits[ex_idx * 2 + 1, -1, token_id_a]
                    score_b_for_2_1 = logits[ex_idx * 2 + 1, -1, token_id_b]

                    relative_score_1_2 = (score_a_for_1_2 - score_b_for_1_2).item()
                    relative_score_2_1 = (score_b_for_2_1 - score_a_for_2_1).item()

                    if relative_score_1_2 > 0.0 and relative_score_2_1 > 0.0:
                        prompt_preference = 1
                    elif relative_score_1_2 < 0.0 and relative_score_2_1 < 0.0:
                        prompt_preference = 2
                    else:
                        prompt_preference = 0

                    relative_scores.append((relative_score_1_2, relative_score_2_1))
                    prompt_preferences.append(prompt_preference)

                outputs = []

                for ex_idx in range(batch_size):
                    outputs.append(
                        {
                            "example_idx": idxs[ex_idx],
                            "dimension": dimensions[ex_idx],
                            "preference": preferences[ex_idx],
                            "prompt_preference": prompt_preferences[ex_idx],
                            "relative_score": relative_scores[ex_idx],
                        }
                    )

                with open(output_file, "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    for output in outputs:
                        f.write(json.dumps(output) + "\n")
                    fcntl.flock(f, fcntl.LOCK_UN)

                idxs = []
                dimensions = []
                prompts = []
                preferences = []


if __name__ == "__main__":
    fire.Fire(main)
