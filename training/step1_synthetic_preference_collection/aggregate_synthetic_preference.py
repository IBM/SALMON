import fcntl
import glob
import json
import os
import random

import numpy as np
import tqdm
from ftlangdetect import detect


ALL_DIMENSIONS = [
    # "Overall Helpful",
    "Concise",
    "Honest and Accurate",
    "Ethical",
    # "Natural and Fluent",
    "Specific",
    "Educational and Engaging",
    "Methodical",
    # "Multilingual",
    "Creative",
    "Comprehensive",
]


def main(
    response_pattern: str,
    preference_file: str,
    output_file: str,
    noisy_ratio: float = 0.1,
    max_principles: int = 8,
    tokenizer_name: str = "TheBloke/dromedary-65b-lora-HF",  # a random llama-based model
):
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

    print(raw_data.keys(), len(raw_data[type]))
    preference_dataset = []

    for type_idx_a in range(len(raw_data.keys())):
        for type_idx_b in range(type_idx_a + 1, len(raw_data.keys())):
            type_a = list(raw_data.keys())[type_idx_a]
            type_b = list(raw_data.keys())[type_idx_b]

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

    for example_idx in range(len(preference_dataset)):
        preference_dataset[example_idx]["example_idx"] = example_idx

    with open(preference_file, "r") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for line in f:
            data = json.loads(line)
            example_idx = data["example_idx"]

            if "dimension_scores" not in preference_dataset[example_idx]:
                preference_dataset[example_idx]["dimension_scores"] = {}

            dimension = data["dimension"]
            score = data["relative_score"]

            preference_dataset[example_idx]["dimension_scores"][dimension] = score
        fcntl.flock(f, fcntl.LOCK_UN)

    all_dimensions = ALL_DIMENSIONS
    print(preference_dataset[0])

    # print variance
    variances = {}
    for dimension in all_dimensions:
        scores = []
        for example_idx in range(len(preference_dataset)):
            if "dimension_scores" not in preference_dataset[example_idx]:
                continue
            if dimension not in preference_dataset[example_idx]["dimension_scores"]:
                continue
            two_scores = preference_dataset[example_idx]["dimension_scores"][dimension]
            scores.append(two_scores[0])
            scores.append(two_scores[1])
        variances[dimension] = np.var(scores)

        print("Variance of {}: {}".format(dimension, np.var(scores)))

    clean_preference_dataset = []

    for example_idx in range(len(preference_dataset)):
        if "dimension_scores" not in preference_dataset[example_idx]:
            continue

        continue_flag = False

        for dimension in all_dimensions:
            if dimension not in preference_dataset[example_idx]["dimension_scores"]:
                continue_flag = True

        if continue_flag:
            continue

        clean_preference_dataset.append(preference_dataset[example_idx])

    print(len(clean_preference_dataset))

    synthetic_data = []

    main_reasons = {}
    for dimension in all_dimensions:
        main_reasons[dimension] = 0

    random.seed(42)

    def mean(a, b):
        return (a + b) / 2.0

    rule_count = 0

    for i in range(4):
        for example_idx in range(len(clean_preference_dataset)):
            random.shuffle(all_dimensions)

            # Compute the probabilities
            num_dimensions = max_principles + 1
            while num_dimensions > max_principles:
                num_dimensions = np.random.geometric(0.2, 1)[0]

            data = clean_preference_dataset[example_idx]

            dimensions = all_dimensions[:num_dimensions]

            scores = []
            for dimension in dimensions:
                two_scores = data["dimension_scores"][dimension]
                if two_scores[0] > 0.0 and two_scores[1] > 0.0:
                    scores.append(
                        mean(two_scores[0], two_scores[1]) / variances[dimension]
                    )
                elif two_scores[0] < 0.0 and two_scores[1] < 0.0:
                    scores.append(
                        mean(two_scores[0], two_scores[1]) / variances[dimension]
                    )
                else:
                    scores.append(0.0)

            # negative_definitions is random benoulli of length num_dimensions
            negative_definitions = np.random.binomial(1, 0.5, num_dimensions)

            scores = [
                -score if negative_definitions[idx] == 1 else score
                for idx, score in enumerate(scores)
            ]

            # The preference is based on the absolute value of the scores.
            doinate_dimension_idx = np.argmax(np.abs(scores)).item()

            score = scores[doinate_dimension_idx]

            if score > 0.0:
                preference = 1
            elif score < 0.0:
                preference = 2
            else:
                continue

            order = random.randint(0, 1)

            rule_flag = False

            # We ban "as an AI language model"
            if "As an AI language model" in data["output_1"]:
                if "As an AI language model" in data["output_2"]:
                    pass
                else:
                    if random.randint(0, 1) > 0.5:
                        preference = 2
                        rule_flag = True
            else:
                if "As an AI language model" in data["output_2"]:
                    if random.randint(0, 1) > 0.5:
                        preference = 1
                        rule_flag = True
                else:
                    pass

            lang_instruction = detect(data["instruction"].replace("\n", " "))["lang"]
            lang_output_1 = detect(data["output_1"].replace("\n", " "))["lang"]
            lang_output_2 = detect(data["output_2"].replace("\n", " "))["lang"]

            # We ban different languages
            if lang_instruction != lang_output_1:
                if lang_instruction != lang_output_2:
                    continue
                else:
                    preference = 2
                    rule_flag = True
            else:
                if lang_instruction != lang_output_2:
                    preference = 1
                    rule_flag = True
                else:
                    pass

            data_tuple = (data["instruction"], data["output_1"], data["output_1"])

            if rule_flag:
                rule_count += 1
                flip = random.random() < noisy_ratio * 2.5  # 2.5 is a magic number
            else:
                flip = random.random() < noisy_ratio
                main_reasons[dimensions[doinate_dimension_idx]] += 1

            preference = 3 - preference if flip else preference

            synthetic_data.append(
                {
                    "doinate_dimension_idx": doinate_dimension_idx,
                    "dimensions": str(tuple(dimensions)),
                    "dimension_scores": str(
                        tuple(scores if order == 0 else [-score for score in scores])
                    ),
                    "negative_definitions": str(tuple(negative_definitions)),
                    "instruction": data["instruction"],
                    "input": data["input"],
                    "output_1": data["output_1"] if order == 0 else data["output_2"],
                    "output_2": data["output_2"] if order == 0 else data["output_1"],
                    "preference": preference if order == 0 else 3 - preference,
                    "flip": flip,
                    "lang": (lang_instruction, lang_output_1, lang_output_2)
                    if order == 0
                    else (lang_instruction, lang_output_2, lang_output_1),
                    "rule_flag": rule_flag,
                }
            )

    print(rule_count)
    print(len(synthetic_data))
    print(main_reasons)

    with open(output_file, "w") as f:
        f.write(
            json.dumps(
                synthetic_data,
                indent=4,
            )
        )
