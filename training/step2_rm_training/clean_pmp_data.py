import json
import random
import tqdm
from datasets import load_dataset
import fire


def find_shared_prefix(str1, str2):
    """
    Find the shared prefix between two strings.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        int: The index position where the shared prefix ends in the first string.
    """
    # Find the minimum length of both strings
    min_length = min(len(str1), len(str2))

    max_common_prefix = 0
    # Loop through characters to find where they start to differ
    for i in range(min_length):
        if str1[i] != str2[i]:
            max_common_prefix = i
            break

    while (
        not str1[:max_common_prefix].endswith("Assistant:")
        # Force the prefix to end with "Assistant:"
    ) and max_common_prefix > 0:
        max_common_prefix -= 1

    return max_common_prefix


def split_example(example):
    """
    Split an example into a shared prefix and the remaining parts of the strings.

    Args:
        example (dict): A dictionary containing 'chosen' and 'rejected' keys with strings as values.

    Returns:
        dict: A dictionary with keys 'query', 'output_1', and 'output_2'.
    """
    chosen = example["chosen"]
    rejected = example["rejected"]

    # Find the index where the shared prefix ends
    shared_index = find_shared_prefix(chosen, rejected)

    # Split the strings
    query = chosen[:shared_index].strip()
    output_1 = chosen[shared_index:].strip()
    output_2 = rejected[shared_index:].strip()

    # Return the result as a dictionary
    return {"query": query, "output_1": output_1, "output_2": output_2}


def main(
    output_file: str,
):
    shp_dataset = load_dataset("stanfordnlp/SHP")["train"]

    merged_data = []

    for ex in tqdm.tqdm(shp_dataset):
        qid = ex["post_id"]
        score = ex["score_ratio"]

        ex_dp = {
            "score": score,
            "instruction": ex["history"],
            "input": "",
            "output_1": ex["human_ref_A"],
            "output_2": ex["human_ref_B"],
            "preference": 1 if ex["labels"] == 1 else 2,
        }

        if score > 2.0:
            merged_data.append(ex_dp)

    print(merged_data[-1])
    print(len(merged_data))

    hh_dataset = load_dataset("Anthropic/hh-rlhf")["train"]

    for ex in hh_dataset:
        processed_ex = split_example(ex)

        random_swap = random.random() < 0.5

        merged_data.append(
            {
                "instruction": processed_ex["query"].strip(),
                "input": "",
                "output_1": processed_ex["output_1"]
                if random_swap
                else processed_ex["output_2"],
                "output_2": processed_ex["output_2"]
                if random_swap
                else processed_ex["output_1"],
                "preference": 1 if random_swap else 2,
            }
        )

    print(merged_data[-1])
    print(len(merged_data))

    random.shuffle(merged_data)

    with open(output_file, "w") as f:
        f.write(
            json.dumps(
                merged_data,
            )
        )


if __name__ == "__main__":
    fire.Fire(main)
