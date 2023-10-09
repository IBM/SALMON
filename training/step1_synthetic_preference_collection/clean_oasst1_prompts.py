import random
import tqdm
import re
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import fire


def load_oasst_data():
    oasst_dataset = load_dataset("OpenAssistant/oasst1")["train"]

    def create_message_trees(dataset):
        """Create message trees from dataset."""
        # Organize data into dictionary based on parent_id
        organized_data = {}
        for message in dataset:
            parent_id = message["parent_id"]
            if parent_id not in organized_data:
                organized_data[parent_id] = []
            organized_data[parent_id].append(message)

        # Traverse data to create trees
        message_trees = []
        for root_messages in organized_data[None]:
            tree = []
            current_message = root_messages
            while current_message is not None:
                tree.append(current_message)
                children = organized_data.get(current_message["message_id"])
                current_message = children[0] if children else None
            message_trees.append(tree)

        return message_trees

    oasst_message_trees = create_message_trees(oasst_dataset)
    oasst_examples = []

    count = 0
    for oasst_tree in oasst_message_trees:
        if len(oasst_tree) >= 2:
            count += 1
            oasst_examples.append(
                {
                    "instruction": oasst_tree[0]["text"],
                    "input": "",
                    "output": "",
                }
            )
    print("OASST examples:", count)
    return oasst_examples


def main(
    output_file: str = "/path/to/oasst1_prompts.json",
):
    oasst_examples = load_oasst_data()

    with open(output_file, "w") as f:
        json.dump(oasst_examples, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
