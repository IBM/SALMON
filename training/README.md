# Training Experiences

The whole **SALMON** process involves three stages: Synthetic Preference Collection, Training the Principle-following Reward Model, and RL Training with the Principle-following Reward Model. In our [paper](https://arxiv.org/abs/placeholder), we provide a detailed description of each of these stages.

## Prerequisites

For efficiency concerns, we utilize the [model parallel](https://github.com/facebookresearch/fairscale/tree/main/fairscale/nn/model_parallel) scheme from [llama](https://github.com/facebookresearch/llama) when sampling responses from the inital policy model. To prepare the sharded model checkpoints of LLaMA or Dromedary on your own machine/cluster, please refer to the [inference guide in Dromedary](https://github.com/IBM/Dromedary/tree/main/inference).

## Step 1:  Collecting Principle-Driven Synthetic Preference

We sample two responses from the initial policy model, and use the policy model itself to select the preferred response based on a certain human-written principle.

Next, for each user prompt, a subset of principles is randomly sampled from the established principle list with certain principles being randomly negated. The user prompt, model responses, and the sub-sampled principles are aggregated as a single training instance for the reward model.

Before diving into the experiments, please install the [`llama_dromedary` pacakge in Dromedary](https://github.com/IBM/Dromedary/tree/main/llama_dromedary) to enable model parallelism.

### Step 1.1: Preparing OASST1 Prompt Dataset

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
cd step1_synthetic_preference_collection

python -u clean_oasst1_prompts.py \
    --output_file "/path/to/your/oasst1_prompts.json"
```

</details>

### Step 1.2: Sampling Responses from the Policy Model

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
salloc --nodes 8 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/generate_oasst1_response0.sh
salloc --nodes 8 --time 6:00:00 --gres=gpu:32g:6 srun bash scripts/generate_oasst1_response1.sh
```

</details>

### Step 1.3: Collecting Synthetic Preferences

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
salloc --nodes 1 --time 24:00:00 --gres=gpu:80g:8 srun bash scripts/generate_synthetic_preference.sh
```

</details>

### Step 1.4: Aggregating the Collected Preferences

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
python -u aggregate_synthetic_preference.py \
    --response_pattern "/path/to/your/oasst1_dromedary2_sft_response*.json" \
    --preference_file "/path/to/your/oasst1_dromedary2_sft_preference.json" \
    --output_file "/path/to/your/oasst1_dromedary2_sft_aggregated_preference.json"
```

</details>
