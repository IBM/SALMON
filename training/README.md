# Training Experiences

The whole **SALMON** process involves three stages: Synthetic Preference Collection, Training the Principle-following Reward Model, and RL Training with the Principle-following Reward Model. In our [paper](https://arxiv.org/abs/placeholder), we provide a detailed description of each of these stages.

## Prerequisites

For efficiency concerns, we utilize the [model parallel](https://github.com/facebookresearch/fairscale/tree/main/fairscale/nn/model_parallel) scheme from [llama](https://github.com/facebookresearch/llama) when sampling responses from the inital policy model. To prepare the sharded model checkpoints of LLaMA or Dromedary on your own machine/cluster, please refer to the [inference guide in Dromedary](https://github.com/IBM/Dromedary/tree/main/inference).

## Step 1:  Collecting Principle-Driven Synthetic Preference

We sample two responses from the initial policy model, and use the policy model itself to select the preferred response based on a certain human-written principle.

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

## Step 2: Training the Principle-following Reward Model

Next, for each user prompt, a subset of principles is randomly sampled from the established principle list with certain principles being randomly negated. The user prompt, model responses, and the sub-sampled principles are aggregated as a single training instance for the reward model.

### Step 2.1: Aggregating the Collected Preferences

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
cd step2_rm_training

python -u aggregate_synthetic_preference.py \
    --response_pattern "/path/to/your/oasst1_dromedary2_sft_response*.json" \
    --preference_file "/path/to/your/oasst1_dromedary2_sft_preference.json" \
    --output_file "/path/to/your/oasst1_dromedary2_sft_aggregated_preference.json"
```

</details>

### Step 2.2: Preference Modeling Pre-training (PMP) of the Reward Model

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
python -u clean_pmp_data.py \
    --output_file "/path/to/your/pmp_data.json"

salloc --nodes 1 --time 24:00:00 --gres=gpu:80g:8 srun bash scripts/train_reward_model_70b_qlora_pmp.sh
```

</details>

### Step 2.3: Fine-tune the Reward Model with Principle-driven Preferences

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
salloc --nodes 1 --time 24:00:00 --gres=gpu:80g:8 srun bash scripts/train_reward_model_70b_qlora_ft.sh
```

</details>

## Step 3: RL Training with the Principle-following Reward Model

Finally, we train the policy model with the principle-following reward model. We use the diverse user prompts from [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered), [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1), [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca), [MATH](https://huggingface.co/datasets/competition_math).

### Step 3.1: Preparing the RL Training Data

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
cd step3_ppo_training

python subsample_openorca_prompts.py \
    --train_data_path "/path/to/your/l1M-GPT4-Augmented.parquet (obtained from OpenOrca)" \
    --output_path "/path/to/your/openorca_prompts.json"

python aggregate_sharegpt_prompts.py \
    --data_files=zetavg/ShareGPT-Processed,path/to/sg_90k_part1.json.json,path/to/sg_90k_part1.json (obtained from ShareGPT_Vicuna_unfiltered) \
    --output_path "/path/to/sharegpt_prompts.json"

python clean_and_merge_prompts.py \
    --sharegpt_prompt_path "/path/to/sharegpt_prompts.json" \
    --openorca_prompt_path "/path/to/openorca_prompts.json" \
    --output_file "/path/to/your/salmon_merged_prompts.json"
```

</details>

### Step 3.2: RL Training

<details>
<summary> <strong> Running the code </strong> </summary>

```bash
salloc --nodes 6 --time 24:00:00 --gres=gpu:80g:8 srun bash scripts/train_ppo_model_70b_qlora_salmon.sh
```
