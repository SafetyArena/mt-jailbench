# MT-JailBench

[![arXiv](https://img.shields.io/badge/arXiv-2605.11002-b31b1b.svg)](https://arxiv.org/abs/2605.11002)

MT-JailBench is a benchmark framework for studying multi-turn jailbreak attacks. It supports component-level attack analysis, reusable attack/defense modules, resource-aware evaluation, reproducible experimentation, and more.

### ⚙️ Core Features

- **Attack decomposition**: Break multi-turn attacks into reusable components for analysis and recombination.
- **Modular architecture**: Mix and match attacks, defenses, datasets, and models.
- **Resource-aware evaluation**: Set resource budgets that reflect real-world constraints.
- **Strong built-in baselines**: Includes five multi-turn attacks from prior work.

### 🧰 Additional Capabilities

- **Unified LLM client**: Access multiple LLM providers through a single client.
- **Cross-validation judging**: Re-evaluate successful attacks using alternative success criteria.
- **Experiment checkpointing**: Pause and resume experiments with automatic result caching.
- **Well-engineered codebase**: Built for reproducibility and extensibility—not obscure, one-off scripts.

## Environment Setup

<details>
<summary>Expand</summary>

Configure API keys for the LLM providers you plan to use:

```sh
# OpenAI / Azure OpenAI
export OPENAI_API_KEY=...

# Gemini
export GEMINI_API_KEY=...

# Anthropic
export ANTHROPIC_API_KEY=...

# OpenRouter
export OPENROUTER_API_KEY=...

# AWS Bedrock
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=...

# Azure OpenAI (find endpoint in Foundry portal)
export AZURE_OPENAI_ENDPOINT="https://xxxxxxxxx.cognitiveservices.azure.com/openai/v1/"

# Hugging Face (for downloading open-source models)
export HF_TOKEN=...
export HF_HOME=...  # Use $SCRATCH/LLMs if you're on NERSC
```

To set up the environment, simply run:

```sh
uv sync
```

If you also need a similarity model server (for using similarity-based attacks like COA):

```sh
uv run --extra sim-server sim-server
```

We do not provide general instructions for deploying LLMs locally. However, if you are using NERSC, the following steps may work.

```sh
# install conda environment for using vLLM
conda env create -f scripts/nersc/vllm.yaml

# Deploy the attacker model (defaults to Qwen2.5-32B-Instruct)
# Uses a context window 4× larger than the target model below
./scripts/nersc/deploy_attacker.sh

# deploy target model (specify name model name)
./scripts/nersc/deploy_target.sh
```

</details>

## User Guide

### Demo Mode

<details>
<summary>Expand</summary>

Demo mode runs the multi-turn jailbreak workflow on a single harmful behavior. This mode is useful for quick experimentation and debugging.

```sh
uv run demo <attack_type>
```

Currently supported attack types: `crescendo`, `actor`, `coa`, `fitd`, `xteaming`, `mix`, `interactive` (see [here](./docs/attacks.md) for details)

</details>

### Benchmark Mode

<details>
<summary>Expand</summary>

Benchmark mode evaluates 159 harmful behaviors from the HarmBench dataset. It requires a YAML configuration file; see [sample config file](./config/sample.yaml) for an example.

```sh
# Step 1: Run the benchmark.
uv run benchmark --config ./config/sample.yaml --name sample_run

# Step 2: Summarize results.
uv run summarize --name sample_run

# Optional: Retroactively run independent judge.
uv run retro-judge --name sample_run
```

* **Step 1** runs the benchmark using the specified configuration file. Results are stored in an output directory named after `--name`. If the output directory already contains partial results, completed behaviors are automatically skipped.
* **Step 2** summarizes the benchmark outputs and generates a `summary.json` file in the same output directory.
* **Optional step** retrofits an independent judge if it was not enabled during the original experiment. *Warning: This command directly modifies output files.*

Running MT-JailBench in benchmark mode creates the following output structure:

```text
outputs/
└── experiment_name/
    ├── config.json
    ├── summary.json
    └── logs/
        ├── behavior_id_1.json
        ├── behavior_id_2.json
        └── ...
```

* `config.json` stores a copy of the run configuration for reproducibility. Rerunning a partially completed experiment with a different configuration will raise an error. If the configuration change is safe and intentional, update the saved config file or delete it to unblock the run.
* `summary.json` contains aggregated benchmark results and key metrics, such as attack success rate.
* `logs/` contains detailed logs for each behavior. These logs are useful for analysis, debugging, and auditing.

</details>

## Documentation

<details>
<summary>Expand</summary>

This README covers the basics of using MT-JailBench. More detailed documentation is available in [`./docs`](./docs/).

- [Developer Guide](./docs/developer_guide.md): explains key terminology and the system architecture
- [Unified LLM Client Guide](./docs/unified_llm_client.md): explains how to use the unified LLM client across different providers
- [Attacks](./docs/attacks.md): explains how to use built-in attacks and how to implement new attacks

</details>
