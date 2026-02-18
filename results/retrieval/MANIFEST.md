# Retrieval Infill Results

## Overview

This directory contains retrieval infill operator results for 7 LLMs across 4 English benchmark datasets, using both attention-based and gradient-based attribution extractors.

**Coverage:** 7 models x 4 datasets x 2 extractors = 56 unique (model, dataset, extractor) configurations.

## Canonical Result Selection

Multiple runs may exist for the same (model, dataset, extractor) configuration. When duplicates are present, the **latest timestamp** (highest `YYYYMMDD_HHMMSS` suffix) is the canonical result used in the analysis and README tables. This convention is implemented in `analysis/generate_figures.py` and `analysis/consistency_check.py`.

## File Naming Convention

```
ice_llm_retrieval_{model}_{dataset}_{YYYYMMDD}_{HHMMSS}.json
```

Each file contains results for a **single extractor** (either `llm_attention` or `llm_gradient`), specified in `config.extractors`. The summary key includes the operator suffix (e.g., `llm_attention/retrieval`).

## Experimental Parameters

All runs use identical parameters:
- **k = 0.2** (20% of tokens intervened)
- **n_permutations = 50** (random baselines per example)
- **n_examples = 500** (evaluation set size)
- **seed = 42** (fixed for reproducibility)
- **operator = retrieval** (leave-one-out corpus sampling with label blacklisting)

## Models

| Short Name | Full Model Path |
|------------|----------------|
| GPT-2 | `gpt2` |
| LFM2-2.6B | `LiquidAI/LFM2-2.6B-Exp` |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` |
| deepseek-7B | `deepseek-ai/deepseek-llm-7b-chat` |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` |

## Datasets

| Key | Full Name | Task |
|-----|-----------|------|
| sst2 | Stanford Sentiment Treebank v2 | Binary sentiment |
| agnews | AG News | 4-class topic classification |
| esnli | e-SNLI | Natural language inference |
| imdb | IMDB Reviews | Binary sentiment (long text) |
