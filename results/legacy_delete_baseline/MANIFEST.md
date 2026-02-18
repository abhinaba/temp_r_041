# Legacy Delete Baseline Results

## Origin

These files are **legacy results from the original paper experiments** (deletion operator, January 2026). They were produced during the main paper's evaluation runs and copied here for comparison with the new retrieval infill operator.

## Important Notes

1. **Incomplete coverage.** Each file contains results for a **single extractor** (either `llm_attention` or `llm_gradient`), not both. The complete deletion results (both extractors, all 7 models x 4 datasets) are reported in Table 5 of the submitted paper.

2. **Debug file.** `ice_llm_nsr_sst2_20260104_203601.json` contains only n=10 examples (a debug run). All other files use n=500.

3. **Missing win rates.** Three files have no extractable win_rate data:
   - `ice_llm_nsr_sst2_20260105_012544.json` (Mistral-7B, SST-2)
   - `ice_llm_nsr_sst2_20260105_013647.json` (Llama-3.1-8B, SST-2)
   - `ice_llm_nsr_imdb_20260105_012659.json` (Mistral-7B, IMDB)

4. **Table 2 values.** The README Table 2 ("Delete Operator Win Rates") reports averages across both extractors from the **complete** original experiment data (matching the paper's Table 5). These cannot be fully reconstructed from the files in this directory alone, since each file has only one extractor.

## File Summary

| Filename | Model | Dataset | n | Extractor | Win Rate |
|----------|-------|---------|---|-----------|----------|
| ice_llm_nsr_agnews_20260104_212529.json | GPT-2 | agnews | 500 | llm_gradient | 0.4865 |
| ice_llm_nsr_agnews_20260104_220305.json | LFM2-2.6B | agnews | 500 | llm_gradient | 0.4955 |
| ice_llm_nsr_agnews_20260104_222043.json | Qwen2.5-7B | agnews | 500 | llm_attention | 0.6216 |
| ice_llm_nsr_agnews_20260104_232835.json | Llama-3.1-8B | agnews | 500 | llm_attention | 0.4753 |
| ice_llm_nsr_agnews_20260104_233818.json | deepseek-7B | agnews | 500 | llm_attention | 0.4935 |
| ice_llm_nsr_agnews_20260105_002708.json | Llama-3.2-3B | agnews | 500 | llm_attention | 0.5599 |
| ice_llm_nsr_agnews_20260105_033942.json | Mistral-7B | agnews | 500 | llm_gradient | 0.6074 |
| ice_llm_nsr_esnli_20260104_223717.json | GPT-2 | esnli | 500 | llm_attention | 0.6401 |
| ice_llm_nsr_esnli_20260104_225212.json | Qwen2.5-7B | esnli | 500 | llm_attention | 0.7727 |
| ice_llm_nsr_esnli_20260104_225844.json | LFM2-2.6B | esnli | 500 | llm_attention | 0.6655 |
| ice_llm_nsr_esnli_20260104_231608.json | Llama-3.1-8B | esnli | 500 | llm_attention | 0.8524 |
| ice_llm_nsr_esnli_20260104_232313.json | deepseek-7B | esnli | 500 | llm_attention | 0.6373 |
| ice_llm_nsr_esnli_20260105_001228.json | Llama-3.2-3B | esnli | 500 | llm_attention | 0.8640 |
| ice_llm_nsr_esnli_20260105_033049.json | Mistral-7B | esnli | 500 | llm_gradient | 0.5073 |
| ice_llm_nsr_imdb_20260104_212155.json | GPT-2 | imdb | 500 | llm_gradient | 0.4259 |
| ice_llm_nsr_imdb_20260104_215212.json | LFM2-2.6B | imdb | 500 | llm_gradient | 0.5403 |
| ice_llm_nsr_imdb_20260104_220554.json | Qwen2.5-7B | imdb | 500 | llm_attention | 0.9491 |
| ice_llm_nsr_imdb_20260104_225927.json | Llama-3.1-8B | imdb | 500 | llm_attention | 0.5253 |
| ice_llm_nsr_imdb_20260104_230222.json | deepseek-7B | imdb | 500 | llm_attention | 0.8459 |
| ice_llm_nsr_imdb_20260104_235831.json | Llama-3.2-3B | imdb | 500 | llm_attention | 0.7126 |
| ice_llm_nsr_imdb_20260105_012659.json | Mistral-7B | imdb | 500 | -- | -- |
| ice_llm_nsr_sst2_20260104_203601.json | GPT-2 | sst2 | **10** | llm_gradient + llm_attention | 0.5980 / 0.6340 |
| ice_llm_nsr_sst2_20260104_214104.json | LFM2-2.6B | sst2 | 500 | llm_gradient | 0.4177 |
| ice_llm_nsr_sst2_20260104_215056.json | Qwen2.5-7B | sst2 | 500 | llm_attention | 0.6842 |
| ice_llm_nsr_sst2_20260104_224705.json | deepseek-7B | sst2 | 500 | llm_attention | 0.6219 |
| ice_llm_nsr_sst2_20260104_234400.json | Llama-3.2-3B | sst2 | 500 | llm_attention | 0.5323 |
| ice_llm_nsr_sst2_20260105_012544.json | Mistral-7B | sst2 | 500 | -- | -- |
| ice_llm_nsr_sst2_20260105_013647.json | Llama-3.1-8B | sst2 | 500 | -- | -- |

## Relationship to Paper

The paper (Table 5) reports deletion win rates for all 7 models x 4 datasets x 2 extractors (attention + gradient). These legacy files contain a representative **subset** of those runs -- typically one extractor per (model, dataset) pair. The README Table 2 values are averaged from the paper's complete Table 5, not derived from these files alone.
