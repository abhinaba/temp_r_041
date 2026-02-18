#!/usr/bin/env python3
"""
ICE-CoT Runner with Retrieval Infill
=====================================
Enhanced runner with:
1. Retrieval infill operator (reviewer-proof, leave-one-out)
2. Detailed debug output
3. Dataset + model version tracking in JSON
4. Operator comparison: deletion vs retrieval

Usage:
    python3 ice_cot_retrieval_runner.py --model phi2 --n 100
    python3 ice_cot_retrieval_runner.py --model qwen_05b --dataset esnli --retrieval-only
    python3 ice_cot_retrieval_runner.py --sentiment-scrub  # Ablation with sentiment filter
"""

import torch
import numpy as np
import json
import argparse
import gc
import sys
import platform
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting ICE-CoT Retrieval Runner")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")


# =============================================================================
# VERSION TRACKING
# =============================================================================

def get_environment_info() -> dict:
    """Get detailed environment info for reproducibility."""
    import transformers
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return info


def get_model_revision(model_name: str) -> Optional[str]:
    """Get HuggingFace model revision SHA."""
    try:
        from huggingface_hub import model_info
        info = model_info(model_name, timeout=10)
        return info.sha
    except Exception as e:
        print(f"  ⚠️ Could not get model revision: {e}")
        return None


def get_dataset_revision(dataset_name: str, config: str = None) -> Optional[str]:
    """Get HuggingFace dataset revision SHA."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info(dataset_name, timeout=10)
        return info.sha
    except Exception as e:
        print(f"  ⚠️ Could not get dataset revision: {e}")
        return None


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODELS = {
    # === EFFICIENT SMALL MODELS (< 16GB VRAM) ===
    # Validated HuggingFace model names as of Jan 2026
    "llama32_1b": {"name": "meta-llama/Llama-3.2-1B-Instruct", "vram_gb": 3},  # Sep 2024
    "llama32_3b": {"name": "meta-llama/Llama-3.2-3B-Instruct", "vram_gb": 7},  # Sep 2024
    "lfm2_1b": {"name": "LiquidAI/LFM2-1.2B", "vram_gb": 3},  # Jul 2025, Apache 2.0
    "lfm25_1b": {"name": "LiquidAI/LFM2.5-1.2B-Instruct", "vram_gb": 3},  # Jan 2026, latest
    "qwen3_06b": {"name": "Qwen/Qwen3-0.6B", "vram_gb": 2},  # 2025, small dense
    "phi4_mini": {"name": "microsoft/Phi-4-mini-instruct", "vram_gb": 8},  # Dec 2024, MIT, 3.8B
    "smollm3": {"name": "HuggingFaceTB/SmolLM3-3B", "vram_gb": 7},  # 2025, fully open
    
    # === MEDIUM MODELS (16-24GB VRAM) ===
    "nemotron_nano": {"name": "nvidia/Nemotron-3-Nano-4B-Instruct", "vram_gb": 9},  # Dec 2025, MoE 3.6B active
    "qwen3_8b": {"name": "Qwen/Qwen3-8B", "vram_gb": 17},  # 2025 Qwen3
    "deepseek_r1_7b": {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "vram_gb": 15},  # Jan 2025, MIT
    "mistral_small3": {"name": "mistralai/Mistral-Small-24B-Instruct-2501", "vram_gb": 24},  # Jan 2025
    
    # === NEW: SCALE EXPERIMENT MODELS ===
    "gpt_oss_20b": {"name": "openai/GPT-OSS-20B", "vram_gb": 16},  # MoE, fits 16GB quantized
    # NOTE: Ministral-3B-Reasoning requires transformers>=4.58 (ministral3 arch)
    # Using Phi-4-mini as fallback - also reasoning-optimized, broadly compatible
    "phi4_reasoning": {"name": "microsoft/Phi-4-mini-reasoning", "vram_gb": 8},  # Reasoning-optimized
    "qwen3_4b": {"name": "Qwen/Qwen3-4B", "vram_gb": 9},  # Fill the 4B gap
    
    # === LARGE MODELS (40GB+ VRAM) ===
    "qwen3_14b": {"name": "Qwen/Qwen3-14B", "vram_gb": 30},  # 2025
    "deepseek_r1_14b": {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "vram_gb": 30},  # Jan 2025
    
    # === LEGACY (backward compatibility) ===
    "qwen_05b": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "vram_gb": 1.5},
    "phi2": {"name": "microsoft/phi-2", "vram_gb": 6},
}

DATASET_CONFIGS = {
    "sst2": {
        "hf_name": "glue",
        "hf_config": "sst2",
        "split": "validation",
        "text_field": "sentence",
        "task_prompt": "Classify the sentiment as positive or negative.",
        "labels": ["positive", "negative"],
    },
    "esnli": {
        "hf_name": "esnli",
        "hf_config": None,
        "split": "validation",
        "text_field": None,  # Custom handling
        "task_prompt": "Determine if the hypothesis is entailment, neutral, or contradiction given the premise.",
        "labels": ["entailment", "neutral", "contradiction"],
    },
    "agnews": {
        "hf_name": "ag_news",
        "hf_config": None,
        "split": "test",
        "text_field": "text",
        "task_prompt": "Classify the news topic as World, Sports, Business, or Technology.",
        "labels": ["World", "Sports", "Business", "Technology"],
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log(msg: str):
    """Timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def generate_cot(model, tokenizer, text: str, task_prompt: str, labels: list, max_tokens: int = 100) -> str:
    """Generate Chain-of-Thought reasoning."""
    label_str = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 2 else " or ".join(labels)
    prompt = f"{task_prompt} Think step by step.\n\nText: {text}\n\nAnalysis:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prompt + generated[len(prompt):] + "\n\nAnswer:"


# =============================================================================
# NOTE: Retrieval now uses RetrievalOperatorICE wrapper (see below)
# =============================================================================


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_experiment(
    model_key: str,
    dataset: str = "sst2",
    n_examples: int = 100,
    operators: List[str] = ["delete", "retrieval"],
    sentiment_scrub: bool = False,
    keep_mode: str = "random",  # "random" for fair test, "last" for diagnostic
    k_values: List[float] = [0.2, 0.5, 0.8],
    n_permutations: int = 50,
    output_dir: Path = Path("results"),
    seed: int = 42,
    model_name_override: str = None,  # For custom HuggingFace models
    store_cot: bool = False  # Save per-example CoTs for plausibility analysis
) -> dict:
    """Run ICE-CoT evaluation with retrieval infill (ICE-consistent).
    
    Args:
        model_key: Key from MODELS dict (e.g., 'qwen3_7b')
        model_name_override: If provided, use this HuggingFace model path instead
        ... (other args)
    """
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Resolve model name: override takes priority
    if model_name_override:
        model_name = model_name_override
    elif model_key in MODELS:
        model_name = MODELS[model_key]["name"]
    else:
        raise ValueError(f"Unknown model key: {model_key}. Use --model-name for custom models.")
    
    ds_config = DATASET_CONFIGS[dataset]
    
    log("="*70)
    log(f"EXPERIMENT: {model_name} on {dataset}")
    log(f"  Operators: {operators}")
    log(f"  Sentiment Scrub: {sentiment_scrub}")
    log(f"  N={n_examples}, k={k_values}, perms={n_permutations}")
    log("="*70)
    
    # Get version info
    log("📌 Getting version info...")
    model_revision = get_model_revision(model_name)
    dataset_revision = get_dataset_revision(ds_config["hf_name"], ds_config["hf_config"])
    env_info = get_environment_info()
    
    log(f"  Model revision: {model_revision or 'unknown'}")
    log(f"  Dataset revision: {dataset_revision or 'unknown'}")
    
    # Load model
    log(f"🤖 Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    vram_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    log(f"  ✓ Model loaded (VRAM: {vram_used:.1f}GB)")
    
    # Load dataset
    log(f"📊 Loading dataset: {dataset}...")
    if ds_config["hf_config"]:
        ds = load_dataset(ds_config["hf_name"], ds_config["hf_config"], split=ds_config["split"])
    else:
        ds = load_dataset(ds_config["hf_name"], split=ds_config["split"])
    
    # Extract texts
    if dataset == "esnli":
        texts = [f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}" for ex in list(ds)[:n_examples]]
    else:
        texts = [ex[ds_config["text_field"]] for ex in list(ds)[:n_examples]]
    
    log(f"  ✓ Loaded {len(texts)} examples")
    
    # Generate CoT
    log("🧠 Generating Chain-of-Thought...")
    cot_examples = []
    for i, text in enumerate(tqdm(texts, desc="CoT Generation")):
        cot = generate_cot(model, tokenizer, text, ds_config["task_prompt"], ds_config["labels"])
        cot_examples.append(cot)
        
        if (i + 1) % 50 == 0:
            clear_memory()
            log(f"  Generated {i+1}/{len(texts)} CoTs")
    
    log(f"  ✓ Generated {len(cot_examples)} CoT examples")
    
    # Save per-example CoTs if requested (for plausibility analysis)
    if store_cot:
        cot_output_dir = output_dir / "cot_examples"
        cot_output_dir.mkdir(parents=True, exist_ok=True)
        cot_file = cot_output_dir / f"{model_key}_{dataset}_cots.jsonl"
        
        # Get human explanations for e-SNLI
        with open(cot_file, "w", encoding="utf-8") as f:
            for i, (text, cot) in enumerate(zip(texts, cot_examples)):
                record = {
                    "idx": i,
                    "text": text,
                    "cot": cot,
                }
                # Add human explanations for e-SNLI
                if dataset == "esnli":
                    ex = list(ds)[i]
                    record["human_explanations"] = [
                        ex.get('explanation_1', ''),
                        ex.get('explanation_2', ''),
                        ex.get('explanation_3', '')
                    ]
                    record["label"] = ["entailment", "neutral", "contradiction"][ex.get('label', 0)]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        log(f"💾 Saved {len(cot_examples)} CoT examples to: {cot_file}")
    
    # Print sample CoT
    log("📝 Sample CoT (first example):")
    print("-" * 40)
    print(cot_examples[0][:500] + "..." if len(cot_examples[0]) > 500 else cot_examples[0])
    print("-" * 40)
    
    # Import evaluator and operators
    from ice_cot_eval import ICECoTEvaluatorV2, CoTEvalConfig, OPERATORS
    from retrieval_operator_ice import RetrievalOperatorICE, build_retrieval_pool_from_cots
    
    # Build retrieval operator if needed (using ICE-compatible wrapper)
    if "retrieval" in operators:
        log("🔍 Building ICE-compatible retrieval operator...")
        retrieval_pool = build_retrieval_pool_from_cots(
            cot_examples, seed=seed, use_sentiment_scrub=sentiment_scrub
        )
        # Create ICE-compatible wrapper (NOT the raw operator)
        retrieval_op_ice = RetrievalOperatorICE(
            pool=retrieval_pool, 
            keep_mode=keep_mode,  # "random" for fair test
            seed=seed
        )
        # Register in OPERATORS dict
        OPERATORS["retrieval"] = retrieval_op_ice
        log(f"  ✓ Pool: {len(retrieval_pool.all_words)} words, {len(retrieval_pool.spans)} spans")
    
    # Run evaluation for each operator
    all_results = {}
    
    for op_name in operators:
        log(f"\n⚙️ Evaluating with {op_name.upper()} operator...")
        
        if op_name not in OPERATORS:
            log(f"  ⚠️ Unknown operator: {op_name}, skipping")
            continue
        
        # Configure evaluator with task-specific labels
        config = CoTEvalConfig(
            operators=[op_name],  # Use only this operator
            labels=ds_config["labels"],  # Task-specific labels (not just SST-2!)
            n_permutations=n_permutations,
            k_values=k_values,
            primary_k_necessity=0.2,
            primary_k_sufficiency=0.8,
            seed=seed
        )
        
        # Create evaluator with this operator
        evaluator = ICECoTEvaluatorV2(model, tokenizer, config)
        evaluator.operators = {op_name: OPERATORS[op_name]}  # Override operators
        
        # Run ICE evaluation (with matched random baselines!)
        results = evaluator.evaluate_dataset(cot_examples)
        
        all_results[op_name] = results
        
        # Print summary
        tax = results.get("taxonomy", {}).get(op_name, {})
        log(f"  ✓ {op_name}: taxonomy={tax}")
    
    # Compile final results
    final_results = {
        # Metadata
        "experiment": {
            "model_key": model_key,
            "model_name": model_name,
            "model_revision": model_revision,
            "dataset": dataset,
            "dataset_revision": dataset_revision,
            "n_examples": len(cot_examples),
            "operators": operators,
            "sentiment_scrub": sentiment_scrub,
            "k_values": k_values,
            "n_permutations": n_permutations,
            "seed": seed,
        },
        "environment": env_info,
        "results": all_results,
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ice_cot_retrieval_{model_key}_{dataset}_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    log(f"\n✅ Results saved: {output_file}")
    
    # Print summary
    log("\n" + "="*70)
    log("SUMMARY")
    log("="*70)
    for op_name, res in all_results.items():
        if isinstance(res, dict):
            # REVIEW12 FIX: Use correct key names from evaluator
            nec_wr = res.get("necessity_wr", {}).get(op_name, {}).get(0.2, "N/A")
            suf_wr = res.get("sufficiency_wr", {}).get(op_name, {}).get(0.8, "N/A")
            tax = res.get("taxonomy", {}).get(op_name, {}).get(0.2, "N/A")
            log(f"  {op_name}: Nec@0.2={nec_wr:.1%}, Suf@0.8={suf_wr:.1%}, Tax={tax}" if isinstance(nec_wr, float) else f"  {op_name}: Nec@0.2={nec_wr}, Suf@0.8={suf_wr}, Tax={tax}")
    
    # Cleanup
    del model
    clear_memory()
    
    return final_results



# NOTE: evaluate_with_retrieval() removed - now using ICECoTEvaluatorV2 exclusively





# NOTE: get_model_score_v2() removed - using evaluator's score() method now


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ICE-CoT Retrieval Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FAST MODE (~2hrs for 7B model) - Recommended for adoption:
  python ice_cot_retrieval_runner.py --model qwen3_8b --fast
  
  # QUICK MODE (~20min) - For testing/debugging only:
  python ice_cot_retrieval_runner.py --model qwen3_06b --quick

  # FULL MODE (~6hrs per dataset) - Publication quality:
  python ice_cot_retrieval_runner.py --model qwen3_8b --n 500 --perms 100

  # Benchmark ANY HuggingFace model:
  python ice_cot_retrieval_runner.py --model-name "your-org/your-model" --fast
"""
    )
    parser.add_argument("--model", type=str, default=None, 
                       choices=list(MODELS.keys()), help="Preset model key (see MODELS dict)")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Any HuggingFace model path (e.g., 'meta-llama/Llama-3.2-3B-Instruct'). Use this to benchmark YOUR model.")
    parser.add_argument("--dataset", type=str, default="sst2",
                       choices=list(DATASET_CONFIGS.keys()), help="Dataset")
    parser.add_argument("--n", type=int, default=None, help="Number of examples (default: 500 full, 200 fast, 50 quick)")
    parser.add_argument("--operators", type=str, nargs="+", 
                       default=["delete", "retrieval"],
                       help="Operators to compare")
    parser.add_argument("--retrieval-only", action="store_true",
                       help="Only run retrieval operator")
    parser.add_argument("--sentiment-scrub", action="store_true",
                       help="Also filter sentiment words in retrieval pool")
    parser.add_argument("--k", type=float, nargs="+", default=None,
                       help="K values to test (default: [0.2, 0.5, 0.8] full, [0.2, 0.8] fast)")
    parser.add_argument("--keep-mode", type=str, default="random",
                       choices=["random", "last"],
                       help="Sufficiency keep mode: random (fairer) or last (diagnostic)")
    parser.add_argument("--perms", type=int, default=None, help="Permutations per example (default: 100 full, 20 fast)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--store-cot", action="store_true",
                       help="💾 Save per-example CoTs to JSONL for plausibility analysis")
    
    # Speed modes
    parser.add_argument("--fast", action="store_true",
                       help="⚡ Fast mode: ~8x faster (n=200, perms=20, k=[0.2,0.8]). Good for adoption & model comparison.")
    parser.add_argument("--quick", action="store_true",
                       help="🚀 Quick mode: ~30x faster (n=50, perms=10, k=[0.2]). For debugging only, NOT publication.")
    
    args = parser.parse_args()
    
    # Apply speed presets (can be overridden by explicit args)
    if args.quick:
        n_default, perms_default, k_default = 50, 10, [0.2]
        mode_label = "🚀 QUICK (debug only)"
    elif args.fast:
        n_default, perms_default, k_default = 200, 20, [0.2, 0.8]
        mode_label = "⚡ FAST (adoption mode)"
    else:
        n_default, perms_default, k_default = 500, 100, [0.2, 0.5, 0.8]
        mode_label = "📊 FULL (publication quality)"
    
    # Use explicit args if provided, otherwise use preset defaults
    n_examples = args.n if args.n is not None else n_default
    n_perms = args.perms if args.perms is not None else perms_default
    k_values = args.k if args.k is not None else k_default
    
    # Resolve model: --model-name takes priority over --model
    if args.model_name:
        model_name = args.model_name
        log(f"Using custom HuggingFace model: {model_name}")
    elif args.model:
        model_name = MODELS[args.model]["name"]
    else:
        model_name = MODELS["qwen_05b"]["name"]  # Default
    
    operators = ["retrieval"] if args.retrieval_only else args.operators
    
    log(f"\n{'='*70}")
    log(f"ICE-CoT RETRIEVAL BENCHMARK - {mode_label}")
    log(f"{'='*70}")
    log(f"Model: {model_name}")
    log(f"Dataset: {args.dataset}")
    log(f"Operators: {operators}")
    log(f"N={n_examples}, k={k_values}, perms={n_perms}")
    log(f"Sentiment Scrub: {args.sentiment_scrub}")
    log(f"Keep Mode: {args.keep_mode}")
    log(f"Seed: {args.seed}")
    if args.fast:
        log("💡 TIP: Use --n, --perms, --k to override fast mode defaults")
    log("="*70 + "\n")
    
    results = run_experiment(
        model_key=args.model if args.model else "custom",
        model_name_override=model_name,  # Pass resolved model name
        dataset=args.dataset,
        n_examples=n_examples,
        operators=operators,
        sentiment_scrub=args.sentiment_scrub,
        keep_mode=args.keep_mode,
        k_values=k_values,
        n_permutations=n_perms,
        output_dir=Path(args.output),
        seed=args.seed,
        store_cot=args.store_cot
    )
    
    log("\n🏁 DONE!")

