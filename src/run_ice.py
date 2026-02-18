#!/usr/bin/env python3
"""
ICE Evaluation Runner

Main script for running ICE faithfulness evaluation on ERASER datasets.

Usage:
    python run_ice.py --dataset esnli --model bert-base-uncased --extractors attention gradient lime
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ice import ICEEvaluator, ICEConfig, get_extractor
from data import get_eraser_dataset


# Default dataset revisions for reproducibility (ICEBench pinned versions)
DATASET_REVISIONS = {
    "sst2": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
    "movies": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",  # movies uses SST-2
    "imdb": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
    # esnli: no git revision — builder hash a160e6a02bbb...

    "boolq": "35b264d03638db9f4ce671b711558bf7ff0f880d5",
    "multirc": "3de24cf8022e94f4ee4b9d55a6f5398991524d646",  # super_glue
    "fever": "2a74f2909caf2b8656343aeb8203e50bf84dccb56",
}


def parse_args():
    parser = argparse.ArgumentParser(description="ICE Faithfulness Evaluation")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="sst2",
        choices=["esnli", "boolq", "multirc", "movies", "sst2", "fever", "imdb"],
        help="ERASER dataset to evaluate"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,  # Auto-select based on dataset
        help="Data split to use (default: auto-select based on dataset)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate"
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional HF dataset revision/commit hash for pinning (reproducibility)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="textattack/bert-base-uncased-SST-2",
        help="Model name or path (should be fine-tuned for the task)"
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="Optional HF model revision/commit hash for reproducibility"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    
    # Extractor arguments
    parser.add_argument(
        "--extractors",
        type=str,
        nargs="+",
        default=["attention", "gradient", "lime"],
        help="Rationale extraction methods to compare"
    )
    
    # ICE configuration
    parser.add_argument(
        "--k",
        type=float,
        default=0.2,
        help="Rationale budget (fraction of tokens)"
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=100,
        help="Number of permutations for randomization test (M) - more = finer p-values"
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap samples (B)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Significance level (default 0.10 for reasonable power)"
    )
    parser.add_argument(
        "--operators",
        type=str,
        default="lite",
        choices=["lite", "full"],
        help="Operator configuration"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def setup_model_and_tokenizer(model_name: str, device: str, model_revision: str = None):
    """Load model and tokenizer"""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=model_revision)
    model.to(device)
    model.eval()
    return model, tokenizer


def run_evaluation(args):
    """Main evaluation function"""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model, device, model_revision=args.model_revision)
    
    # Auto-select split based on dataset if not specified
    if args.split is None:
        split_map = {
            "sst2": "validation",
            "movies": "validation", 
            "esnli": "test",
            "boolq": "validation",
            "multirc": "validation",
            "fever": "test",
            "imdb": "test"  # IMDB only has train/test/unsupervised
        }
        args.split = split_map.get(args.dataset, "test")
        print(f"Auto-selected split: {args.split}")
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset_revision = args.dataset_revision or DATASET_REVISIONS.get(args.dataset)
    if dataset_revision:
        print(f"Using pinned revision: {dataset_revision}")
    dataset = get_eraser_dataset(
        args.dataset,
        tokenizer,
        split=args.split,
        max_length=args.max_length,
        max_examples=args.max_examples,
        revision=dataset_revision
    )
    print(f"Loaded {len(dataset)} examples")
    
    # Configure ICE
    config = ICEConfig(
        operators=args.operators,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        k_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        device=device,
        seed=args.seed
    )
    
    # Initialize evaluator
    evaluator = ICEEvaluator(model, tokenizer, config)
    
    # Run evaluation for each extractor
    results = {}
    
    for extractor_name in args.extractors:
        print(f"\n{'='*60}")
        print(f"Evaluating: {extractor_name}")
        print('='*60)
        
        try:
            extractor = get_extractor(extractor_name, model, tokenizer, device)
            result = evaluator.evaluate_dataset(
                dataset,
                extractor,
                k=args.k,
                show_progress=True
            )
            results[extractor_name] = result
            
            # Print summary
            print(f"\n{result.summary()}")
            
        except Exception as e:
            print(f"Error evaluating {extractor_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ice_results_{args.dataset}_{timestamp}.json"
    
    # Prepare serializable results
    serializable_results = {
        "config": {
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "model": args.model,
            "model_revision": args.model_revision,
            "n_examples": len(dataset),
            "k": args.k,
            "n_permutations": args.n_permutations,
            "n_bootstrap": args.n_bootstrap,
            "operators": args.operators,
            "timestamp": timestamp
        },
        "results": {}
    }
    
    for name, result in results.items():
        serializable_results["results"][name] = result.to_dict()
    
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print comparison table
    print_comparison_table(results)
    
    # Check for ranking flips
    analyze_ranking_flips(results, args)
    
    return results


def print_comparison_table(results: dict):
    """Print comparison table of all methods"""
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    headers = ["Method", "Suf (NSD)", "95% CI", "Comp (NSD)", "95% CI", "Sig. Rate"]
    row_format = "{:<15} {:>10} {:>20} {:>10} {:>20} {:>10}"
    
    print(row_format.format(*headers))
    print("-"*80)
    
    for name, result in results.items():
        suf = result.sufficiency_stats
        comp = result.comprehensiveness_stats
        
        suf_ci = f"[{suf['mean_ci_lower']:.3f}, {suf['mean_ci_upper']:.3f}]"
        comp_ci = f"[{comp['mean_ci_lower']:.3f}, {comp['mean_ci_upper']:.3f}]"
        sig_rate = suf.get('fdr_rejection_rate', 0) * 100
        
        print(row_format.format(
            name,
            f"{suf['mean']:.3f}",
            suf_ci,
            f"{comp['mean']:.3f}",
            comp_ci,
            f"{sig_rate:.1f}%"
        ))
    
    # NEW: Win Rate and Effect Size Table (KEY METRICS FOR DIFFERENTIATION)
    print("\n" + "="*80)
    print("FAITHFULNESS METRICS (vs. Random Baseline)")
    print("="*80)
    print("\nThese metrics differentiate methods even when significance rates are low:")
    print()
    
    header = "{:<15} {:>12} {:>12} {:>12} {:>15}".format(
        "Method", "Win Rate", "Effect Size", "Percentile", "Interpretation"
    )
    print(header)
    print("-" * 80)
    
    for name, result in results.items():
        # Compute mean win rate and effect size across examples
        win_rates = [r.get("win_rate", 0) for r in result.example_results]
        effect_sizes = [r.get("effect_size", 0) for r in result.example_results]
        percentiles = [r.get("percentile", 50) for r in result.example_results]
        
        mean_win_rate = np.mean(win_rates) if win_rates else 0
        mean_effect_size = np.mean(effect_sizes) if effect_sizes else 0
        mean_percentile = np.mean(percentiles) if percentiles else 50
        
        # Interpretation
        if mean_win_rate > 0.7:
            interp = "✓ Good"
        elif mean_win_rate > 0.5:
            interp = "~ Marginal"
        else:
            interp = "✗ Poor"
        
        print("{:<15} {:>11.1f}% {:>12.2f} {:>11.1f}% {:>15}".format(
            name,
            mean_win_rate * 100,
            mean_effect_size,
            mean_percentile,
            interp
        ))
    
    print()
    print("Legend:")
    print("  Win Rate: % of random rationales beaten (>50% = better than random)")
    print("  Effect Size: Cohen's d (0.2=small, 0.5=medium, 0.8=large)")
    print("  Percentile: Where method falls in random distribution (>50 = above average)")


def analyze_ranking_flips(results: dict, args):
    """Analyze potential ranking flips vs ERASER"""
    print("\n" + "="*80)
    print("RANKING ANALYSIS")
    print("="*80)
    
    if len(results) < 2:
        print("Need at least 2 methods to compare rankings")
        return
    
    # Sort by sufficiency NSD
    ranked = sorted(
        results.items(),
        key=lambda x: x[1].sufficiency_stats['mean'],
        reverse=True
    )
    
    print("\nRanking by Sufficiency (ICE):")
    for i, (name, result) in enumerate(ranked, 1):
        suf = result.sufficiency_stats
        print(f"  {i}. {name}: {suf['mean']:.3f} ± {suf['mean_se']:.3f}")
    
    # Check for potential flips (non-overlapping CIs)
    print("\nPotential Ranking Flips (non-overlapping 95% CIs):")
    flips_found = False
    
    for i, (name1, res1) in enumerate(ranked):
        for name2, res2 in ranked[i+1:]:
            suf1 = res1.sufficiency_stats
            suf2 = res2.sufficiency_stats
            
            # Check if CIs don't overlap
            if suf1['mean_ci_lower'] > suf2['mean_ci_upper']:
                print(f"  ✓ {name1} significantly better than {name2}")
                flips_found = True
    
    if not flips_found:
        print("  No significant differences found (CIs overlap)")
    
    # Check significance rates
    print("\nSignificance Analysis (vs. random rationales):")
    print(f"  (Using α={args.alpha}, M={args.n_permutations} permutations)")
    print()
    
    for name, result in results.items():
        suf = result.sufficiency_stats
        sig_rate = suf.get('fdr_rejection_rate', 0)
        n_sig = suf.get('n_significant_fdr', 0)
        n_total = result.n_examples
        
        if sig_rate < 0.1:
            status = "⚠️  LOW"
        elif sig_rate < 0.5:
            status = "⚠️  MODERATE"
        else:
            status = "✓ GOOD"
        
        print(f"  {name}: {n_sig}/{n_total} ({sig_rate*100:.1f}%) significant {status}")
    
    # P-value distribution analysis
    print("\n" + "=" * 80)
    print("P-VALUE DISTRIBUTION")
    print("=" * 80)
    
    for name, result in results.items():
        p_values = np.array([r["randomization_p_value"] for r in result.example_results])
        
        print(f"\n{name}:")
        print(f"  Mean p-value: {np.mean(p_values):.3f}")
        print(f"  Median p-value: {np.median(p_values):.3f}")
        print(f"  Min p-value: {np.min(p_values):.3f}")
        
        # Histogram
        bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
        hist, _ = np.histogram(p_values, bins=bins)
        print(f"  Distribution:")
        for i in range(len(bins)-1):
            bar = "█" * min(hist[i], 40)
            pct = hist[i] / len(p_values) * 100
            marker = " ← significant" if bins[i+1] <= args.alpha else ""
            print(f"    p∈[{bins[i]:.2f},{bins[i+1]:.2f}): {hist[i]:3d} ({pct:5.1f}%) {bar}{marker}")


def main():
    args = parse_args()
    results = run_evaluation(args)
    return results


if __name__ == "__main__":
    main()
