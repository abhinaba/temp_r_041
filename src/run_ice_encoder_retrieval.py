#!/usr/bin/env python3
"""
ICE Evaluation for Encoder Models — Delete vs Retrieval Infill

Same evaluation as run_ice.py but adds retrieval infill as an alternative operator.
- Delete: keeps only top-k tokens (original)
- Retrieval: replaces non-rationale tokens with tokens from other examples

Uses AutoModelForSequenceClassification (BERT-based encoders).

Encoder models are small enough to run multiple in parallel on a single H100 80GB.

Usage:
    # Single model, retrieval only
    python scripts/run_ice_encoder_retrieval.py \
        --model textattack/bert-base-uncased-SST-2 --dataset sst2 \
        --operator retrieval --extractors attention gradient

    # Both operators
    python scripts/run_ice_encoder_retrieval.py \
        --model textattack/bert-base-uncased-SST-2 --dataset sst2 \
        --operator both --extractors attention gradient integrated_gradients lime
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import get_eraser_dataset
from ice import get_extractor


# ============================================================
# Retrieval Infill Pool (token-level, leave-one-out)
# ============================================================

class RetrievalInfillPool:
    """Token-level retrieval pool for encoder model evaluation."""

    def __init__(self, seed=42):
        self.all_tokens = []  # list of (token_id, example_id)
        self.rng = np.random.default_rng(seed)
        self._built = False

    def build_pool(self, dataset, tokenizer):
        """Build pool from dataset examples."""
        self.all_tokens = []
        special_ids = set()
        for attr in ['pad_token_id', 'cls_token_id', 'sep_token_id',
                      'unk_token_id', 'mask_token_id']:
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special_ids.add(tid)

        for ex_id, item in enumerate(dataset):
            if isinstance(item, dict):
                input_ids = item["input_ids"]
                attention_mask = item["attention_mask"]
            else:
                input_ids, attention_mask, _ = item
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()
            for tid, mask_val in zip(input_ids, attention_mask):
                if mask_val == 1 and tid not in special_ids:
                    self.all_tokens.append((tid, ex_id))

        self._built = True
        print(f"  Retrieval pool: {len(self.all_tokens)} tokens from {ex_id + 1} examples")

    def sample_tokens(self, n, exclude_example_id):
        if not self._built:
            raise RuntimeError("Pool not built")
        valid = [t for t, eid in self.all_tokens if eid != exclude_example_id]
        if len(valid) == 0:
            raise ValueError(f"No tokens available excluding example {exclude_example_id}")
        if len(valid) < n:
            return list(self.rng.choice(valid, size=n, replace=True))
        return list(self.rng.choice(valid, size=n, replace=False))


# ============================================================
# Default configs
# ============================================================

DATASET_REVISIONS = {
    "sst2": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
    "movies": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
    "imdb": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
    # esnli: no git revision — builder hash a160e6a02bbb...

    "boolq": "35b264d03638db9f4ce671b711558bf7ff0f880d5",
    "multirc": "3de24cf8022e94f4ee4b9d55a6f5398991524d646",
}

MODEL_MAP = {
    "sst2": "textattack/bert-base-uncased-SST-2",
    "imdb": "textattack/bert-base-uncased-imdb",
    "esnli": "textattack/bert-base-uncased-snli",
    "boolq": "bert-base-uncased",
    "multirc": "bert-base-uncased",
}


def parse_args():
    parser = argparse.ArgumentParser(description="ICE Encoder Evaluation — Retrieval Infill")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["esnli", "boolq", "multirc", "movies", "sst2", "imdb"])
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: auto-select based on dataset)")
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--extractors", type=str, nargs="+",
                        default=["attention", "gradient", "integrated_gradients", "lime"])
    parser.add_argument("--operator", type=str, default="retrieval",
                        choices=["delete", "retrieval", "both"])
    parser.add_argument("--k", type=float, default=0.2)
    parser.add_argument("--n_permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/encoder_retrieval")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    return parser.parse_args()


# ============================================================
# Importance extraction
# ============================================================

def get_attention_importance(model, input_ids, attention_mask, target_class, device):
    """Extract attention-based importance scores."""
    ids = input_ids.unsqueeze(0).to(device) if input_ids.dim() == 1 else input_ids.to(device)
    mask = attention_mask.unsqueeze(0).to(device) if attention_mask.dim() == 1 else attention_mask.to(device)
    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=mask, output_attentions=True)
    # Average attention from last layer, CLS token (index 0) attending to all
    attn = outputs.attentions[-1]  # (batch, heads, seq, seq)
    importance = attn[0, :, 0, :].mean(dim=0).cpu()  # average across heads
    return importance


def get_gradient_importance(model, input_ids, attention_mask, target_class, device):
    """Extract gradient-based importance scores."""
    ids = input_ids.unsqueeze(0).to(device) if input_ids.dim() == 1 else input_ids.to(device)
    mask = attention_mask.unsqueeze(0).to(device) if attention_mask.dim() == 1 else attention_mask.to(device)
    embeds = model.get_input_embeddings()(ids)
    embeds = embeds.detach().clone()
    embeds.requires_grad = True
    outputs = model(inputs_embeds=embeds, attention_mask=mask)
    target_logit = outputs.logits[0, target_class]
    target_logit.backward()
    importance = embeds.grad.abs().sum(dim=-1).squeeze().cpu()
    return importance


def get_integrated_gradients_importance(model, input_ids, attention_mask, target_class, device, n_steps=20):
    """Integrated gradients importance."""
    ids = input_ids.unsqueeze(0).to(device) if input_ids.dim() == 1 else input_ids.to(device)
    mask = attention_mask.unsqueeze(0).to(device) if attention_mask.dim() == 1 else attention_mask.to(device)
    embed_layer = model.get_input_embeddings()
    baseline_embeds = torch.zeros_like(embed_layer(ids))
    target_embeds = embed_layer(ids).detach()
    integrated_grads = torch.zeros_like(target_embeds)

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interp = baseline_embeds + alpha * (target_embeds - baseline_embeds)
        interp = interp.detach().clone()
        interp.requires_grad = True
        outputs = model(inputs_embeds=interp, attention_mask=mask)
        target_logit = outputs.logits[0, target_class]
        target_logit.backward()
        integrated_grads += interp.grad

    integrated_grads = integrated_grads / n_steps
    importance = (integrated_grads * (target_embeds - baseline_embeds)).abs().sum(dim=-1).squeeze().cpu()
    return importance


def get_lime_importance(model, tokenizer, input_ids, attention_mask, target_class, device, n_samples=100):
    """LIME-style importance via random masking."""
    ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    seq_len = len(ids_list)
    mask_token_id = getattr(tokenizer, 'mask_token_id', tokenizer.unk_token_id or 0)

    perturbations = np.random.binomial(1, 0.5, size=(n_samples, seq_len)).astype(np.float32)
    scores = []

    for i in range(n_samples):
        perturbed_ids = ids_list.copy()
        for j in range(seq_len):
            if perturbations[i, j] == 0:
                perturbed_ids[j] = mask_token_id
        p_ids = torch.tensor([perturbed_ids], device=device)
        p_mask = attention_mask.unsqueeze(0).to(device) if attention_mask.dim() == 1 else attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids=p_ids, attention_mask=p_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            scores.append(probs[0, target_class].item())

    scores = np.array(scores)
    # Linear regression: score ~ perturbation features
    X = perturbations
    y = scores
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    cov = ((X - X_mean) * (y - y_mean)[:, None]).mean(axis=0)
    var = ((X - X_mean) ** 2).mean(axis=0) + 1e-8
    importance = np.abs(cov / var)
    return torch.tensor(importance, dtype=torch.float32)


EXTRACTOR_FUNCTIONS = {
    "attention": get_attention_importance,
    "gradient": get_gradient_importance,
    "integrated_gradients": get_integrated_gradients_importance,
    "lime": None,  # handled separately (needs tokenizer)
}


# ============================================================
# Single example evaluation
# ============================================================

def evaluate_single(model, tokenizer, input_ids, attention_mask, target_class,
                     importance, device, k, n_permutations, operator,
                     pool=None, example_id=None):
    """Evaluate a single example with delete or retrieval operator."""
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.squeeze(0)

    # Special tokens
    special_ids = set()
    for attr in ['cls_token_id', 'sep_token_id', 'pad_token_id']:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    # Valid (non-special, attended) positions
    valid_positions = []
    for i in range(len(input_ids)):
        if attention_mask[i] == 1 and input_ids[i].item() not in special_ids:
            valid_positions.append(i)

    if len(valid_positions) < 3:
        return None

    n_tokens = max(1, int(k * len(valid_positions)))

    # Get importance for valid positions and select top-k
    valid_importance = [(i, importance[i].item()) for i in valid_positions]
    valid_importance.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, _ in valid_importance[:n_tokens]]
    top_k_set = set(top_k_indices)

    def score_input(ids_tensor, mask_tensor):
        """Get probability of target class."""
        with torch.no_grad():
            ids = ids_tensor.unsqueeze(0).to(device) if ids_tensor.dim() == 1 else ids_tensor.to(device)
            mask = mask_tensor.unsqueeze(0).to(device) if mask_tensor.dim() == 1 else mask_tensor.to(device)
            outputs = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            return probs[0, target_class].item()

    # Original score
    original_score = score_input(input_ids, attention_mask)

    # Baseline score (empty input with just special tokens)
    cls_id = getattr(tokenizer, 'cls_token_id', None)
    sep_id = getattr(tokenizer, 'sep_token_id', None)
    if cls_id is not None and sep_id is not None:
        baseline_ids = torch.tensor([cls_id, sep_id])
        baseline_mask = torch.ones(2, dtype=torch.long)
    else:
        baseline_ids = torch.tensor([tokenizer.pad_token_id or 0])
        baseline_mask = torch.ones(1, dtype=torch.long)
    baseline_score = score_input(baseline_ids, baseline_mask)

    # --- Rationale score ---
    if operator == "retrieval" and pool is not None:
        non_rationale = [i for i in valid_positions if i not in top_k_set]
        if non_rationale:
            replacement = pool.sample_tokens(len(non_rationale), exclude_example_id=example_id)
            modified_ids = input_ids.clone()
            for j, idx in enumerate(non_rationale):
                modified_ids[idx] = replacement[j]
            rationale_score = score_input(modified_ids, attention_mask)
        else:
            rationale_score = original_score
    else:
        # Delete: keep only special + rationale tokens
        keep_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for idx in top_k_indices:
            keep_mask[idx] = True
        # Also keep special tokens
        for i in range(len(input_ids)):
            if input_ids[i].item() in special_ids and attention_mask[i] == 1:
                keep_mask[i] = True
        kept_ids = input_ids[keep_mask]
        kept_mask = torch.ones_like(kept_ids)
        rationale_score = score_input(kept_ids, kept_mask)

    # --- Random baselines ---
    random_scores = []
    rng = np.random.default_rng(None)  # fresh randomness for each example
    for _ in range(n_permutations):
        rand_idx = rng.choice(valid_positions, size=n_tokens, replace=False).tolist()
        rand_set = set(rand_idx)

        if operator == "retrieval" and pool is not None:
            non_rand = [i for i in valid_positions if i not in rand_set]
            if non_rand:
                repl = pool.sample_tokens(len(non_rand), exclude_example_id=example_id)
                mod_ids = input_ids.clone()
                for j, idx in enumerate(non_rand):
                    mod_ids[idx] = repl[j]
                rand_score = score_input(mod_ids, attention_mask)
            else:
                rand_score = original_score
        else:
            keep_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            for idx in rand_idx:
                keep_mask[idx] = True
            for i in range(len(input_ids)):
                if input_ids[i].item() in special_ids and attention_mask[i] == 1:
                    keep_mask[i] = True
            kept_ids = input_ids[keep_mask]
            kept_mask = torch.ones_like(kept_ids)
            rand_score = score_input(kept_ids, kept_mask)

        random_scores.append(rand_score)

    random_scores = np.array(random_scores)
    win_rate = float(np.mean(rationale_score > random_scores))
    effect_size = float((rationale_score - np.mean(random_scores)) / (np.std(random_scores) + 1e-8))
    p_value = float((1 + np.sum(random_scores >= rationale_score)) / (n_permutations + 1))

    # NSD (normalized score difference)
    suf_nsd = (rationale_score - baseline_score) / (original_score - baseline_score + 1e-8)

    return {
        "original_score": original_score,
        "baseline_score": baseline_score,
        "rationale_score": rationale_score,
        "win_rate": win_rate,
        "effect_size": effect_size,
        "p_value": p_value,
        "sufficiency_nsd": float(suf_nsd),
        "n_tokens": n_tokens,
        "random_score_mean": float(np.mean(random_scores)),
        "random_score_std": float(np.std(random_scores)),
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Auto-select model if not specified
    if args.model is None:
        args.model = MODEL_MAP.get(args.dataset, "bert-base-uncased")
    print(f"Model: {args.model}")

    # Load model — use eager attention so output_attentions=True works
    # (SDPA backend silently drops attentions, causing "tuple index out of range")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.model_revision,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model.to(device).eval()

    # Auto-select split
    if args.split is None:
        split_map = {
            "sst2": "validation", "movies": "validation",
            "esnli": "test", "boolq": "validation",
            "multirc": "validation", "imdb": "test",
        }
        args.split = split_map.get(args.dataset, "test")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset} (split={args.split})")
    dataset_revision = args.dataset_revision or DATASET_REVISIONS.get(args.dataset)
    dataset = get_eraser_dataset(
        args.dataset, tokenizer, split=args.split,
        max_length=args.max_length, max_examples=args.max_examples,
        revision=dataset_revision
    )
    dataset_list = list(dataset)
    print(f"Loaded {len(dataset_list)} examples")

    operators_to_run = [args.operator] if args.operator != "both" else ["delete", "retrieval"]

    all_results = {}

    for op_name in operators_to_run:
        print(f"\n{'='*60}")
        print(f"Operator: {op_name}")
        print(f"{'='*60}")

        # Build pool for retrieval
        pool = None
        if op_name == "retrieval":
            pool = RetrievalInfillPool(seed=args.seed)
            pool.build_pool(dataset_list, tokenizer)

        op_results = {}

        for extractor_name in args.extractors:
            print(f"\n  Evaluating: {extractor_name} + {op_name}")
            win_rates, effect_sizes, nsd_scores, p_values = [], [], [], []

            # Use framework extractor for attention/gradient (proven to work with delete)
            # Fall back to custom functions for lime/integrated_gradients
            framework_extractor = None
            if extractor_name in ("attention", "gradient"):
                try:
                    framework_extractor = get_extractor(extractor_name, model, tokenizer, device)
                    print(f"    Using framework {type(framework_extractor).__name__}")
                except Exception as e:
                    print(f"    Framework extractor failed, using custom: {e}")

            for ex_id, item in enumerate(tqdm(dataset_list, desc=f"{extractor_name}/{op_name}")):
                if isinstance(item, dict):
                    input_ids = item["input_ids"]
                    attention_mask = item["attention_mask"]
                else:
                    input_ids, attention_mask, _ = item

                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids)
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)

                # Get model prediction
                with torch.no_grad():
                    ids = input_ids.unsqueeze(0).to(device) if input_ids.dim() == 1 else input_ids.to(device)
                    mask = attention_mask.unsqueeze(0).to(device) if attention_mask.dim() == 1 else attention_mask.to(device)
                    outputs = model(input_ids=ids, attention_mask=mask)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    confidence = probs.max().item()
                    target_class = outputs.logits.argmax(dim=-1).item()

                if confidence < 0.5:
                    continue

                # Get importance scores
                try:
                    if framework_extractor is not None:
                        importance = framework_extractor.get_importance_scores(
                            input_ids, attention_mask, target_class
                        )
                    elif extractor_name == "lime":
                        importance = get_lime_importance(
                            model, tokenizer, input_ids, attention_mask, target_class, device
                        )
                    elif extractor_name == "integrated_gradients":
                        importance = get_integrated_gradients_importance(
                            model, input_ids, attention_mask, target_class, device
                        )
                    else:  # gradient fallback
                        importance = get_gradient_importance(
                            model, input_ids, attention_mask, target_class, device
                        )
                except Exception as e:
                    if ex_id < 3:
                        print(f"  [WARN] {extractor_name} extraction failed on example {ex_id}: {e}")
                    continue

                # Evaluate
                try:
                    result = evaluate_single(
                        model, tokenizer, input_ids, attention_mask, target_class,
                        importance, device, args.k, args.n_permutations,
                        operator=op_name, pool=pool, example_id=ex_id
                    )
                    if result:
                        win_rates.append(result["win_rate"])
                        effect_sizes.append(result["effect_size"])
                        nsd_scores.append(result["sufficiency_nsd"])
                        p_values.append(result["p_value"])
                except Exception as e:
                    if ex_id < 3:
                        print(f"  [WARN] {extractor_name} evaluation failed on example {ex_id}: {e}")
                    continue

            if win_rates:
                from scipy import stats as sp_stats
                try:
                    wilcoxon_stat, wilcoxon_p = sp_stats.wilcoxon(
                        nsd_scores, alternative='greater'
                    )
                except Exception:
                    wilcoxon_stat, wilcoxon_p = 0, 1.0

                op_results[extractor_name] = {
                    "win_rate": float(np.mean(win_rates)),
                    "effect_size": float(np.mean(effect_sizes)),
                    "mean_nsd": float(np.mean(nsd_scores)),
                    "std_nsd": float(np.std(nsd_scores)),
                    "mean_p_value": float(np.mean(p_values)),
                    "n_significant": int(np.sum(np.array(p_values) < 0.05)),
                    "n_examples": len(win_rates),
                    "wilcoxon_p": float(wilcoxon_p),
                }
                wr = op_results[extractor_name]["win_rate"]
                print(f"  {extractor_name}/{op_name}: WR={wr*100:.1f}%, n={len(win_rates)}")
            else:
                op_results[extractor_name] = {"error": "no_valid_results"}

        all_results[op_name] = op_results

    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output_dir) / f"ice_encoder_retrieval_{args.dataset}_{args.operator}_{timestamp}.json"

    save = {
        "config": {
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "model": args.model,
            "model_revision": args.model_revision,
            "extractors": args.extractors,
            "operator": args.operator,
            "k": args.k,
            "n_permutations": args.n_permutations,
            "max_examples": args.max_examples,
            "seed": args.seed,
            "timestamp": timestamp,
        },
        "results": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Extractor':<25} {'Operator':<12} {'Win Rate':>10} {'Effect Size':>12} {'NSD':>8} {'n':>5}")
    print("-" * 70)
    for op_name, op_results in all_results.items():
        for ext_name, res in op_results.items():
            if "error" not in res:
                print(f"{ext_name:<25} {op_name:<12} {res['win_rate']*100:>9.1f}% {res['effect_size']:>12.2f} {res['mean_nsd']:>8.3f} {res['n_examples']:>5}")


if __name__ == "__main__":
    main()
