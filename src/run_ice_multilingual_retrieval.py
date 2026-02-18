#!/usr/bin/env python3
"""
ICE Evaluation for Multilingual LLMs — Delete vs Retrieval Infill

Same as run_ice_multilingual_nsr.py but with an additional operator: Retrieval Infill.
- Delete: keeps only top-k tokens (original approach)
- Retrieval: keeps all tokens, replaces non-rationale tokens with tokens
  from other examples (leave-one-out pool)

Usage:
    # Retrieval only
    python scripts/run_ice_multilingual_retrieval.py \
        --model gpt2 --languages de_native fr_native --extractor gradient \
        --operator retrieval

    # Both operators
    python scripts/run_ice_multilingual_retrieval.py \
        --model gpt2 --languages de_native fr_native --extractor gradient \
        --operator both
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ============================================================
# Retrieval Infill Pool (token-level, leave-one-out)
# ============================================================

class RetrievalInfillPool:
    """
    Token-level retrieval pool for ICE multilingual evaluation.
    Stores token IDs from each example with leave-one-out tagging.
    """

    def __init__(self, seed=42):
        self.token_sequences = []
        self.all_tokens = []
        self.rng = np.random.default_rng(seed)
        self._built = False
        self.label_token_ids = set()

    def build_pool(self, texts, tokenizer, label_token_ids=None):
        self.label_token_ids = set(label_token_ids or [])
        self.token_sequences = []
        self.all_tokens = []

        special_ids = set()
        for attr in ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id']:
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special_ids.add(tid)

        exclude = special_ids | self.label_token_ids

        for ex_id, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False)
            filtered = np.array([t for t in ids if t not in exclude])
            if len(filtered) >= 3:
                self.token_sequences.append((filtered, ex_id))
                for t in filtered:
                    self.all_tokens.append((int(t), ex_id))

        self._built = True
        print(f"  Retrieval pool: {len(self.all_tokens)} tokens, "
              f"{len(self.token_sequences)} sequences from {len(texts)} examples")

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
# Configurations (same as run_ice_multilingual_nsr.py)
# ============================================================

LANGUAGES = ["french", "german", "hindi", "chinese",
             "de_native", "fr_native", "hi_native", "cn_native",
             "tr_native", "ar_native"]

DATASET_REVISIONS = {
    "multilingual-sentiments": "a3080a58e5631380b388dc572d",
    "germeval2017": "99da66e994364c565ff980960c83fc9039f81266",
    "allocine": "a4654f4896408912913a62ace89614879a549287",
    "chnsenticorp": "b0c4c119c3fb33b8e735969202ef9ad13d7177e5a",
    "indicsentiment": "dc8f3f66886531c6897fedffcae938a68fc5013",
}

NATIVE_DATASETS = {
    "de_native": {
        "name": "uhhlt/GermEval2017",
        "split": "test_syn",
        "text_field": "Text",
        "label_field": "Sentiment",
        "revision_key": "germeval2017",
    },
    "fr_native": {
        "name": "allocine",
        "split": "test",
        "text_field": "review",
        "label_field": "label",
        "revision_key": "allocine",
    },
    "cn_native": {
        "name": "lansinuote/ChnSentiCorp",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "revision_key": "chnsenticorp",
    },
    "hi_native": {
        "name": "ai4bharat/IndicSentiment",
        "config": "translation-hi",
        "split": "test",
        "text_field": "INDIC REVIEW",
        "label_field": "LABEL",
        "revision_key": "indicsentiment",
    },
    "tr_native": {
        "name": "winvoker/turkish-sentiment-analysis-dataset",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
    },
    "ar_native": {
        "name": "labr",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
    },
}

PROMPTS = {
    "french": """Classifiez le sentiment du texte suivant comme positif ou négatif.

Texte: {text}

Sentiment:""",
    "fr_native": """Classifiez le sentiment du texte suivant comme positif ou négatif.

Texte: {text}

Sentiment:""",
    "german": """Klassifizieren Sie die Stimmung des folgenden Textes als positiv oder negativ.

Text: {text}

Stimmung:""",
    "de_native": """Klassifizieren Sie die Stimmung des folgenden Textes als positiv oder negativ.

Text: {text}

Stimmung:""",
    "hindi": """निम्नलिखित पाठ की भावना को सकारात्मक या नकारात्मक के रूप में वर्गीकृत करें।

पाठ: {text}

भावना:""",
    "hi_native": """निम्नलिखित पाठ की भावना को सकारात्मक या नकारात्मक के रूप में वर्गीकृत करें।

पाठ: {text}

भावना:""",
    "chinese": """请判断以下文本的情感是正面还是负面。

文本: {text}

情感:""",
    "cn_native": """请判断以下文本的情感是正面还是负面。

文本: {text}

情感:""",
    "turkish": """Aşağıdaki metnin duygusunu pozitif veya negatif olarak sınıflandırın.

Metin: {text}

Duygu:""",
    "tr_native": """Aşağıdaki metnin duygusunu pozitif veya negatif olarak sınıflandırın.

Metin: {text}

Duygu:""",
    "arabic": """صنف مشاعر النص التالي كإيجابي أو سلبي.

النص: {text}

المشاعر:""",
    "ar_native": """صنف مشاعر النص التالي كإيجابي أو سلبي.

النص: {text}

المشاعر:""",
}

LABELS = {
    "french": {0: " négatif", 1: " positif"},
    "fr_native": {0: " négatif", 1: " positif"},
    "german": {0: " negativ", 1: " positiv"},
    "de_native": {0: " negativ", 1: " positiv"},
    "hindi": {0: "नकारात्मक", 1: "सकारात्मक"},
    "hi_native": {0: "नकारात्मक", 1: "सकारात्मक"},
    "chinese": {0: "负面", 1: "正面"},
    "cn_native": {0: "负面", 1: "正面"},
    "turkish": {0: " negatif", 1: " pozitif"},
    "tr_native": {0: " negatif", 1: " pozitif"},
    "arabic": {0: " سلبي", 1: " إيجابي"},
    "ar_native": {0: " سلبي", 1: " إيجابي"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="ICE Multilingual Evaluation — Retrieval Infill")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model")
    parser.add_argument("--languages", nargs="+", default=["de_native", "fr_native"],
                        choices=LANGUAGES, help="Languages to evaluate")
    parser.add_argument("--extractor", type=str, default="gradient",
                        choices=["gradient", "attention"], help="Extraction method")
    parser.add_argument("--operator", type=str, default="retrieval",
                        choices=["delete", "retrieval", "both"],
                        help="Intervention operator")
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--k", type=float, default=0.2, help="Rationale budget")
    parser.add_argument("--n_permutations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--model_parallel", action="store_true")
    parser.add_argument("--gpu_ids", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--output_dir", type=str, default="results/multilingual_retrieval")
    return parser.parse_args()


# ============================================================
# Dataset loading (same as original)
# ============================================================

def load_native_dataset(max_examples=100, dataset_revision=None, lang="de_native"):
    if lang not in NATIVE_DATASETS:
        print(f"Unknown native dataset: {lang}")
        return [], []
    config = NATIVE_DATASETS[lang]
    try:
        ds_kwargs = {"split": config["split"]}
        revision_key = config.get("revision_key")
        if dataset_revision:
            ds_kwargs["revision"] = dataset_revision
        elif revision_key and DATASET_REVISIONS.get(revision_key):
            ds_kwargs["revision"] = DATASET_REVISIONS[revision_key]
        if "config" in config:
            ds = load_dataset(config["name"], config["config"], **ds_kwargs)
        else:
            ds = load_dataset(config["name"], **ds_kwargs)
        text_field = config["text_field"]
        label_field = config["label_field"]
        if lang == "de_native":
            sentiment_map = {"positive": 1, "negative": 0}
            filtered = []
            for ex in ds:
                sentiment = str(ex.get(label_field, "")).lower()
                if sentiment in sentiment_map:
                    text = str(ex.get(text_field, ""))[:400]
                    if text:
                        filtered.append((text, sentiment_map[sentiment]))
        elif lang == "hi_native":
            filtered = []
            for ex in ds:
                text = str(ex.get(text_field, ""))[:400]
                label = ex.get(label_field)
                try:
                    l_str = str(label).strip()
                    if l_str in ["2", "positive", "Positive"]:
                        filtered.append((text, 1))
                    elif l_str in ["0", "negative", "Negative"]:
                        filtered.append((text, 0))
                except:
                    pass
        elif lang == "tr_native":
            # Turkish: string labels "Positive"/"Negative"/"Notr" - skip neutral
            sentiment_map = {"Positive": 1, "Negative": 0}
            filtered = []
            for ex in ds:
                label_str = str(ex.get(label_field, "")).strip()
                if label_str in sentiment_map:
                    text = str(ex.get(text_field, ""))[:400]
                    if text:
                        filtered.append((text, sentiment_map[label_str]))
        else:
            filtered = []
            for ex in ds:
                text = str(ex.get(text_field, ""))[:400]
                label = ex.get(label_field)
                if text and label in [0, 1]:
                    filtered.append((text, label))
        texts = [t for t, _ in filtered][:max_examples]
        labels = [l for _, l in filtered][:max_examples]
        print(f"Loaded {len(texts)} {lang} examples (native)")
        return texts, labels
    except Exception as e:
        print(f"Failed to load {lang}: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def load_multilingual_dataset(lang, max_examples=100, dataset_revision=None):
    if lang in NATIVE_DATASETS:
        return load_native_dataset(max_examples, dataset_revision, lang)
    try:
        ds_kwargs = {"split": "test"}
        if dataset_revision:
            ds_kwargs["revision"] = dataset_revision
        elif "multilingual-sentiments" in DATASET_REVISIONS:
            ds_kwargs["revision"] = DATASET_REVISIONS["multilingual-sentiments"]
        ds = load_dataset("tyqiangz/multilingual-sentiments", lang, **ds_kwargs)
        filtered = [(ex["text"][:400], ex["label"]) for ex in ds if ex["label"] != 1]
        texts = [t for t, _ in filtered][:max_examples]
        labels = [0 if l == 0 else 1 for _, l in filtered][:max_examples]
        return texts, labels
    except Exception as e:
        print(f"Failed to load {lang}: {e}")
        return [], []


# ============================================================
# Label helpers (same as original)
# ============================================================

def get_label_token_ids(lang, tokenizer):
    label_map = LABELS[lang]
    return {k: tokenizer.encode(v, add_special_tokens=False) for k, v in label_map.items()}


def compute_label_prob(logits, label_token_ids, label_key):
    token_ids = label_token_ids[label_key]
    avg_logit = sum(logits[tid].item() for tid in token_ids) / len(token_ids)
    return avg_logit


def get_prediction(logits, label_token_ids):
    sorted_keys = sorted(label_token_ids.keys())
    scores = [compute_label_prob(logits, label_token_ids, k) for k in sorted_keys]
    probs = torch.softmax(torch.tensor(scores), dim=0)
    pred_idx = probs.argmax().item()
    return sorted_keys[pred_idx], probs[pred_idx].item(), probs


def get_prediction_score(ids, mask, model, device, label_token_ids, target_key):
    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=mask)
        logits = outputs.logits[0, -1, :]
        _, _, probs = get_prediction(logits, label_token_ids)
        return probs


# ============================================================
# Evaluation functions with operator support
# ============================================================

def evaluate_example_gradient(model, tokenizer, text, lang, k, n_permutations,
                               device, operator, pool=None, example_id=None):
    """Gradient-based evaluation with delete/retrieval operator support."""
    prompt = PROMPTS[lang].format(text=text)
    label_token_ids = get_label_token_ids(lang, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Original prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]
    predicted_key, confidence, probs = get_prediction(logits, label_token_ids)
    if confidence < 0.4:
        return None
    original_score = probs[predicted_key].item()

    # Baseline (empty) prediction
    baseline_ids = tokenizer.encode("Classify:", return_tensors="pt", add_special_tokens=True).to(device)
    baseline_mask = torch.ones_like(baseline_ids)
    baseline_probs = get_prediction_score(baseline_ids, baseline_mask, model, device, label_token_ids, predicted_key)
    baseline_score = baseline_probs[predicted_key].item() if baseline_probs is not None else 0.0

    # Gradient importance
    with torch.no_grad():
        embeds = model.get_input_embeddings()(input_ids)
    embeds = embeds.detach().clone()
    embeds.requires_grad = True
    outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
    target_token_id = label_token_ids[predicted_key][0]
    target_logit = outputs.logits[0, -1, target_token_id]
    target_logit.backward()
    importance = embeds.grad.abs().sum(dim=-1).squeeze()

    valid_pos = list(range(len(importance)))
    n_tokens = max(1, int(k * len(valid_pos)))
    top_k = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:n_tokens]
    top_k_set = set(top_k)

    # --- Rationale score ---
    if operator == "retrieval" and pool is not None:
        # Replace non-rationale with pool tokens
        non_rationale = [i for i in valid_pos if i not in top_k_set]
        replacement = pool.sample_tokens(len(non_rationale), exclude_example_id=example_id)
        modified_ids = input_ids.clone()
        for j, idx in enumerate(non_rationale):
            modified_ids[0, idx] = replacement[j]
        rat_probs = get_prediction_score(modified_ids, attention_mask, model, device, label_token_ids, predicted_key)
    else:
        # Delete: keep only top-k tokens
        top_k_sorted = sorted(top_k)
        rationale_ids = input_ids[0, top_k_sorted].unsqueeze(0)
        rationale_mask = torch.ones(1, len(top_k_sorted), device=device)
        rat_probs = get_prediction_score(rationale_ids, rationale_mask, model, device, label_token_ids, predicted_key)
    rationale_score = rat_probs[predicted_key].item()

    # --- Random baselines ---
    random_scores = []
    for _ in range(n_permutations):
        rand_idx = sorted(np.random.choice(valid_pos, size=n_tokens, replace=False))
        rand_set = set(rand_idx)

        if operator == "retrieval" and pool is not None:
            non_rand = [i for i in valid_pos if i not in rand_set]
            repl = pool.sample_tokens(len(non_rand), exclude_example_id=example_id)
            mod_ids = input_ids.clone()
            for j, idx in enumerate(non_rand):
                mod_ids[0, idx] = repl[j]
            rand_probs = get_prediction_score(mod_ids, attention_mask, model, device, label_token_ids, predicted_key)
        else:
            rand_ids = input_ids[0, rand_idx].unsqueeze(0)
            rand_mask = torch.ones(1, len(rand_idx), device=device)
            rand_probs = get_prediction_score(rand_ids, rand_mask, model, device, label_token_ids, predicted_key)
        random_scores.append(rand_probs[predicted_key].item())

    random_scores = np.array(random_scores)
    win_rate = np.mean(rationale_score > random_scores)
    effect_size = (rationale_score - np.mean(random_scores)) / (np.std(random_scores) + 1e-8)
    p_value = (1 + np.sum(random_scores >= rationale_score)) / (n_permutations + 1)

    rationale_nsr = (rationale_score - baseline_score) / (original_score - baseline_score + 1e-8)
    random_nsr_mean = (np.mean(random_scores) - baseline_score) / (original_score - baseline_score + 1e-8)

    return {
        "original_score": original_score,
        "baseline_score": baseline_score,
        "rationale_score": rationale_score,
        "win_rate": win_rate,
        "effect_size": effect_size,
        "p_value": p_value,
        "predicted_key": predicted_key,
        "n_tokens": n_tokens,
        "random_score_std": float(np.std(random_scores)),
        "random_score_mean": float(np.mean(random_scores)),
        "rationale_nsr": rationale_nsr,
        "random_nsr_mean": random_nsr_mean,
    }


def evaluate_example_attention(model, tokenizer, text, lang, k, n_permutations,
                                device, operator, pool=None, example_id=None):
    """Attention-based evaluation with delete/retrieval operator support."""
    prompt = PROMPTS[lang].format(text=text)
    label_token_ids = get_label_token_ids(lang, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Original prediction with attentions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        logits = outputs.logits[0, -1, :]
    predicted_key, confidence, probs = get_prediction(logits, label_token_ids)
    if confidence < 0.4:
        return None
    original_score = probs[predicted_key].item()

    # Baseline score
    baseline_ids = tokenizer.encode("Classify:", return_tensors="pt", add_special_tokens=True).to(device)
    baseline_mask = torch.ones_like(baseline_ids)
    baseline_probs = get_prediction_score(baseline_ids, baseline_mask, model, device, label_token_ids, predicted_key)
    baseline_score = baseline_probs[predicted_key].item() if baseline_probs is not None else 0.0

    # Attention importance (last token query, average heads)
    attentions = outputs.attentions[-1]
    attn_weights = attentions[0, :, -1, :].mean(dim=0).cpu()
    importance = attn_weights

    valid_pos = list(range(len(importance)))
    n_tokens = max(1, int(k * len(valid_pos)))
    top_k = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:n_tokens]
    top_k_set = set(top_k)

    # --- Rationale score ---
    if operator == "retrieval" and pool is not None:
        non_rationale = [i for i in valid_pos if i not in top_k_set]
        replacement = pool.sample_tokens(len(non_rationale), exclude_example_id=example_id)
        modified_ids = input_ids.clone()
        for j, idx in enumerate(non_rationale):
            modified_ids[0, idx] = replacement[j]
        rat_probs = get_prediction_score(modified_ids, attention_mask, model, device, label_token_ids, predicted_key)
    else:
        top_k_sorted = sorted(top_k)
        rationale_ids = input_ids[0, top_k_sorted].unsqueeze(0)
        rationale_mask = torch.ones(1, len(top_k_sorted), device=device)
        rat_probs = get_prediction_score(rationale_ids, rationale_mask, model, device, label_token_ids, predicted_key)
    rationale_score = rat_probs[predicted_key].item()

    # --- Random baselines ---
    random_scores = []
    for _ in range(n_permutations):
        rand_idx = sorted(np.random.choice(valid_pos, size=n_tokens, replace=False))
        rand_set = set(rand_idx)

        if operator == "retrieval" and pool is not None:
            non_rand = [i for i in valid_pos if i not in rand_set]
            repl = pool.sample_tokens(len(non_rand), exclude_example_id=example_id)
            mod_ids = input_ids.clone()
            for j, idx in enumerate(non_rand):
                mod_ids[0, idx] = repl[j]
            rand_probs = get_prediction_score(mod_ids, attention_mask, model, device, label_token_ids, predicted_key)
        else:
            rand_ids = input_ids[0, rand_idx].unsqueeze(0)
            rand_mask = torch.ones(1, len(rand_idx), device=device)
            rand_probs = get_prediction_score(rand_ids, rand_mask, model, device, label_token_ids, predicted_key)
        random_scores.append(rand_probs[predicted_key].item())

    random_scores = np.array(random_scores)
    win_rate = np.mean(rationale_score > random_scores)
    effect_size = (rationale_score - np.mean(random_scores)) / (np.std(random_scores) + 1e-8)
    p_value = (1 + np.sum(random_scores >= rationale_score)) / (n_permutations + 1)

    rationale_nsr = (rationale_score - baseline_score) / (original_score - baseline_score + 1e-8)
    random_nsr_mean = (np.mean(random_scores) - baseline_score) / (original_score - baseline_score + 1e-8)

    return {
        "original_score": original_score,
        "baseline_score": baseline_score,
        "rationale_score": rationale_score,
        "win_rate": win_rate,
        "effect_size": effect_size,
        "p_value": p_value,
        "predicted_key": predicted_key,
        "n_tokens": n_tokens,
        "random_score_std": float(np.std(random_scores)),
        "random_score_mean": float(np.mean(random_scores)),
        "rationale_nsr": rationale_nsr,
        "random_nsr_mean": random_nsr_mean,
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    operators_to_run = [args.operator] if args.operator != "both" else ["delete", "retrieval"]

    # Load model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "revision": args.model_revision,
    }
    if args.extractor == "attention":
        model_kwargs["attn_implementation"] = "eager"

    if args.model_parallel:
        print(f"=== MODEL PARALLEL MODE across GPUs {args.gpu_ids} ===")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
        model_kwargs["device_map"] = "auto"
    else:
        device_id = 0
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": device_id}
        else:
            model_kwargs["device_map"] = "cpu"

    print(f"\nLoading model: {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "7b" in args.model.lower() or "8b" in args.model.lower() or "14b" in args.model.lower():
        print("Enabling gradient checkpointing for large model")
        model.gradient_checkpointing_enable()

    model.eval()

    if hasattr(model, 'get_input_embeddings'):
        device = model.get_input_embeddings().weight.device
    elif hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    print(f"Main device: {device}")

    all_results = {}

    for op_name in operators_to_run:
        print(f"\n{'='*60}")
        print(f"Operator: {op_name}")
        print(f"{'='*60}")

        op_results = {}

        for lang in args.languages:
            print(f"\nEvaluating {lang} ({op_name})...")
            texts, labels = load_multilingual_dataset(lang, args.max_examples, args.dataset_revision)
            if not texts:
                op_results[lang] = {"error": "no_data"}
                continue

            # Build retrieval pool for this language if needed
            pool = None
            if op_name == "retrieval":
                label_token_ids_dict = get_label_token_ids(lang, tokenizer)
                all_label_tids = set()
                for tids in label_token_ids_dict.values():
                    all_label_tids.update(tids)
                pool = RetrievalInfillPool(seed=args.seed)
                pool.build_pool(texts, tokenizer, label_token_ids=all_label_tids)

            evaluate_fn = evaluate_example_gradient if args.extractor == "gradient" else evaluate_example_attention
            win_rates, effect_sizes, nsr_scores = [], [], []

            for ex_id, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts), desc=f"{lang}/{op_name}")):
                try:
                    res = evaluate_fn(
                        model, tokenizer, text, lang, args.k, args.n_permutations,
                        device, operator=op_name, pool=pool, example_id=ex_id
                    )
                    if res:
                        win_rates.append(res["win_rate"])
                        effect_sizes.append(res["effect_size"])
                        nsr_scores.append(res.get("rationale_nsr", 0))
                except Exception as e:
                    continue

            if win_rates:
                op_results[lang] = {
                    "win_rate": float(np.mean(win_rates)),
                    "effect_size": float(np.mean(effect_sizes)),
                    "mean_nsr": float(np.mean([n for n in nsr_scores if not np.isnan(n)])),
                    "n_examples": len(win_rates),
                }
                print(f"{lang}/{op_name}: Win Rate {op_results[lang]['win_rate']*100:.1f}%")
            else:
                op_results[lang] = {"error": "no_valid_results"}

        all_results[op_name] = op_results

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Save results
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    langs_tag = "_".join(args.languages[:2])
    op_tag = args.operator
    output_file = Path(args.output_dir) / f"ice_multilingual_retrieval_{args.model.split('/')[-1]}_{langs_tag}_{args.extractor}_{op_tag}_{timestamp}.json"

    save = {
        "config": {
            "model": args.model,
            "model_revision": args.model_revision,
            "languages": args.languages,
            "extractor": args.extractor,
            "operator": args.operator,
            "max_examples": args.max_examples,
            "k": args.k,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "timestamp": timestamp,
        },
        "results": all_results,
    }
    with open(output_file, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for op_name, op_results in all_results.items():
        print(f"\nOperator: {op_name}")
        for lang, res in op_results.items():
            if "error" in res:
                print(f"  {lang}: Error - {res['error']}")
            else:
                print(f"  {lang}: WR={res['win_rate']*100:.1f}%, ES={res['effect_size']:.2f}, NSR={res['mean_nsr']:.3f}")


if __name__ == "__main__":
    main()
