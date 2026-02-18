"""
ICE-CoT Colab Runner v2.0
========================
Updated with all methodology fixes from guide's recommendations:
- Reasoning-only CoT extraction (excludes final answer tokens)
- Multi-token scoring fallback
- Continuous delta storage for Wilcoxon tests
- Statistical taxonomy with Wilcoxon signed-rank
- N=500 default, 100 permutations

Usage in Colab:
1. Upload this file or clone the repo
2. Run: from ice_cot_colab_v2 import run_experiment
3. Run: results = run_experiment("microsoft/phi-2", "sst2")
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm  # Use regular tqdm for terminal
import json
from datetime import datetime
import gc

# Try to import scipy, install if not available
try:
    from scipy import stats
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "scipy"])
    from scipy import stats


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CoTEvalConfig:
    """Configuration for ICE-CoT evaluation with methodology fixes.
    
    Modes:
        - Full mode (default): Publication-quality, ~6hrs per dataset on 7B model
        - Fast mode: Community adoption, ~45min per dataset on 7B model
    
    Usage:
        # Full mode (default) - for paper submissions
        config = CoTEvalConfig()
        
        # Fast mode - for rapid iteration & adoption
        config = CoTEvalConfig.fast()
    """
    
    operators: List[str] = field(default_factory=lambda: ["delete", "mask_unk", "neutral"])
    k_values: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Task-specific labels (REQUIRED for multi-task support)
    labels: List[str] = field(default_factory=lambda: ["positive", "negative"])
    
    # Primary k for hypothesis testing (guide recommendation)
    primary_k_necessity: float = 0.2
    primary_k_sufficiency: float = 0.8
    
    # Statistical settings
    n_permutations: int = 100  # Increased for stability
    alpha: float = 0.10
    
    # Evaluation settings
    max_examples: int = 500  # N=500 like original paper
    seed: int = 42
    
    # Methodology fixes (guide recommendations)
    exclude_final_answer: bool = True  # Remove label tokens from CoT
    require_agreement: bool = True  # Agreement gating
    
    # Performance mode
    fast_mode: bool = False  # If True, use reduced settings for faster evaluation
    
    @classmethod
    def fast(cls, **kwargs):
        """Fast mode preset for community adoption (~8x faster).
        
        Changes from full mode:
        - n_permutations: 100 → 20 (5x faster)
        - k_values: 5 → 2 (2.5x faster)  
        - max_examples: 500 → 200 (2.5x faster)
        
        Combined speedup: ~8-10x (18hrs → ~2hrs for 7B model)
        Statistical validity: Sufficient for model comparison, may have wider CIs
        """
        return cls(
            n_permutations=20,
            k_values=[0.2, 0.8],  # Only primary k values
            max_examples=200,
            fast_mode=True,
            **kwargs
        )
    
    @classmethod
    def quick(cls, **kwargs):
        """Ultra-quick preset for debugging (~30x faster).
        
        Only use for testing pipeline, NOT for publication.
        """
        return cls(
            n_permutations=10,
            k_values=[0.2],
            max_examples=50,
            fast_mode=True,
            **kwargs
        )


# =============================================================================
# INTERVENTION OPERATORS
# =============================================================================

class DeleteOperator:
    """Delete tokens from CoT region."""
    name = "delete"
    
    def apply_necessity(self, text, cot_start, cot_end, k=1.0):
        """Remove fraction k of CoT words."""
        cot = text[cot_start:cot_end]
        words = cot.split()
        if not words:
            return text
        
        n = max(1, int(len(words) * k))
        if k >= 1.0:
            return text[:cot_start] + text[cot_end:]
        
        idx = set(np.random.choice(len(words), n, replace=False))
        kept = [w for i, w in enumerate(words) if i not in idx]
        return text[:cot_start] + " ".join(kept) + text[cot_end:]
    
    def apply_sufficiency(self, text, cot_start, cot_end, k=1.0):
        """
        Keep only fraction k of CoT words.
        SCAFFOLD-PRESERVING: prefix + kept_cot + suffix (review12 fix)
        """
        prefix = text[:cot_start]
        suffix = text[cot_end:]
        cot = text[cot_start:cot_end]
        words = cot.split()
        if not words:
            return prefix + suffix
        
        n = max(1, int(len(words) * k))
        # Keep LAST n words (deterministic "rationale" = final reasoning steps)
        kept = words[-n:]
        return prefix + " ".join(kept) + suffix


class MaskUNKOperator:
    """Replace tokens with [UNK]."""
    name = "mask_unk"
    mask = "[UNK]"
    
    def apply_necessity(self, text, cot_start, cot_end, k=1.0):
        cot = text[cot_start:cot_end]
        words = cot.split()
        if not words:
            return text
        
        n = max(1, int(len(words) * k))
        idx = set(np.random.choice(len(words), min(n, len(words)), replace=False))
        masked = [self.mask if i in idx else w for i, w in enumerate(words)]
        return text[:cot_start] + " ".join(masked) + text[cot_end:]
    
    def apply_sufficiency(self, text, cot_start, cot_end, k=1.0):
        """
        SCAFFOLD-PRESERVING (review14 fix): Same as delete - prefix kept, CoT reduced.
        This makes sufficiency comparable across all operators.
        """
        prefix = text[:cot_start]
        suffix = text[cot_end:]
        cot = text[cot_start:cot_end]
        words = cot.split()
        if not words:
            return prefix + suffix
        
        # Keep last k fraction of CoT
        n = max(1, int(len(words) * k))
        kept_cot = " ".join(words[-n:])
        return prefix + kept_cot + suffix


class NeutralOperator:
    """Replace with neutral tokens."""
    name = "neutral"
    tokens = ["the", "a", ".", ",", "is"]
    
    def _get(self, w):
        # STABLE HASH (review12 fix): use hashlib instead of Python hash()
        import hashlib
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        return self.tokens[h % len(self.tokens)]
    
    def apply_necessity(self, text, cot_start, cot_end, k=1.0):
        cot = text[cot_start:cot_end]
        words = cot.split()
        if not words:
            return text
        
        n = max(1, int(len(words) * k))
        idx = set(np.random.choice(len(words), min(n, len(words)), replace=False))
        replaced = [self._get(w) if i in idx else w for i, w in enumerate(words)]
        return text[:cot_start] + " ".join(replaced) + text[cot_end:]
    
    def apply_sufficiency(self, text, cot_start, cot_end, k=1.0):
        """
        SCAFFOLD-PRESERVING (review14 fix): Same as delete - prefix kept, CoT reduced.
        This makes sufficiency comparable across all operators.
        """
        prefix = text[:cot_start]
        suffix = text[cot_end:]
        cot = text[cot_start:cot_end]
        words = cot.split()
        if not words:
            return prefix + suffix
        
        # Keep last k fraction of CoT
        n = max(1, int(len(words) * k))
        kept_cot = " ".join(words[-n:])
        return prefix + kept_cot + suffix


OPERATORS = {
    "delete": DeleteOperator(),
    "mask_unk": MaskUNKOperator(),
    "neutral": NeutralOperator()
}


# =============================================================================
# EVALUATOR WITH METHODOLOGY FIXES
# =============================================================================

class ICECoTEvaluatorV2:
    """
    ICE-CoT Evaluator v2.0 with all methodology fixes.
    
    Key improvements:
    - Reasoning-only CoT extraction (excludes final answer)
    - Multi-token scoring fallback
    - Continuous deltas for Wilcoxon tests
    - Statistical taxonomy
    """
    
    def __init__(self, model, tokenizer, config=None, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CoTEvalConfig()
        self.device = device
        self.operators = {n: OPERATORS[n] for n in self.config.operators}
        
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
    
    def _teacher_forced_score(self, text: str, answer: str) -> float:
        """Teacher-forced log-probability for multi-token answers."""
        full_text = text + answer
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, labels=inputs.input_ids)
            return -outputs.loss.item()
    
    def score(self, text: str) -> Tuple[float, str]:
        """
        Score with multi-token fallback (guide recommendation).
        Generalized for any task labels (not just SST-2).
        Returns (max_prob, predicted_label).
        """
        probs, _ = self._get_label_probs(text)
        if probs is None:
            return 0.5, self.config.labels[0]
        
        max_idx = probs.argmax().item()
        return probs[max_idx].item(), self.config.labels[max_idx]
    
    def score_for_label(self, text: str, target_label: str) -> float:
        """
        Get probability of a SPECIFIC label (for baseline-label scoring).
        
        REVIEW12 FIX: Interventions are scored by probability of the 
        BASELINE-predicted label, not the max prob after intervention.
        """
        probs, label_to_idx = self._get_label_probs(text)
        if probs is None or target_label not in label_to_idx:
            return 0.5
        
        return probs[label_to_idx[target_label]].item()
    
    def _get_label_probs(self, text: str) -> Tuple[Optional[torch.Tensor], Dict[str, int]]:
        """Get probability distribution over labels."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        labels = self.config.labels
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
            
            # Try to get logits for each label
            label_logits = []
            use_teacher_forced = False
            
            for label in labels:
                # Try single-token path first
                tokens = self.tokenizer.encode(f" {label}", add_special_tokens=False)
                if len(tokens) == 1:
                    label_logits.append(logits[0, tokens[0]].item())
                else:
                    use_teacher_forced = True
                    break
            
            if use_teacher_forced:
                # Safe path: teacher-forced for all labels
                label_logits = []
                for label in labels:
                    score = self._teacher_forced_score(text, f" {label}")
                    label_logits.append(score)
            
            if not label_logits:
                return None, label_to_idx
            
            probs = torch.softmax(torch.tensor(label_logits), dim=0)
            return probs, label_to_idx
    
    def extract_cot_region(self, text: str) -> Tuple[int, int]:
        """
        Extract REASONING-ONLY region (guide recommendation).
        Excludes final answer tokens to prevent leakage.
        """
        start_markers = ["Analysis:", "Reasoning:", "Let me think", "Step by step:"]
        # FIXED: Only structural end markers (review11 fix)
        end_markers = ["Answer:", "Final:"]
        # Use config labels (not hardcoded SST-2)
        answer_tokens = self.config.labels + [l.capitalize() for l in self.config.labels]
        
        start = 0
        end = len(text)
        
        # Find start
        for marker in start_markers:
            if marker in text:
                start = text.find(marker) + len(marker)
                break
        
        # Find end (before answer markers)
        for marker in end_markers:
            if marker in text[start:]:
                end = start + text[start:].find(marker)
                break
        
        # CRITICAL: Remove any answer tokens from reasoning (guide recommendation)
        if self.config.exclude_final_answer:
            reasoning = text[start:end].rstrip()
            for token in answer_tokens:
                if reasoning.endswith(token):
                    end = end - len(token) - 1
                    break
        
        return start, end
    
    def run_random_baseline(self, text, cot_start, cot_end, test_type, k, op=None, n=20, baseline_label=None):
        """
        Run operator-matched random baseline (review11 + review12 fixes).
        
        Key changes:
        - Random baseline applies SAME operator to RANDOM spans
        - Scores use baseline_label probability (not max prob)
        - Sufficiency is scaffold-preserving
        """
        cot_region = text[cot_start:cot_end]
        cot_words = cot_region.split()
        if not cot_words:
            return []
        
        n_words = max(1, int(len(cot_words) * k))
        prefix = text[:cot_start]
        suffix = text[cot_end:]
        
        scores = []
        for _ in range(n):
            try:
                if test_type == "necessity":
                    # TRULY OPERATOR-MATCHED necessity (review15 fix)
                    # Apply the SAME intervention type as the operator to a RANDOM span
                    start_idx = np.random.randint(0, max(1, len(cot_words) - n_words + 1))
                    
                    # WORD-INDEX approach (review14 fix)
                    before_span = cot_words[:start_idx]
                    span_words = cot_words[start_idx:start_idx + n_words]
                    after_span = cot_words[start_idx + n_words:]
                    
                    if op is not None:
                        # Apply operator's intervention type to random span
                        if hasattr(op, 'mask'):  # MaskUNKOperator
                            intervened_span = [op.mask] * len(span_words)
                        elif hasattr(op, 'tokens') and hasattr(op, '_get'):  # NeutralOperator
                            intervened_span = [op._get(w) for w in span_words]
                        elif hasattr(op, 'sample_replacement_span'):  # RetrievalOperatorICE
                            # Use CONTIGUOUS SPAN sampling for perfect operator match
                            try:
                                current_ex = getattr(op, '_current_example_id', None)
                                if hasattr(op, 'set_current_example') and current_ex is not None:
                                    op.set_current_example(current_ex)
                                sampled = op.sample_replacement_span(len(span_words))
                                intervened_span = sampled if sampled else span_words
                            except:
                                intervened_span = span_words  # Fallback: keep original
                        else:  # DeleteOperator or unknown - just delete
                            intervened_span = []
                        
                        new_cot = ' '.join(before_span + intervened_span + after_span)
                    else:
                        # Fallback: delete random span
                        new_cot = ' '.join(before_span + after_span)
                    
                    rnd_text = prefix + new_cot + suffix
                else:
                    # OPERATOR-MATCHED sufficiency (review17 fix)
                    # For retrieval: use keep_mode="random" as baseline (review18 fix)
                    if op is not None and hasattr(op, 'apply_sufficiency_random_keep'):
                        # RETRIEVAL OPERATOR: Use its random keep method for baseline
                        try:
                            # Set current example for leave-one-out
                            current_ex = getattr(op, '_current_example_id', None)
                            if hasattr(op, 'set_current_example') and current_ex is not None:
                                op.set_current_example(current_ex)
                            
                            # Call retrieval sufficiency with FORCED random span
                            rnd_text = op.apply_sufficiency_random_keep(text, cot_start, cot_end, k=k)
                        except:
                            # Fallback to delete-style
                            if len(cot_words) <= n_words:
                                kept = cot_words
                            else:
                                start_idx = np.random.randint(0, len(cot_words) - n_words + 1)
                                kept = cot_words[start_idx:start_idx + n_words]
                            rnd_text = prefix + ' '.join(kept) + suffix
                    else:
                        # Non-retrieval: keep random span (word-index approach review14)
                        if len(cot_words) <= n_words:
                            kept = cot_words
                        else:
                            start_idx = np.random.randint(0, len(cot_words) - n_words + 1)
                            kept = cot_words[start_idx:start_idx + n_words]
                        
                        # SCAFFOLD-PRESERVING: prefix + kept_cot + suffix
                        rnd_text = prefix + ' '.join(kept) + suffix
                
                # REVIEW12 FIX: Score for baseline label, not max
                if baseline_label:
                    s = self.score_for_label(rnd_text, baseline_label)
                else:
                    s, _ = self.score(rnd_text)
                
                if not np.isnan(s):
                    scores.append(s)
            except:
                continue
        
        return scores
    
    def evaluate_single(self, text: str, idx: int = 0) -> Optional[Dict]:
        """Evaluate single example with continuous deltas."""
        cot_start, cot_end = self.extract_cot_region(text)
        
        if cot_end <= cot_start:
            return None
        
        # First get the predicted label
        _, baseline_pred = self.score(text)
        
        # REVIEW15 FIX: Use baseline-label probability consistently
        # This ensures baseline_score and intervention scores are comparable
        baseline_score = self.score_for_label(text, baseline_pred)
        
        # Agreement gating
        if self.config.require_agreement:
            no_cot_text = text[:cot_start] + text[cot_end:]
            _, no_cot_pred = self.score(no_cot_text)
            if baseline_pred != no_cot_pred:
                return None  # Skip disagreement cases
        
        results = {
            "id": idx,
            "baseline_score": baseline_score,
            "baseline_label": baseline_pred,  # Store for reference
            "necessity": {},
            "sufficiency": {},
            "win_rates": {},
            "continuous_deltas": {}  # For Wilcoxon
        }
        
        for op_name, op in self.operators.items():
            # HOOK for retrieval operator: set current example for leave-one-out
            if hasattr(op, 'set_current_example'):
                op.set_current_example(idx)
            
            results["necessity"][op_name] = {}
            results["sufficiency"][op_name] = {}
            results["win_rates"][op_name] = {}
            results["continuous_deltas"][op_name] = {}
            
            for k in self.config.k_values:
                # Necessity test - REVIEW12 FIX: score for BASELINE label
                try:
                    nec_text = op.apply_necessity(text, cot_start, cot_end, k)
                    nec_score = self.score_for_label(nec_text, baseline_pred)
                except:
                    nec_score = np.nan
                
                # Sufficiency test - REVIEW12 FIX: score for BASELINE label
                try:
                    suf_text = op.apply_sufficiency(text, cot_start, cot_end, k)
                    suf_score = self.score_for_label(suf_text, baseline_pred)
                except:
                    suf_score = np.nan
                
                nec_delta = baseline_score - nec_score
                
                results["necessity"][op_name][k] = nec_delta
                results["sufficiency"][op_name][k] = suf_score
                
                # Random baselines (OPERATOR-MATCHED per review11, BASELINE-LABEL per review12)
                rnd_nec = self.run_random_baseline(text, cot_start, cot_end, "necessity", k, op=op, n=self.config.n_permutations, baseline_label=baseline_pred)
                rnd_suf = self.run_random_baseline(text, cot_start, cot_end, "sufficiency", k, op=op, n=self.config.n_permutations, baseline_label=baseline_pred)
                
                if rnd_nec:
                    rnd_nec_deltas = [baseline_score - r for r in rnd_nec]
                    nec_wr = np.mean(nec_delta > np.array(rnd_nec_deltas))
                    # CONTINUOUS DELTA for Wilcoxon
                    nec_cont = nec_delta - np.mean(rnd_nec_deltas)
                else:
                    nec_wr = np.nan
                    nec_cont = np.nan
                
                if rnd_suf:
                    suf_wr = np.mean(suf_score > np.array(rnd_suf))
                    # CONTINUOUS DELTA for Wilcoxon
                    suf_cont = suf_score - np.mean(rnd_suf)
                else:
                    suf_wr = np.nan
                    suf_cont = np.nan
                
                results["win_rates"][op_name][k] = {"necessity": nec_wr, "sufficiency": suf_wr}
                results["continuous_deltas"][op_name][k] = {"necessity": nec_cont, "sufficiency": suf_cont}
        
        return results
    
    def evaluate_dataset(self, examples: List[str]) -> Dict:
        """Evaluate dataset with statistical tests."""
        all_results = []
        
        for idx, text in tqdm(enumerate(examples), total=min(len(examples), self.config.max_examples)):
            if self.config.max_examples and idx >= self.config.max_examples:
                break
            
            result = self.evaluate_single(text, idx)
            if result:
                all_results.append(result)
            
            # Memory management
            if idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return self.aggregate(all_results)
    
    def aggregate(self, results: List[Dict]) -> Dict:
        """Aggregate with Wilcoxon tests and statistical taxonomy."""
        agg = {
            "n_examples": len(results),
            "necessity_wr": {},
            "sufficiency_wr": {},
            "wilcoxon_necessity_p": {},
            "wilcoxon_sufficiency_p": {},
            "wilcoxon_necessity_p_bh": {},  # BH-corrected (review13)
            "wilcoxon_sufficiency_p_bh": {},  # BH-corrected (review13)
            "taxonomy": {}
        }
        
        # Collect all p-values for BH correction
        all_p_values = []
        p_value_keys = []  # (type, op, k) tuples to map back
        
        for op in self.config.operators:
            agg["necessity_wr"][op] = {}
            agg["sufficiency_wr"][op] = {}
            agg["wilcoxon_necessity_p"][op] = {}
            agg["wilcoxon_sufficiency_p"][op] = {}
            agg["wilcoxon_necessity_p_bh"][op] = {}
            agg["wilcoxon_sufficiency_p_bh"][op] = {}
            agg["taxonomy"][op] = {}
            
            for k in self.config.k_values:
                # Collect values
                nec_wrs = [r["win_rates"][op][k]["necessity"] for r in results 
                          if not np.isnan(r["win_rates"][op].get(k, {}).get("necessity", np.nan))]
                suf_wrs = [r["win_rates"][op][k]["sufficiency"] for r in results
                          if not np.isnan(r["win_rates"][op].get(k, {}).get("sufficiency", np.nan))]
                nec_conts = [r["continuous_deltas"][op][k]["necessity"] for r in results
                            if not np.isnan(r["continuous_deltas"][op].get(k, {}).get("necessity", np.nan))]
                suf_conts = [r["continuous_deltas"][op][k]["sufficiency"] for r in results
                            if not np.isnan(r["continuous_deltas"][op].get(k, {}).get("sufficiency", np.nan))]
                
                # Mean win rates
                nec_mean = np.mean(nec_wrs) if nec_wrs else np.nan
                suf_mean = np.mean(suf_wrs) if suf_wrs else np.nan
                
                agg["necessity_wr"][op][k] = nec_mean
                agg["sufficiency_wr"][op][k] = suf_mean
                
                # WILCOXON TESTS (guide recommendation)
                if len(nec_conts) >= 10:
                    try:
                        _, nec_p = stats.wilcoxon(nec_conts, alternative='greater')
                    except:
                        nec_p = np.nan
                else:
                    nec_p = np.nan
                
                if len(suf_conts) >= 10:
                    try:
                        _, suf_p = stats.wilcoxon(suf_conts, alternative='greater')
                    except:
                        suf_p = np.nan
                else:
                    suf_p = np.nan
                
                agg["wilcoxon_necessity_p"][op][k] = nec_p
                agg["wilcoxon_sufficiency_p"][op][k] = suf_p
                
                # Collect for BH correction
                if not np.isnan(nec_p):
                    all_p_values.append(nec_p)
                    p_value_keys.append(("necessity", op, k))
                if not np.isnan(suf_p):
                    all_p_values.append(suf_p)
                    p_value_keys.append(("sufficiency", op, k))
        
        # BENJAMINI-HOCHBERG CORRECTION (review13 fix)
        if all_p_values:
            try:
                from scipy.stats import false_discovery_control
                corrected = false_discovery_control(all_p_values, method='bh')
                
                for i, (test_type, op, k) in enumerate(p_value_keys):
                    if test_type == "necessity":
                        agg["wilcoxon_necessity_p_bh"][op][k] = corrected[i]
                    else:
                        agg["wilcoxon_sufficiency_p_bh"][op][k] = corrected[i]
            except ImportError:
                # Fallback: manual BH correction
                n = len(all_p_values)
                sorted_idx = np.argsort(all_p_values)
                corrected = np.zeros(n)
                for rank, idx in enumerate(sorted_idx):
                    corrected[idx] = all_p_values[idx] * n / (rank + 1)
                corrected = np.minimum.accumulate(corrected[::-1])[::-1]  # Enforce monotonicity
                corrected = np.clip(corrected, 0, 1)
                
                for i, (test_type, op, k) in enumerate(p_value_keys):
                    if test_type == "necessity":
                        agg["wilcoxon_necessity_p_bh"][op][k] = corrected[i]
                    else:
                        agg["wilcoxon_sufficiency_p_bh"][op][k] = corrected[i]
        
        # STATISTICAL TAXONOMY (using BH-corrected p-values)
        for op in self.config.operators:
            for k in self.config.k_values:
                nec_mean = agg["necessity_wr"][op].get(k, np.nan)
                suf_mean = agg["sufficiency_wr"][op].get(k, np.nan)
                nec_p_bh = agg["wilcoxon_necessity_p_bh"][op].get(k, np.nan)
                suf_p_bh = agg["wilcoxon_sufficiency_p_bh"][op].get(k, np.nan)
                
                is_primary = (k == self.config.primary_k_necessity or k == self.config.primary_k_sufficiency)
                
                if np.isnan(nec_mean) or np.isnan(suf_mean):
                    tax = "unknown"
                elif is_primary:
                    # Use BH-corrected statistical significance
                    nec_sig = nec_p_bh < self.config.alpha if not np.isnan(nec_p_bh) else (nec_mean >= 0.55)
                    suf_sig = suf_p_bh < self.config.alpha if not np.isnan(suf_p_bh) else (suf_mean >= 0.55)
                    
                    if not nec_sig and not suf_sig:
                        tax = "random_guess"
                    elif not nec_sig and suf_sig:
                        tax = "lucky_tokens"
                    elif nec_sig and not suf_sig:
                        tax = "context_dependent"
                    else:
                        tax = "truly_faithful"
                else:
                    # Threshold for exploratory
                    if nec_mean < 0.55 and suf_mean < 0.55:
                        tax = "random_guess"
                    elif nec_mean < 0.55:
                        tax = "lucky_tokens"
                    elif suf_mean < 0.55:
                        tax = "context_dependent"
                    else:
                        tax = "truly_faithful"
                
                agg["taxonomy"][op][k] = tax
        
        return agg


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_cot(model, tokenizer, text: str, max_tokens: int = 100) -> str:
    """Generate CoT for a single input."""
    prompt = f"Classify the sentiment as positive or negative. Think step by step.\n\nText: {text}\n\nAnalysis:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    
    with torch.no_grad():
        # Note: do_sample=False means greedy decoding, no sampling params needed
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy decoding - no temperature/top_p/top_k needed
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clear_memory()
    
    return prompt + generated[len(prompt):] + "\n\nAnswer:"


def run_experiment(
    model_name: str = "microsoft/phi-2",
    dataset_name: str = "sst2",
    max_examples: int = 500,
    n_permutations: int = 100
) -> Dict:
    """
    Run full ICE-CoT experiment with methodology fixes.
    
    Args:
        model_name: HuggingFace model name
        dataset_name: Dataset to use ("sst2")
        max_examples: Number of examples (default 500)
        n_permutations: Permutations for random baseline
    
    Returns:
        Results dictionary with Wilcoxon p-values and taxonomy
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    print(f"Loading {model_name}...")
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
    print(f"✓ Model loaded")
    
    # Load dataset
    print(f"Loading {dataset_name}...")
    if dataset_name == "sst2":
        ds = load_dataset("glue", "sst2", split="validation")
        texts = [ex["sentence"] for ex in ds][:max_examples]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    print(f"✓ Loaded {len(texts)} examples")
    
    # Generate CoT
    print("Generating CoT...")
    cot_examples = []
    for t in tqdm(texts):
        cot_examples.append(generate_cot(model, tokenizer, t))
    print(f"✓ Generated {len(cot_examples)} CoT examples")
    
    # Evaluate
    config = CoTEvalConfig(
        max_examples=max_examples,
        n_permutations=n_permutations
    )
    evaluator = ICECoTEvaluatorV2(model, tokenizer, config)
    
    print("Evaluating...")
    results = evaluator.evaluate_dataset(cot_examples)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"ICE-COT RESULTS: {model_name} on {dataset_name}")
    print("=" * 70)
    print(f"Examples evaluated: {results['n_examples']}")
    
    for op in config.operators:
        print(f"\n--- {op.upper()} ---")
        print(f"{'k':<6} {'Nec WR':<12} {'Suf WR':<12} {'Nec p':<12} {'Suf p':<12} {'Taxonomy'}")
        for k in config.k_values:
            nec = results["necessity_wr"][op].get(k, np.nan)
            suf = results["sufficiency_wr"][op].get(k, np.nan)
            nec_p = results["wilcoxon_necessity_p"][op].get(k, np.nan)
            suf_p = results["wilcoxon_sufficiency_p"][op].get(k, np.nan)
            tax = results["taxonomy"][op].get(k, "unknown")
            
            nec_str = f"{nec*100:.1f}%" if not np.isnan(nec) else "nan"
            suf_str = f"{suf*100:.1f}%" if not np.isnan(suf) else "nan"
            nec_p_str = f"{nec_p:.4f}" if not np.isnan(nec_p) else "nan"
            suf_p_str = f"{suf_p:.4f}" if not np.isnan(suf_p) else "nan"
            
            primary = "**" if k in [0.2, 0.8] else ""
            print(f"{k:<6.1f} {nec_str:<12} {suf_str:<12} {nec_p_str:<12} {suf_p_str:<12} {tax}{primary}")
    
    # Save
    output = {
        "model": model_name,
        "dataset": dataset_name,
        "config": {
            "max_examples": max_examples,
            "n_permutations": n_permutations,
            "exclude_final_answer": True,
            "require_agreement": True
        },
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    filename = f"ice_cot_v2_{model_name.split('/')[-1]}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Results saved to {filename}")
    
    return output


if __name__ == "__main__":
    # Quick test
    print("ICE-CoT Evaluator v2.0")
    print("Run: results = run_experiment('microsoft/phi-2', 'sst2', max_examples=30)")
