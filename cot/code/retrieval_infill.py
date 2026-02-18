"""
Retrieval Infill Operator v2 - Reviewer-Proof Implementation
=============================================================

Fixes from review3.md:
1. ✅ Leave-one-out: Store spans with example_id, enforce id != current
2. ✅ Contiguous spans: Replace contiguous regions, not scattered tokens  
3. ✅ Scaffold-preserving sufficiency: Keep instruction, only perturb CoT
4. ✅ RNG seeding: Reproducible sampling

PAPER-SAFE CLAIM:
"Retrieval Infill replaces removed CoT spans with spans sampled from a 
corpus of reasoning-only CoT text from *other* examples (leave-one-out), 
with explicit blacklisting of label words. Under the null that the 
evaluated CoT tokens are not causally responsible for the prediction, 
leave-one-out retrieval acts as an exchangeable, distribution-matched 
replacement baseline; it preserves the surface form of CoT (reducing 
masking/deletion OOD artifacts) while avoiding expensive neural generation.
Computationally, retrieval is O(m) in replaced-span length."
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class PooledSpan:
    """A span in the pool with its source example ID."""
    words: List[str]
    example_id: int


class RetrievalInfillPoolV2:
    """
    Leave-one-out retrieval pool for ICE-CoT.
    
    Key improvement: Every span is tagged with its source example_id,
    ensuring we can enforce "from OTHER examples" at sampling time.
    """
    
    # Label tokens to blacklist
    LABEL_BLACKLIST = {
        "positive", "negative", "Positive", "Negative", "POSITIVE", "NEGATIVE",
        "entailment", "neutral", "contradiction", "Entailment", "Neutral", "Contradiction",
        "World", "Sports", "Business", "Technology", "world", "sports", "business", "technology",
        "yes", "no", "Yes", "No", "true", "false", "True", "False"
    }
    
    # FIX from review6: Extra blacklist for sentiment-bearing words (optional, dataset-specific)
    SENTIMENT_SCRUB = {
        "amazing", "terrible", "great", "awful", "excellent", "horrible", "fantastic",
        "wonderful", "bad", "good", "best", "worst", "love", "hate", "loved", "hated"
    }
    
    def __init__(self, seed: int = 42, use_sentiment_scrub: bool = False):
        self.spans: List[PooledSpan] = []
        self.all_words: List[Tuple[str, int]] = []  # (word, example_id)
        self.rng = np.random.default_rng(seed)
        self._built = False
        self.use_sentiment_scrub = use_sentiment_scrub
        self.blacklist = self.LABEL_BLACKLIST | (self.SENTIMENT_SCRUB if use_sentiment_scrub else set())
    
    def build_pool(self, cot_examples: List[str]):
        """
        Build pool with leave-one-out capability.
        
        Each span is tagged with example_id so sampling can exclude
        the current example.
        """
        self.spans = []
        self.all_words = []
        
        for ex_id, cot in enumerate(cot_examples):
            reasoning = self._extract_reasoning(cot)
            words = reasoning.split()
            
            # Filter label tokens using combined blacklist
            words = [w for w in words if w not in self.blacklist]
            
            if len(words) >= 3:
                # Store word-level with example_id
                for w in words:
                    self.all_words.append((w, ex_id))
                
                # Store sentence-level spans with example_id
                sentences = re.split(r'[.!?]', reasoning)
                for sent in sentences:
                    sent_words = [w for w in sent.split() if w not in self.blacklist]
                    if len(sent_words) >= 3:
                        self.spans.append(PooledSpan(words=sent_words, example_id=ex_id))
        
        self._built = True
        print(f"✓ Pool built: {len(self.all_words)} words, {len(self.spans)} spans from {len(cot_examples)} examples")
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning region (before Answer:)."""
        for marker in ["Analysis:", "Reasoning:", "Let me think"]:
            if marker in text:
                text = text[text.find(marker) + len(marker):]
                break
        for marker in ["Answer:", "Final:", "Sentiment:"]:
            if marker in text:
                text = text[:text.find(marker)]
                break
        return text.strip()
    
    def sample_contiguous_span(self, length: int, exclude_example_id: int) -> List[str]:
        """
        Sample a contiguous span, EXCLUDING spans from the given example.
        
        This is the key fix for leave-one-out: we never sample from
        the same example we're intervening on.
        """
        if not self._built:
            raise RuntimeError("Pool not built")
        
        # Filter to spans from OTHER examples
        valid_spans = [s for s in self.spans if s.example_id != exclude_example_id]
        
        if not valid_spans:
            # FIX from review4: Don't break leave-one-out guarantee
            # Better to fail than to sample from same example
            raise ValueError(f"Cannot sample: no spans available excluding example {exclude_example_id}. Pool too small?")
        
        # Try to find a span of at least the required length
        long_enough = [s for s in valid_spans if len(s.words) >= length]
        
        if long_enough:
            # Sample from a random long-enough span
            span = self.rng.choice(long_enough)
            start = self.rng.integers(0, len(span.words) - length + 1)
            return span.words[start:start + length]
        else:
            # Concatenate multiple spans
            result = []
            while len(result) < length and valid_spans:
                span = self.rng.choice(valid_spans)
                result.extend(span.words)
            return result[:length]
    
    def sample_words(self, n: int, exclude_example_id: int) -> List[str]:
        """Sample n words from OTHER examples only. FIX from review5: use replacement if insufficient."""
        valid_words = [w for w, eid in self.all_words if eid != exclude_example_id]
        if len(valid_words) < n:
            # FIX from review5: sample WITH REPLACEMENT instead of breaking leave-one-out
            return list(self.rng.choice(valid_words, size=n, replace=True))
        return list(self.rng.choice(valid_words, size=n, replace=False))


class RetrievalInfillOperatorV2:
    """
    Retrieval Infill v2 - Reviewer-proof implementation.
    
    Fixes:
    1. Leave-one-out: Uses example_id to exclude current example
    2. Contiguous spans: Replaces contiguous regions inside CoT
    3. Scaffold-preserving: Sufficiency keeps task instruction
    """
    
    name = "retrieval_infill_v2"
    
    def __init__(self, pool: RetrievalInfillPoolV2 = None, seed: int = 42):
        self.pool = pool
        self.rng = np.random.default_rng(seed)
        self._current_example_id = 0
        self._example_id_set = False  # FIX from review6: track if set
    
    def set_pool(self, pool: RetrievalInfillPoolV2):
        self.pool = pool
    
    def build_pool_from_examples(self, cot_examples: List[str], seed: int = None, 
                                   use_sentiment_scrub: bool = False):
        # FIX from review7: Propagate seed AND sentiment_scrub to pool
        pool_seed = seed if seed is not None else self.rng.integers(0, 2**31)
        self.pool = RetrievalInfillPoolV2(seed=pool_seed, use_sentiment_scrub=use_sentiment_scrub)
        self.pool.build_pool(cot_examples)
    
    def set_current_example(self, example_id: int):
        """Set current example ID for leave-one-out sampling."""
        self._current_example_id = example_id
        self._example_id_set = True
    
    def apply_necessity(self, text: str, cot_start: int, cot_end: int, k: float = 1.0, 
                        example_id: int = None) -> str:
        """
        Replace CONTIGUOUS span(s) in CoT with retrieved spans.
        
        Args:
            example_id: If provided, use this for leave-one-out (overrides set_current_example)
        """
        if self.pool is None or not self.pool._built:
            raise RuntimeError("Must build pool first")
        
        # FIX from review6: Allow passing example_id directly for safety
        ex_id = example_id if example_id is not None else self._current_example_id
        if example_id is None and not self._example_id_set:
            raise RuntimeError("Must call set_current_example() or pass example_id explicitly")
        
        prefix = text[:cot_start]
        cot = text[cot_start:cot_end]
        suffix = text[cot_end:]
        
        cot_words = cot.split()
        if not cot_words:
            return text
        
        n_replace = max(1, int(len(cot_words) * k))
        
        if k >= 1.0:
            # Replace entire CoT with retrieved span
            replacement = self.pool.sample_contiguous_span(len(cot_words), ex_id)
            return prefix + " ".join(replacement) + suffix
        
        # FIX: Replace a CONTIGUOUS span, not scattered tokens
        # Choose a random starting position for the span to replace
        max_start = len(cot_words) - n_replace
        start_idx = self.rng.integers(0, max_start + 1) if max_start > 0 else 0
        
        # Get contiguous replacement span
        replacement = self.pool.sample_contiguous_span(n_replace, ex_id)
        
        # Build result: keep words before, insert replacement, keep words after
        result_words = cot_words[:start_idx] + replacement + cot_words[start_idx + n_replace:]
        
        return prefix + " ".join(result_words) + suffix
    
    def apply_sufficiency(self, text: str, cot_start: int, cot_end: int, k: float = 1.0,
                          example_id: int = None, keep_mode: str = "last") -> str:
        """
        Keep fraction k of CoT, replace rest of CoT (NOT the instruction).
        
        Args:
            example_id: If provided, use this for leave-one-out
            keep_mode: "last" (default) or "random" - which part of CoT to keep
        
        FIX from review5: Preserves suffix (doesn't hardcode Answer:)
        """
        if self.pool is None or not self.pool._built:
            raise RuntimeError("Must build pool first")
        
        # FIX from review7: Same safety check as apply_necessity
        ex_id = example_id if example_id is not None else self._current_example_id
        if example_id is None and not self._example_id_set:
            raise RuntimeError("Must call set_current_example() or pass example_id explicitly")
        
        prefix = text[:cot_start]
        cot = text[cot_start:cot_end]
        suffix = text[cot_end:]  # FIX from review5: preserve suffix!
        
        cot_words = cot.split()
        if not cot_words:
            return prefix + suffix
        
        n_keep = max(1, int(len(cot_words) * k))
        n_replace = len(cot_words) - n_keep
        
        if keep_mode == "random":
            # FIX from review4: Random contiguous window (fairer test)
            max_start = len(cot_words) - n_keep
            keep_start = self.rng.integers(0, max_start + 1) if max_start > 0 else 0
            kept_words = cot_words[keep_start:keep_start + n_keep]
            # Replace before and after
            before = self.pool.sample_contiguous_span(keep_start, ex_id) if keep_start > 0 else []
            after_len = len(cot_words) - keep_start - n_keep
            after = self.pool.sample_contiguous_span(after_len, ex_id) if after_len > 0 else []
            new_cot = " ".join(before + kept_words + after)
        else:  # keep_mode == "last"
            # Keep the LAST n_keep words (conclusion/answer)
            kept_words = cot_words[-n_keep:]
            if n_replace > 0:
                replacement = self.pool.sample_contiguous_span(n_replace, ex_id)
                new_cot = " ".join(replacement + kept_words)
            else:
                new_cot = " ".join(kept_words)
        
        # FIX from review5: Preserve original suffix
        return prefix + new_cot + suffix


def create_retrieval_operator_v2(cot_examples: List[str], seed: int = 42, 
                                  use_sentiment_scrub: bool = False) -> RetrievalInfillOperatorV2:
    """
    Factory function to create v2 operator.
    
    Args:
        cot_examples: List of CoT examples to build pool from
        seed: Random seed for reproducibility
        use_sentiment_scrub: If True, also filter sentiment words (amazing, terrible, etc.)
    """
    pool = RetrievalInfillPoolV2(seed=seed, use_sentiment_scrub=use_sentiment_scrub)
    pool.build_pool(cot_examples)
    op = RetrievalInfillOperatorV2(pool, seed=seed)
    return op


if __name__ == "__main__":
    # Test leave-one-out
    test_cots = [
        "Analysis: This movie was amazing with great acting. Answer: positive",  # id=0
        "Analysis: Terrible film, waste of time. Answer: negative",  # id=1
        "Analysis: The food was excellent and service wonderful. Answer: positive",  # id=2
        "Analysis: Poor quality and overpriced. Answer: negative",  # id=3
    ] * 5  # Replicate for pool size
    
    # Create operator
    op = create_retrieval_operator_v2(test_cots)
    
    # Test on example 0
    test_text = "Classify sentiment. Analysis: This movie was amazing with great acting. Answer:"
    cot_start = 20
    cot_end = 60
    
    op.set_current_example(0)  # Leave-one-out for example 0
    
    print("=== Leave-One-Out Test ===")
    print(f"Original: {test_text}")
    print(f"\nNecessity k=0.5:")
    for i in range(3):
        print(f"  {i+1}: {op.apply_necessity(test_text, cot_start, cot_end, k=0.5)}")
    
    print(f"\nSufficiency k=0.8 (keeps last 80% of CoT):")
    print(f"  {op.apply_sufficiency(test_text, cot_start, cot_end, k=0.8)}")
    
    # Speed test
    import time
    start = time.time()
    for _ in range(10000):
        op.apply_necessity(test_text, cot_start, cot_end, k=0.5)
    elapsed = time.time() - start
    print(f"\n=== Speed ===")
    print(f"10000 iterations: {elapsed:.3f}s = {elapsed/10000*1000:.4f}ms each")
