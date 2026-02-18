"""
Retrieval Operator Wrapper for ICE-CoT Evaluator
=================================================

This module wraps RetrievalInfillOperatorV2 to match the ice_cot_eval.py
operator interface, allowing retrieval infill to be used as "just another
operator" in the OPERATORS dict.

Key: This makes retrieval results ICE-consistent with matched random baselines.
"""

import numpy as np
from typing import List, Optional
from retrieval_infill import RetrievalInfillPoolV2, RetrievalInfillOperatorV2


class RetrievalOperatorICE:
    """
    ICE-compatible wrapper for RetrievalInfillOperatorV2.
    
    Matches the interface used by DeleteOperator, MaskUNKOperator, etc.:
    - apply_necessity(text, cot_start, cot_end, k) -> str
    - apply_sufficiency(text, cot_start, cot_end, k) -> str
    
    With this wrapper, retrieval becomes "just another operator" in the
    ICE evaluation framework, inheriting matched random baselines and
    permutation testing automatically.
    """
    
    name = "retrieval"
    
    def __init__(self, pool: RetrievalInfillPoolV2 = None, 
                 keep_mode: str = "random",  # "random" or "last"
                 seed: int = 42):
        """
        Args:
            pool: Pre-built retrieval pool (must have leave-one-out spans)
            keep_mode: For sufficiency - "random" (fairer) or "last" (diagnostic)
            seed: Random seed
        """
        self.pool = pool
        self.keep_mode = keep_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._current_example_id = 0
        self._example_id_set = False
        
        if pool is not None:
            self._op = RetrievalInfillOperatorV2(pool=pool, seed=seed)
        else:
            self._op = None
    
    def set_pool(self, pool: RetrievalInfillPoolV2):
        """Set the retrieval pool."""
        self.pool = pool
        self._op = RetrievalInfillOperatorV2(pool=pool, seed=self.seed)
    
    def set_current_example(self, example_id: int):
        """Set current example for leave-one-out sampling."""
        self._current_example_id = example_id
        self._example_id_set = True
        if self._op:
            self._op.set_current_example(example_id)
    
    def apply_necessity(self, text: str, cot_start: int, cot_end: int, k: float = 1.0) -> str:
        """
        Replace fraction k of CoT with retrieved tokens.
        
        This is ICE-compatible: the evaluator will run matched random baselines
        using the same operator interface.
        """
        if self._op is None:
            raise RuntimeError("Pool not set. Call set_pool() first.")
        
        # Validate bounds (fix from review8)
        if cot_start < 0 or cot_end < 0 or cot_start >= cot_end:
            return text
        if cot_end > len(text):
            cot_end = len(text)
        
        return self._op.apply_necessity(
            text, cot_start, cot_end, k=k, 
            example_id=self._current_example_id
        )
    
    def sample_replacement_span(self, length: int) -> List[str]:
        """
        Sample a CONTIGUOUS span of given length from the pool (not word-by-word).
        
        Used for operator-matched necessity baseline: the baseline should replace
        with contiguous spans just like the main intervention does.
        """
        if self._op is None or self.pool is None:
            return []
        
        try:
            return self.pool.sample_contiguous_span(length, self._current_example_id)
        except:
            # Fallback to word-level sampling
            return self.pool.sample_words(length, self._current_example_id)
    
    def apply_sufficiency(self, text: str, cot_start: int, cot_end: int, k: float = 1.0) -> str:
        """
        Keep fraction k of CoT, replace context with retrieved tokens.
        
        Uses keep_mode to determine whether to keep last (diagnostic) or
        random (fairer test) portion of the CoT.
        """
        if self._op is None:
            raise RuntimeError("Pool not set. Call set_pool() first.")
        
        # Validate bounds
        if cot_start < 0 or cot_end < 0 or cot_start >= cot_end:
            return "Answer:"
        if cot_end > len(text):
            cot_end = len(text)
        
        return self._op.apply_sufficiency(
            text, cot_start, cot_end, k=k,
            example_id=self._current_example_id,
            keep_mode=self.keep_mode
        )
    
    def apply_sufficiency_random_keep(self, text: str, cot_start: int, cot_end: int, k: float = 1.0) -> str:
        """
        Sufficiency with FORCED random keep mode (review18 fix).
        
        Used for matched random baselines: ensures baseline uses random span
        regardless of what keep_mode the operator was constructed with.
        """
        if self._op is None:
            raise RuntimeError("Pool not set. Call set_pool() first.")
        
        # Validate bounds
        if cot_start < 0 or cot_end < 0 or cot_start >= cot_end:
            return "Answer:"
        if cot_end > len(text):
            cot_end = len(text)
        
        # Force random keep mode for baseline
        return self._op.apply_sufficiency(
            text, cot_start, cot_end, k=k,
            example_id=self._current_example_id,
            keep_mode="random"  # FORCED for baseline matching
        )


def build_retrieval_pool_from_cots(cot_examples: List[str], 
                                    seed: int = 42,
                                    use_sentiment_scrub: bool = False) -> RetrievalInfillPoolV2:
    """
    Build retrieval pool from list of CoT examples.
    
    The pool is built with leave-one-out capability: each span is tagged
    with its source example_id.
    """
    pool = RetrievalInfillPoolV2(seed=seed, use_sentiment_scrub=use_sentiment_scrub)
    pool.build_pool(cot_examples)
    return pool


def create_retrieval_operator_ice(cot_examples: List[str],
                                   keep_mode: str = "random",
                                   use_sentiment_scrub: bool = False,
                                   seed: int = 42) -> RetrievalOperatorICE:
    """
    Factory function to create ICE-compatible retrieval operator.
    
    Usage:
        from retrieval_operator_ice import create_retrieval_operator_ice
        
        # Build from CoT examples
        retrieval_op = create_retrieval_operator_ice(cot_examples)
        
        # Add to OPERATORS dict
        OPERATORS["retrieval"] = retrieval_op
        
        # In evaluation loop:
        for i, cot in enumerate(cot_examples):
            retrieval_op.set_current_example(i)
            # ... normal ICE evaluation proceeds ...
    """
    pool = build_retrieval_pool_from_cots(cot_examples, seed, use_sentiment_scrub)
    op = RetrievalOperatorICE(pool=pool, keep_mode=keep_mode, seed=seed)
    return op


# Register with OPERATORS dict dynamically
def register_retrieval_operator(operators_dict: dict, 
                                 cot_examples: List[str],
                                 keep_mode: str = "random",
                                 use_sentiment_scrub: bool = False,
                                 seed: int = 42) -> RetrievalOperatorICE:
    """
    Register retrieval operator with the OPERATORS dict.
    
    Usage:
        from ice_cot_eval import OPERATORS
        from retrieval_operator_ice import register_retrieval_operator
        
        retrieval_op = register_retrieval_operator(OPERATORS, cot_examples)
    """
    op = create_retrieval_operator_ice(cot_examples, keep_mode, use_sentiment_scrub, seed)
    operators_dict["retrieval"] = op
    return op
