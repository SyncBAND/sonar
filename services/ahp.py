"""
SONAR.AI — Analytic Hierarchy Process (AHP) Weight Derivation
=============================================================

Replaces manually guessed MCDA weights with mathematically derived
weights from structured expert pairwise comparisons.

How it works:
    1. Experts answer: "How much more important is Factor A vs Factor B?"
       using the Saaty scale (1 = equal, 3 = moderate, 5 = strong, 7 = very strong, 9 = extreme)
    2. AHP builds a pairwise comparison matrix
    3. Eigenvector method derives weights
    4. Consistency Ratio (CR) validates the judgments aren't contradictory
       (CR < 0.10 = acceptable, CR > 0.10 = expert should revise)

Supports:
    - Single expert (one matrix → one weight set)
    - Multiple experts (geometric mean aggregation → consensus weights)
    - Profiles saved/loaded from JSON config

Usage:
    from services.ahp import AHPEngine, get_mcda_weights

    # Quick — uses default or configured profile
    weights = get_mcda_weights()
    # → {"strategic_importance": 0.38, "evidence_strength": 0.27, ...}

    # Full — run an AHP session
    engine = AHPEngine(CRITERIA)
    engine.set_comparison("strategic_importance", "evidence_strength", 3)
    result = engine.compute()
    # → {"weights": {...}, "consistency_ratio": 0.04, "is_consistent": True}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# AHP CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# Saaty's Random Index for consistency checking (n = 1..10)
# Source: Saaty, T.L. (1980) "The Analytic Hierarchy Process"
RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}

# Our four MCDA criteria
CRITERIA = [
    "strategic_importance",
    "evidence_strength",
    "growth_momentum",
    "maturity_readiness",
]

# Saaty scale reference (for UI / documentation)
SAATY_SCALE = {
    1: "Equal importance",
    2: "Between equal and moderate",
    3: "Moderate importance",
    4: "Between moderate and strong",
    5: "Strong importance",
    6: "Between strong and very strong",
    7: "Very strong importance",
    8: "Between very strong and extreme",
    9: "Extreme importance",
}

# Config file location
AHP_CONFIG_PATH = os.getenv(
    "AHP_CONFIG_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "ahp_config.json"),
)


# =====================================================================
# AHP ENGINE
# =====================================================================

class AHPEngine:
    """
    Analytic Hierarchy Process weight calculator.

    Parameters
    ----------
    criteria : list[str]
        Names of the criteria to compare.
    """

    def __init__(self, criteria: Optional[List[str]] = None):
        self.criteria = criteria or list(CRITERIA)
        self.n = len(self.criteria)
        self._idx = {name: i for i, name in enumerate(self.criteria)}

        # Initialize comparison matrix as identity (all equal)
        self.matrix = np.ones((self.n, self.n), dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────
    # Pairwise comparisons
    # ─────────────────────────────────────────────────────────────────

    def set_comparison(self, criterion_a: str, criterion_b: str, value: float):
        """
        Set how much more important A is than B.

        Parameters
        ----------
        criterion_a, criterion_b : str
            Criteria names.
        value : float
            Saaty scale value (1-9).
            If A is 'moderately more important' than B → value = 3
            The reciprocal (B vs A = 1/3) is set automatically.
        """
        i = self._idx[criterion_a]
        j = self._idx[criterion_b]
        self.matrix[i, j] = value
        self.matrix[j, i] = 1.0 / value

    def set_matrix(self, comparisons: Dict[str, Dict[str, float]]):
        """
        Set all comparisons from a nested dict.

        Format: {criterion_a: {criterion_b: value, ...}, ...}
        Only upper triangle needed; reciprocals are computed.
        """
        for a, targets in comparisons.items():
            for b, value in targets.items():
                if a in self._idx and b in self._idx:
                    self.set_comparison(a, b, value)

    def set_flat_comparisons(self, pairs: Dict[str, float]):
        """
        Set comparisons from flat dict with 'A_vs_B' keys.

        Format: {"strategic_importance_vs_evidence_strength": 3, ...}
        """
        for key, value in pairs.items():
            parts = key.split("_vs_")
            if len(parts) == 2:
                a, b = parts[0].strip(), parts[1].strip()
                if a in self._idx and b in self._idx:
                    self.set_comparison(a, b, value)

    # ─────────────────────────────────────────────────────────────────
    # Compute weights
    # ─────────────────────────────────────────────────────────────────

    def compute(self) -> Dict[str, Any]:
        """
        Compute AHP weights from the comparison matrix.

        Returns
        -------
        dict with:
            weights : dict[str, float]   — normalized weights summing to 1.0
            consistency_ratio : float    — CR (should be < 0.10)
            is_consistent : bool         — True if CR < 0.10
            lambda_max : float           — principal eigenvalue
            consistency_index : float    — CI
            matrix : list[list[float]]   — the comparison matrix used
        """
        weights = self._eigenvector_weights()
        lambda_max = self._principal_eigenvalue(weights)
        ci = (lambda_max - self.n) / (self.n - 1) if self.n > 1 else 0.0
        ri = RANDOM_INDEX.get(self.n, 1.49)
        cr = ci / ri if ri > 0 else 0.0

        weight_dict = {
            self.criteria[i]: round(float(weights[i]), 4)
            for i in range(self.n)
        }

        return {
            "weights": weight_dict,
            "consistency_ratio": round(cr, 4),
            "is_consistent": cr < 0.10,
            "lambda_max": round(lambda_max, 4),
            "consistency_index": round(ci, 4),
            "matrix": self.matrix.tolist(),
            "criteria": list(self.criteria),
        }

    def _eigenvector_weights(self) -> np.ndarray:
        """
        Compute priority vector using the eigenvector method.

        The principal eigenvector of the comparison matrix gives the weights.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix)

        # Find the principal (largest real) eigenvalue
        real_parts = eigenvalues.real
        max_idx = np.argmax(real_parts)

        # Corresponding eigenvector (take real part)
        principal = eigenvectors[:, max_idx].real

        # Normalize to sum to 1
        principal = np.abs(principal)
        total = principal.sum()
        if total > 0:
            principal = principal / total

        return principal

    def _principal_eigenvalue(self, weights: np.ndarray) -> float:
        """Compute λ_max for consistency checking."""
        weighted_sum = self.matrix @ weights
        ratios = weighted_sum / weights
        return float(np.mean(ratios))

    # ─────────────────────────────────────────────────────────────────
    # Display helpers
    # ─────────────────────────────────────────────────────────────────

    def print_matrix(self):
        """Pretty-print the comparison matrix."""
        header = [""] + [c[:12] for c in self.criteria]
        print(f"{'':>20s}", end="")
        for c in self.criteria:
            print(f"{c[:16]:>16s}", end="")
        print()
        for i, row_name in enumerate(self.criteria):
            print(f"{row_name:>20s}", end="")
            for j in range(self.n):
                val = self.matrix[i, j]
                if val >= 1:
                    print(f"{val:>16.1f}", end="")
                else:
                    print(f"{'1/' + str(int(1/val)):>16s}", end="")
            print()


# =====================================================================
# MULTI-EXPERT AGGREGATION
# =====================================================================

def aggregate_experts(
    expert_matrices: List[np.ndarray],
    criteria: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregate multiple expert comparison matrices using geometric mean.

    This is the standard AHP group decision method (Saaty & Peniwati, 2008).
    The geometric mean preserves the reciprocal property of pairwise comparisons.

    Parameters
    ----------
    expert_matrices : list of np.ndarray
        Each matrix is an n×n pairwise comparison matrix.
    criteria : list[str], optional
        Criteria names.

    Returns
    -------
    dict with aggregated weights and per-expert results.
    """
    criteria = criteria or list(CRITERIA)
    n = len(criteria)
    k = len(expert_matrices)

    if k == 0:
        raise ValueError("Need at least one expert matrix")

    # Geometric mean of all matrices element-wise
    stacked = np.stack(expert_matrices)  # (k, n, n)
    geo_mean = np.exp(np.mean(np.log(stacked), axis=0))  # (n, n)

    # Compute weights from aggregated matrix
    engine = AHPEngine(criteria)
    engine.matrix = geo_mean
    aggregated = engine.compute()

    # Also compute per-expert weights for transparency
    per_expert = []
    for i, mat in enumerate(expert_matrices):
        eng = AHPEngine(criteria)
        eng.matrix = mat
        per_expert.append(eng.compute())

    return {
        "consensus": aggregated,
        "per_expert": per_expert,
        "num_experts": k,
    }


# =====================================================================
# PRESET PROFILES
# =====================================================================

# These are starting points for different stakeholder perspectives.
# Each profile defines pairwise comparisons that can be loaded directly.

AHP_PROFILES: Dict[str, Dict[str, float]] = {

    # ── Default: Balanced strategic scanner ──────────────────────────
    # Strategic importance moderately > evidence > growth = maturity
    "default": {
        "strategic_importance_vs_evidence_strength": 2,
        "strategic_importance_vs_growth_momentum": 3,
        "strategic_importance_vs_maturity_readiness": 2,
        "evidence_strength_vs_growth_momentum": 2,
        "evidence_strength_vs_maturity_readiness": 1,
        "growth_momentum_vs_maturity_readiness": 1,
    },

    # ── Strategy Board: "What matters to Amprion?" dominates ────────
    # Strategic importance strongly > everything else
    "strategy_board": {
        "strategic_importance_vs_evidence_strength": 5,
        "strategic_importance_vs_growth_momentum": 5,
        "strategic_importance_vs_maturity_readiness": 3,
        "evidence_strength_vs_growth_momentum": 1,
        "evidence_strength_vs_maturity_readiness": 0.5,
        "growth_momentum_vs_maturity_readiness": 0.5,
    },

    # ── Innovation Scout: "What's emerging?" — growth matters most ──
    # Growth momentum strongly > strategic; evidence moderate
    "innovation_scout": {
        "strategic_importance_vs_evidence_strength": 0.5,
        "strategic_importance_vs_growth_momentum": 0.25,
        "strategic_importance_vs_maturity_readiness": 1,
        "evidence_strength_vs_growth_momentum": 0.333,
        "evidence_strength_vs_maturity_readiness": 2,
        "growth_momentum_vs_maturity_readiness": 4,
    },

    # ── Grid Planner: "What can we build now?" — maturity + evidence ─
    # Maturity > evidence > strategic > growth
    "grid_planner": {
        "strategic_importance_vs_evidence_strength": 0.5,
        "strategic_importance_vs_growth_momentum": 2,
        "strategic_importance_vs_maturity_readiness": 0.333,
        "evidence_strength_vs_growth_momentum": 3,
        "evidence_strength_vs_maturity_readiness": 0.5,
        "growth_momentum_vs_maturity_readiness": 0.2,
    },

    # ── Risk Manager: "What threatens us?" — strategic + evidence ────
    # Strategic > evidence >> growth = maturity
    "risk_manager": {
        "strategic_importance_vs_evidence_strength": 2,
        "strategic_importance_vs_growth_momentum": 5,
        "strategic_importance_vs_maturity_readiness": 5,
        "evidence_strength_vs_growth_momentum": 3,
        "evidence_strength_vs_maturity_readiness": 3,
        "growth_momentum_vs_maturity_readiness": 1,
    },
}


# =====================================================================
# CONFIG FILE I/O
# =====================================================================

def load_ahp_config(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load AHP configuration from JSON file."""
    path = path or AHP_CONFIG_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Failed to load AHP config from %s: %s", path, e)
        return None


def save_ahp_config(config: Dict[str, Any], path: Optional[str] = None):
    """Save AHP configuration to JSON file."""
    path = path or AHP_CONFIG_PATH
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Saved AHP config to %s", path)


# =====================================================================
# MAIN ENTRY POINT — get_mcda_weights()
# =====================================================================

# Hardcoded fallback (the original manually set weights)
_FALLBACK_WEIGHTS = {
    "strategic_importance": 0.35,
    "evidence_strength":    0.25,
    "growth_momentum":      0.20,
    "maturity_readiness":   0.20,
}

_cached_weights: Optional[Dict[str, float]] = None
_cached_ahp_result: Optional[Dict[str, Any]] = None


def get_mcda_weights(
    profile: Optional[str] = None,
    force_recompute: bool = False,
) -> Dict[str, float]:
    """
    Get MCDA weights, derived via AHP.

    Priority:
        1. ahp_config.json file (if exists) — custom expert comparisons
        2. Named profile from AHP_PROFILES
        3. 'default' profile (AHP-derived, close to original weights)

    Set env var AHP_PROFILE to use a specific profile:
        export AHP_PROFILE=grid_planner

    Returns
    -------
    dict with four weights summing to ~1.0
    """
    global _cached_weights, _cached_ahp_result

    if _cached_weights is not None and not force_recompute:
        return _cached_weights

    # 1. Try config file
    config = load_ahp_config()
    if config and "comparisons" in config:
        log.info("Loading AHP weights from config file: %s", AHP_CONFIG_PATH)
        engine = AHPEngine(CRITERIA)
        engine.set_flat_comparisons(config["comparisons"])
        result = engine.compute()
        _cached_ahp_result = result

        if result["is_consistent"]:
            _cached_weights = result["weights"]
            log.info("AHP weights (CR=%.4f): %s", result["consistency_ratio"],
                     result["weights"])
            return _cached_weights
        else:
            log.warning(
                "AHP config inconsistent (CR=%.4f > 0.10). "
                "Using weights anyway but expert should revise comparisons.",
                result["consistency_ratio"]
            )
            _cached_weights = result["weights"]
            return _cached_weights

    # 2. Try named profile
    profile_name = profile or os.getenv("AHP_PROFILE", "default")
    if profile_name in AHP_PROFILES:
        log.info("Using AHP profile: %s", profile_name)
        engine = AHPEngine(CRITERIA)
        engine.set_flat_comparisons(AHP_PROFILES[profile_name])
        result = engine.compute()
        _cached_ahp_result = result
        _cached_weights = result["weights"]

        if not result["is_consistent"]:
            log.warning("AHP profile '%s' has CR=%.4f (>0.10)",
                        profile_name, result["consistency_ratio"])

        log.info("AHP weights [%s] (CR=%.4f): %s",
                 profile_name, result["consistency_ratio"], result["weights"])
        return _cached_weights

    # 3. Fallback
    log.info("No AHP config found, using hardcoded fallback weights")
    _cached_weights = dict(_FALLBACK_WEIGHTS)
    return _cached_weights


def get_last_ahp_result() -> Optional[Dict[str, Any]]:
    """Get the full AHP computation result (for transparency/debugging)."""
    return _cached_ahp_result


def _get_active_profile_name() -> str:
    """
    Get the name of the currently active AHP profile.
    
    Returns the profile name based on:
    1. AHP_PROFILE environment variable if set
    2. 'custom' if using ahp_config.json file
    3. 'default' otherwise
    """
    # Check if config file exists and has comparisons
    config = load_ahp_config()
    if config and "comparisons" in config:
        return "custom (ahp_config.json)"
    
    # Check environment variable or default
    profile_name = os.getenv("AHP_PROFILE", "default")
    if profile_name in AHP_PROFILES:
        return profile_name
    
    return "default"


def compute_profile_weights(profile_name: str) -> Dict[str, Any]:
    """Compute weights for a named profile. For comparison/UI."""
    if profile_name not in AHP_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. "
                         f"Available: {list(AHP_PROFILES.keys())}")
    engine = AHPEngine(CRITERIA)
    engine.set_flat_comparisons(AHP_PROFILES[profile_name])
    return engine.compute()


def compute_all_profiles() -> Dict[str, Dict[str, Any]]:
    """Compute weights for all preset profiles. For comparison."""
    return {
        name: compute_profile_weights(name)
        for name in AHP_PROFILES
    }
