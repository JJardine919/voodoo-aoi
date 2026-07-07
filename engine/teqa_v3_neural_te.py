"""
teqa_v3_neural_te.py — RECONSTRUCTED 33-TE transponder chain.

Reconstructed from published spec (Zenodo DOI 18529346 + VDJ_RECOMBINATION doc)
after QNIF_Master.py / teqa_v3_neural_te.py were lost in a desktop crash.

This is NOT the original implementation. It is a clean-room reconstruction
that preserves:
  - The 33 TE ordering (q0..q32)
  - The modification types (enhance/disrupt/invert) per TE family class
  - The non-associative chain property (order matters: A→B→C ≠ A→(B,C))
  - The entanglement couplings between neural TEs (q25..q32) and
    Class I/II qubits via CNOT/CZ-style sign-flip approximations

Used to replace the legacy fallback `entropy_transponders()` in
expert_collapse_bridge.py, which was diagnosed (HOLDPOINT_expert_bridge.md)
as the cause of Voodoo rejecting the bridge at unified=0.095.

If Voodoo's unified > 0.4 after switching to this chain, the STRUCTURE
of the fix is confirmed — specific per-TE functions can be refined later.
If unified remains low, the reconstruction is missing something from the
original's specific per-TE operations and we look harder for the originals.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 33 TE family list with modification types
# Ordering matches qubit indices q0..q32 from the published spec.
# Modification types are assigned by family class conventions:
#   - Class I retrotransposons: mostly enhance (copy+paste = amplification)
#   - Class II DNA transposons: mix of enhance/disrupt/invert
#     (cut+paste = rearrangement, some invert due to inverted repeats)
#   - Neural TEs: context-dependent; silencers invert, synapse/arc enhance
# ============================================================

TE_FAMILIES: List[Tuple[str, str]] = [
    # Class I Retrotransposons (q0..q10)
    ("BEL_Pao",            "enhance"),   # q0
    ("DIRS1",              "enhance"),   # q1
    ("Ty1_copia",          "disrupt"),   # q2
    ("Ty3_gypsy",          "enhance"),   # q3
    ("Ty5",                "disrupt"),   # q4
    ("Alu",                "enhance"),   # q5
    ("LINE",               "enhance"),   # q6
    ("Penelope",           "enhance"),   # q7
    ("RTE",                "disrupt"),   # q8
    ("SINE",               "enhance"),   # q9
    ("VIPER_Ngaro",        "disrupt"),   # q10

    # Class II DNA Transposons (q11..q24)
    ("CACTA",              "enhance"),   # q11
    ("Crypton",            "invert"),    # q12
    ("Helitron",           "enhance"),   # q13
    ("hobo",               "disrupt"),   # q14
    ("I_element",          "invert"),    # q15
    ("Mariner_Tc1",        "enhance"),   # q16
    ("Mavericks_Polinton", "enhance"),   # q17
    ("Mutator",            "disrupt"),   # q18
    ("P_element",          "enhance"),   # q19
    ("PIF_Harbinger",      "enhance"),   # q20
    ("piggyBac",           "disrupt"),   # q21
    ("pogo",               "invert"),    # q22
    ("Rag_like",           "enhance"),   # q23
    ("Transib",            "invert"),    # q24  (RAG1/RAG2 ancestor)

    # Neural TE Families (q25..q32)
    ("L1_Neuronal",        "enhance"),   # q25
    ("L1_Somatic",         "enhance"),   # q26
    ("HERV_Synapse",       "enhance"),   # q27
    ("SVA_Regulatory",     "disrupt"),   # q28
    ("Alu_Exonization",    "disrupt"),   # q29
    ("TRIM28_Silencer",    "invert"),    # q30
    ("piwiRNA_Neural",     "enhance"),   # q31
    ("Arc_Capsid",         "enhance"),   # q32
]

TE_NAMES = [t[0] for t in TE_FAMILIES]
TE_QUBIT = {name: idx for idx, (name, _) in enumerate(TE_FAMILIES)}

MOD_ANGLE_MULT = {
    "enhance": 1.5,
    "disrupt": 0.3,
    "invert":  -1.0,
    "rewire":  1.0,
}

# Entanglement structure (from teqa-v3-neural-te-integration.md)
# Each tuple: (neural_qubit, target_qubit, gate_type)
# We approximate CNOT via sign-conditional flip of target dim,
# CZ via sign-conditional phase flip on both.
ENTANGLEMENT_COUPLINGS: List[Tuple[int, int, str]] = [
    (25, 6,  "CNOT"),  # L1_Neuronal      <-> LINE
    (26, 25, "CNOT"),  # L1_Somatic       <-> L1_Neuronal
    (27, 3,  "CNOT"),  # HERV_Synapse     <-> Ty3_gypsy
    (28, 5,  "CNOT"),  # SVA_Regulatory   <-> Alu
    (29, 5,  "CNOT"),  # Alu_Exonization  <-> Alu
    (30, 25, "CZ"),    # TRIM28_Silencer  <-> L1_Neuronal  (all neural via CZ simplified to q25)
    (30, 26, "CZ"),    # TRIM28_Silencer  <-> L1_Somatic
    (30, 32, "CZ"),    # TRIM28_Silencer  <-> Arc_Capsid
    (31, 25, "CZ"),    # piwiRNA_Neural   <-> L1_Neuronal
    (31, 26, "CZ"),    # piwiRNA_Neural   <-> L1_Somatic
    (32, 3,  "CNOT"),  # Arc_Capsid       <-> Ty3_gypsy
]


# ============================================================
# Main chain function
# ============================================================

def transponder_chain(
    state: np.ndarray,
    te_activations: Optional[Dict[str, float]] = None,
    entropy_scale: float = 1.0,
) -> np.ndarray:
    """
    Apply all 33 TE transformations in sequence to a state vector.

    Each TE applies a Givens-like rotation on a dimension pair indexed by
    its qubit position modulo the state size. The rotation angle is driven
    by the state's global entropy, scaled by the TE's modification type
    (enhance/disrupt/invert) and optional activation strength.

    After the linear chain, entanglement couplings between neural TEs and
    their targets are applied as sign-conditional flips (CNOT approximation)
    or joint sign flips (CZ approximation).

    Args:
        state:            input vector of any dimension >= 2 (typically 24D or 96D)
        te_activations:   optional dict mapping TE name -> activation strength [0,1].
                          Defaults to 1.0 for all if not provided.
        entropy_scale:    multiplier on the base angle (lets callers tune aggressiveness).

    Returns:
        Transformed state of the same shape as input.

    This function is intentionally PURE — no global state, no side effects,
    fully deterministic given inputs. The non-associative character comes
    from the loop ordering, not from randomness.
    """
    s = np.asarray(state, dtype=np.float64).ravel().copy()
    n = len(s)
    if n < 2:
        return s

    # Global entropy drives the base angle (same construction as legacy
    # entropy_transponders, so the fallback and the chain share a
    # calibration point).
    p = np.exp(s - np.max(s))
    p /= p.sum()
    p = np.clip(p, 1e-12, None)
    entropy = float(-np.sum(p * np.log2(p)))
    base_angle = entropy * math.pi / n * entropy_scale

    acts = te_activations or {}

    # --- 1) Linear 33-step chain (order matters) ---
    for idx, (name, mod) in enumerate(TE_FAMILIES):
        d1 = idx % n
        d2 = (idx + 1) % n
        if d1 == d2:
            continue

        activation = float(acts.get(name, 1.0))
        angle = base_angle * MOD_ANGLE_MULT[mod] * activation

        c, sn = math.cos(angle), math.sin(angle)
        a, b = s[d1], s[d2]
        s[d1] = c * a - sn * b
        s[d2] = sn * a + c * b

    # --- 2) Entanglement couplings (neural TEs <-> Class I/II targets) ---
    for q_src, q_tgt, gate in ENTANGLEMENT_COUPLINGS:
        d_src = q_src % n
        d_tgt = q_tgt % n
        if d_src == d_tgt:
            continue

        if gate == "CNOT":
            # Approximation: if source dim is strongly negative, flip target sign.
            # This preserves L2 norm (sign flips are isometries) and encodes the
            # directional dependency without violating energy conservation.
            if s[d_src] < 0:
                s[d_tgt] = -s[d_tgt]
        elif gate == "CZ":
            # Approximation: if BOTH source and target are negative, flip both.
            # CZ introduces a phase, which on real-valued states maps to a joint
            # sign flip when the qubits are both in the |1> (negative) state.
            if s[d_src] < 0 and s[d_tgt] < 0:
                s[d_src] = -s[d_src]
                s[d_tgt] = -s[d_tgt]

    return s


# ============================================================
# Self-test
# ============================================================

def _selftest() -> None:
    rng = np.random.default_rng(42)
    x24 = rng.standard_normal(24)
    x24 *= 4.0 / np.linalg.norm(x24)

    # Chain preserves dimensionality
    y = transponder_chain(x24)
    assert y.shape == x24.shape, "chain must preserve shape"

    # Chain is deterministic
    y2 = transponder_chain(x24)
    assert np.allclose(y, y2), "chain must be deterministic"

    # Order matters: swap two states and see different outputs.
    # This is the non-commutativity property the HOLDPOINT fix requires.
    x_other = x24.copy()
    x_other[0], x_other[1] = x_other[1], x_other[0]
    y_other = transponder_chain(x_other)
    assert not np.allclose(y, y_other), "chain must be non-commutative on input permutations"

    # Chain does NOT equal the legacy single-pass entropy transponder
    # (otherwise the fix is a no-op).
    # Quick sanity: at least some dims should differ meaningfully.
    p = np.exp(x24 - x24.max()); p /= p.sum(); p = np.clip(p, 1e-12, None)
    ent = -p * np.log2(p)
    med = np.median(ent)
    gate = np.where(ent <= med, 1.0, np.exp(-(ent - med)))
    gated = x24 * gate
    phase = float(ent.sum()) * math.pi / 24
    c, ss = math.cos(phase), math.sin(phase)
    legacy = gated.copy()
    for k in range(0, 23, 2):
        a, b = legacy[k], legacy[k + 1]
        legacy[k] = c * a - ss * b
        legacy[k + 1] = ss * a + c * b
    assert not np.allclose(y, legacy), "chain must differ from legacy fallback"

    print("[teqa_v3_neural_te] selftest PASS  input_norm=%.3f output_norm=%.3f" %
          (np.linalg.norm(x24), np.linalg.norm(y)))


if __name__ == "__main__":
    _selftest()
