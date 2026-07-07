"""
Moonshine Grading — Monster Group Representation Weights for AOI Collapse

Encodes the monstrous moonshine structure (McKay/Conway-Norton/Borcherds)
into coupling weights for the 96D = 4×24 Leech shell architecture.

The J-invariant Fourier coefficients decompose into irreducible Monster
representations with small non-negative coefficients:
    J(τ) = 1/q + 744 + 196884q + 21493760q² + 864299970q³ + ...

Each coefficient maps to a shell in the 96D collapse:
    Shell 0 (input):         J_0 = 1           = r1
    Shell 1 (Voodoo gauge):  J_1 = 196884      = r1 + r2
    Shell 2 (interaction):   J_2 = 21493760    = r1 + r2 + r3
    Shell 3 (cross-coupling):J_3 = 864299970   = 2r1 + 2r2 + r3 + r4

The Monster representation dimensions r_n:
    r1 = 1, r2 = 196883, r3 = 21296876, r4 = 842609326,
    r5 = 18538750076, r6 = 19360062527, r7 = 293553734298

Math: extends DOI chain via monstrous moonshine (McKay 1978,
Conway-Norton 1979, Borcherds 1992 Fields Medal proof).
"""

import numpy as np


# ============================================================
# Monster Group Constants
# ============================================================

# J-invariant Fourier coefficients (A014708)
# J(τ) = 1/q + 744 + Σ J_n q^n
J_COEFFICIENTS = np.array([
    1,              # J_0: constant (identity)
    196884,         # J_1: q^1
    21493760,       # J_2: q^2
    864299970,      # J_3: q^3
    20245856256,    # J_4: q^4
    333202640600,   # J_5: q^5
], dtype=np.float64)

# Irreducible Monster representation dimensions (A001379)
MONSTER_REPS = np.array([
    1,              # r1: trivial representation
    196883,         # r2: smallest faithful
    21296876,       # r3
    842609326,      # r4
    18538750076,    # r5
    19360062527,    # r6
    293553734298,   # r7
], dtype=np.float64)

# The moonshine decompositions:
# J_0 = 1         = r1
# J_1 = 196884    = r1 + r2
# J_2 = 21493760  = r1 + r2 + r3
# J_3 = 864299970 = 2r1 + 2r2 + r3 + r4
# Multiplicity matrices: MOONSHINE_DECOMP[j, n] = multiplicity of r_n in J_j
MOONSHINE_DECOMP = np.array([
    [1, 0, 0, 0, 0, 0, 0],  # J_0 = r1
    [1, 1, 0, 0, 0, 0, 0],  # J_1 = r1 + r2
    [1, 1, 1, 0, 0, 0, 0],  # J_2 = r1 + r2 + r3
    [2, 2, 1, 1, 0, 0, 0],  # J_3 = 2r1 + 2r2 + r3 + r4
], dtype=np.float64)


# ============================================================
# Shell Coupling Weights (log-normalized J-coefficients)
# ============================================================

# Log-scale the J-coefficients for coupling weights.
# Raw values span 1 to 10^9 — log brings them to human scale.
# Normalized so Shell 0 (identity) = 1.0
_LOG_J = np.log1p(J_COEFFICIENTS[:4])  # First 4 for 4 shells
SHELL_WEIGHTS = _LOG_J / _LOG_J[0]     # Relative to identity shell

# Cross-shell coupling matrix: w[i,j] = geometric mean of shell weights
# This gives the natural moonshine-graded coupling strength between shells
CROSS_SHELL_COUPLING = np.zeros((4, 4), dtype=np.float64)
for _i in range(4):
    for _j in range(4):
        if _i != _j:
            CROSS_SHELL_COUPLING[_i, _j] = np.sqrt(
                SHELL_WEIGHTS[_i] * SHELL_WEIGHTS[_j]
            )
# Normalize so max coupling = 1.0
_max_coupling = CROSS_SHELL_COUPLING.max()
if _max_coupling > 0:
    CROSS_SHELL_COUPLING /= _max_coupling


# ============================================================
# Moonshine Fano Weights
# ============================================================

def moonshine_fano_weights(shell_idx):
    """
    Return 7 Fano triple weights for a given shell, graded by moonshine.

    The Monster representation structure gives each shell a different
    "algebraic depth" — Shell 0 has only r1 (trivial), Shell 3 has
    four representations active. More active representations = more
    Fano triples should be weighted higher (richer non-associativity).

    Returns: 7-element array of weights for the 7 Fano triples.
    """
    if shell_idx >= len(MOONSHINE_DECOMP):
        shell_idx = len(MOONSHINE_DECOMP) - 1

    # How many Monster representations are active in this shell?
    active_reps = MOONSHINE_DECOMP[shell_idx]
    total_multiplicity = active_reps.sum()

    # Base weight from shell's J-coefficient (log-normalized)
    base = SHELL_WEIGHTS[shell_idx]

    # Distribute across 7 Fano triples proportional to representation depth
    # Each triple gets base weight × (1 + fraction of active reps)
    n_active = np.count_nonzero(active_reps)
    rep_depth = n_active / len(MONSTER_REPS)

    weights = np.ones(7, dtype=np.float64) * base * (1.0 + rep_depth)

    # The 7th Fano triple (7,1,3) closes the cycle — weight it by
    # total multiplicity (moonshine grading of algebraic closure)
    weights[6] *= (1.0 + total_multiplicity / 10.0)

    return weights / weights.max()  # Normalize to [0, 1]


# ============================================================
# QUBO Moonshine Priors
# ============================================================

def moonshine_qubo_prior(dim_idx):
    """
    Return a moonshine-graded prior for a QUBO dimension.

    Dimensions in lower shells (closer to identity representation)
    get a slight preference — the Monster's grading says these are
    algebraically more stable. Higher shells (richer representation
    structure) are more speculative but potentially more informative.

    Args:
        dim_idx: dimension index 0-95

    Returns:
        float: prior weight in [-0.1, 0.1] range (added to QUBO linear term)
    """
    shell = dim_idx // 24
    local_dim = dim_idx % 24
    octonion_idx = local_dim // 8  # Which of 3 octonions in the shell

    # Shell grading: identity shell is most stable
    shell_prior = -0.02 * (1.0 / SHELL_WEIGHTS[shell])

    # Representation depth bonus: shells with more active Monster reps
    # have richer structure — slight bonus for keeping their features
    n_active = np.count_nonzero(MOONSHINE_DECOMP[shell])
    depth_bonus = -0.01 * (n_active / len(MONSTER_REPS))

    return shell_prior + depth_bonus


def moonshine_qubo_coupling(dim_a, dim_b, base_coupling=0.1):
    """
    Moonshine-modulated coupling between two QUBO dimensions.

    Cross-shell couplings are weighted by the moonshine cross-shell
    coupling matrix. Same-shell couplings use the shell's own weight.

    Args:
        dim_a, dim_b: dimension indices 0-95
        base_coupling: base coupling strength

    Returns:
        float: modulated coupling strength
    """
    shell_a = dim_a // 24
    shell_b = dim_b // 24

    if shell_a == shell_b:
        # Intra-shell: weight by shell's own moonshine grade
        return base_coupling * SHELL_WEIGHTS[shell_a] / SHELL_WEIGHTS.max()
    else:
        # Cross-shell: use moonshine coupling matrix
        return base_coupling * CROSS_SHELL_COUPLING[shell_a, shell_b]


# ============================================================
# Moonshine Alignment Metric
# ============================================================

def moonshine_alignment(betti, fano_norms):
    """
    Measure how well the collapse topology aligns with moonshine grading.

    The Monster's graded representation predicts sparse topology:
    - b0 should be nonzero (connected components from identity rep)
    - b1, b2 should be small (few cycles — algebraic closure)
    - b3 can be nonzero (cavities from higher representations)

    Fano norms should grade by shell weight — higher shells should
    show more non-associativity (richer Monster representations).

    Returns:
        float: alignment score in [0, 1], higher = more moonshine-like
    """
    # Betti alignment: moonshine predicts b0 > 0, sparse higher Betti
    b = np.array(betti, dtype=np.float64)
    betti_score = 1.0
    if b[0] == 0:
        betti_score *= 0.5  # No connected components is anti-moonshine
    if b[1] > b[0]:
        betti_score *= 0.7  # More cycles than components is unusual
    if b[3] > 0:
        betti_score *= 1.2  # Higher cavities are moonshine-compatible
    betti_score = min(1.0, betti_score)

    # Fano alignment: norms should roughly grade by shell participation
    if len(fano_norms) >= 7:
        fn = np.array(fano_norms[:7], dtype=np.float64)
        fn_norm = fn / (fn.max() + 1e-12)

        # The 7 Fano triples map to octonion indices 0-6
        # Triples involving higher indices should show more non-associativity
        # (because they couple more Monster representations)
        expected_grade = np.array([0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 1.0])
        corr_matrix = np.corrcoef(fn_norm, expected_grade)
        fano_corr = float(corr_matrix[0, 1]) if np.isfinite(corr_matrix[0, 1]) else 0.0
        fano_score = max(0.0, (fano_corr + 1.0) / 2.0)  # Map [-1,1] to [0,1]
    else:
        fano_score = 0.5

    return float(np.sqrt(betti_score * fano_score))


def moonshine_grade(unified_confidence, betti, fano_norms):
    """
    Assign a moonshine grade level to the collapse result.

    Maps the signal to which J-invariant coefficient level it corresponds to:
        Grade 0: J_0 = 1           (trivial, identity-level)
        Grade 1: J_1 = 196884      (first non-trivial Monster structure)
        Grade 2: J_2 = 21493760    (three representations active)
        Grade 3: J_3 = 864299970   (full 4-rep structure)

    Higher grade = deeper algebraic structure detected in the signal.
    """
    alignment = moonshine_alignment(betti, fano_norms)
    total_betti = sum(betti)

    # Grade thresholds — recalibrated 2026-04-28.
    # Original (1 + total_betti/24) rewarded high Betti count, which noise has.
    # Fix: Betti SPECIFICITY. Moonshine predicts b0+b3 dominant, b1+b2 sparse.
    if total_betti == 0:
        score = unified_confidence * alignment * 0.5
    else:
        moonshine_betti = (betti[0] + betti[3]) / total_betti
        betti_multiplier = 1.0 + max(0.0, moonshine_betti - 0.25) / 0.75
        score = unified_confidence * alignment * betti_multiplier

    if score > 0.7:
        return 3, score, "J_3: Full moonshine structure (2r1+2r2+r3+r4)"
    elif score > 0.45:
        return 2, score, "J_2: Three-representation depth (r1+r2+r3)"
    elif score > 0.25:
        return 1, score, "J_1: First non-trivial (r1+r2 = 1+196883)"
    else:
        return 0, score, "J_0: Identity level (r1 only)"


# ============================================================
# Verification
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Moonshine Grading — Monster Group Constants Verification")
    print("=" * 60)

    # Verify J-coefficient decompositions
    print("\n[1] J-invariant decomposition check")
    for j_idx in range(4):
        computed = float(MOONSHINE_DECOMP[j_idx] @ MONSTER_REPS)
        actual = J_COEFFICIENTS[j_idx]
        match = "PASS" if abs(computed - actual) < 1 else "FAIL"
        print(f"    J_{j_idx} = {int(actual):>15,d}  decomp = {int(computed):>15,d}  {match}")

    # Shell weights
    print("\n[2] Shell weights (log-normalized)")
    for i, w in enumerate(SHELL_WEIGHTS):
        print(f"    Shell {i}: {w:.4f}")

    # Cross-shell coupling
    print("\n[3] Cross-shell coupling matrix")
    for i in range(4):
        row = " ".join(f"{CROSS_SHELL_COUPLING[i,j]:.3f}" for j in range(4))
        print(f"    Shell {i}: [{row}]")

    # Fano weights per shell
    print("\n[4] Moonshine Fano weights per shell")
    for shell in range(4):
        w = moonshine_fano_weights(shell)
        print(f"    Shell {shell}: [{', '.join(f'{x:.3f}' for x in w)}]")

    # QUBO priors
    print("\n[5] QUBO moonshine priors (sample dimensions)")
    for dim in [0, 12, 24, 48, 72, 95]:
        p = moonshine_qubo_prior(dim)
        shell = dim // 24
        print(f"    dim={dim:2d} (shell {shell}): prior={p:+.4f}")

    # Moonshine alignment examples
    print("\n[6] Moonshine alignment examples")
    test_cases = [
        ([6, 0, 0, 0], [0.5]*7, "flat topology"),
        ([3, 0, 0, 1], [0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.8], "Voodoo-like"),
        ([6, 1, 0, 2], [0.2, 0.3, 0.4, 0.6, 0.5, 0.3, 0.7], "moonshine read"),
        ([10, 0, 0, 4], [0.1]*7, "high b0+b3"),
    ]
    for betti, fano, label in test_cases:
        a = moonshine_alignment(betti, fano)
        g, s, desc = moonshine_grade(0.5, betti, fano)
        print(f"    {label:20s} betti={betti} align={a:.3f} grade={g} ({desc})")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
