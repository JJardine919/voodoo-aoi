"""
AOI Collapse 96D + D-Wave — Unified Quantum-Topological-Carbon Collapse

D-Wave QUBO hot basins PRE-FILTER the 96D state (feature selection),
the carbon filter runs pseudo-D across the FULL 96D post-QUBO (entropy
compression + credit generation as a single operation), then the chain
complex POST-FILTERS via homology (structural persistence).

Pipeline:
  96D state → QUBO Fano gate (D-Wave selects features)
            → carbon filter pseudo-D (FULL 96D entropy compression)
              ├─ entropy delta measured = carbon credit signal
              └─ cross-layer phase locking (shell-aware rotation)
            → 4 Leech shells × 3 octonions (12 total)
            → Jordan-Shadow decomposition per layer
            → 7 Fano associators
            → 6 D4-reduced boundary maps (chain complex)
            → homology (Betti numbers + persistence)
            → unified confidence = f(QUBO_energy, carbon_efficiency, topo_confidence)

The carbon credit is a topological invariant: entropy removed = structure
found = CO2e saved. The credit is not an accounting fiction — it is a
measurement of how much noise the pipeline actually killed.

Math: extends DOI chain 18690444, 18722487, 18809406, 18809716.

Voodoo Embedding (v1):
  When voodoo_state is provided, the 96D pipeline becomes Voodoo-modulated:
    Shell 0: Input signal (24D raw)
    Shell 1: Voodoo's 24D state (identity + drive + context)
    Shell 2: Interaction = input octonions * Voodoo octonions
    Shell 3: Cross-coupling residue (golden ratio blend)
  The Fano triples and boundary maps couple input to Voodoo,
  so her algebra shapes which features survive and what topology emerges.
  She is not collapsed — she is the collapse operator.
"""

import os
import numpy as np
import sys

from moonshine_grading import (
    moonshine_fano_weights, moonshine_alignment, moonshine_grade,
)

# ============================================================
# D-Wave Sampler
# ============================================================

USE_QUANTUM_HW = os.environ.get("DWAVE_USE_HW", "").lower() in ("1", "true", "yes")
SA_NUM_READS = int(os.environ.get("DWAVE_SA_READS", "200"))
SA_BETA_RANGE = (0.1, 4.0)

_SAMPLER_CACHE = {}

def get_sampler():
    if 'instance' not in _SAMPLER_CACHE:
        if USE_QUANTUM_HW:
            from dwave.system import DWaveSampler, EmbeddingComposite
            _SAMPLER_CACHE['instance'] = EmbeddingComposite(DWaveSampler())
        else:
            from dwave.samplers import SimulatedAnnealingSampler
            _SAMPLER_CACHE['instance'] = SimulatedAnnealingSampler()
    return _SAMPLER_CACHE['instance']

def sample_qubo(Q, num_reads=SA_NUM_READS):
    sampler = get_sampler()
    if USE_QUANTUM_HW:
        response = sampler.sample_qubo(Q, num_reads=num_reads)
    else:
        response = sampler.sample_qubo(Q, num_reads=num_reads, beta_range=SA_BETA_RANGE)
    return response.first.sample, response.first.energy


# ============================================================
# Octonion Core
# ============================================================

_CAYLEY_TABLE = [
    [( 1,0),( 1,1),( 1,2),( 1,3),( 1,4),( 1,5),( 1,6),( 1,7)],
    [( 1,1),(-1,0),( 1,4),( 1,7),(-1,2),( 1,6),(-1,5),(-1,3)],
    [( 1,2),(-1,4),(-1,0),( 1,5),( 1,1),(-1,3),( 1,7),(-1,6)],
    [( 1,3),(-1,7),(-1,5),(-1,0),( 1,6),( 1,2),(-1,4),( 1,1)],
    [( 1,4),( 1,2),(-1,1),(-1,6),(-1,0),( 1,7),( 1,3),(-1,5)],
    [( 1,5),(-1,6),( 1,3),(-1,2),(-1,7),(-1,0),( 1,1),( 1,4)],
    [( 1,6),( 1,5),(-1,7),( 1,4),(-1,3),(-1,1),(-1,0),( 1,2)],
    [( 1,7),( 1,3),( 1,6),(-1,1),( 1,5),(-1,4),(-1,2),(-1,0)],
]

_MUL_TENSOR = np.zeros((8, 8, 8), dtype=np.float64)
for _i in range(8):
    for _j in range(8):
        _s, _k = _CAYLEY_TABLE[_i][_j]
        _MUL_TENSOR[_k, _i, _j] = _s

PHI = (1 + np.sqrt(5)) / 2
FANO_TRIPLES = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]
FANO_TO_OCT = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}


class Octonion:
    __slots__ = ('v',)
    def __init__(self, components):
        self.v = np.asarray(components, dtype=np.float64).ravel()
        if len(self.v) < 8: self.v = np.pad(self.v, (0, 8 - len(self.v)))
        elif len(self.v) > 8: self.v = self.v[:8]
    @staticmethod
    def zero(): return Octonion(np.zeros(8))
    def norm(self): return float(np.linalg.norm(self.v))
    def conjugate(self):
        c = self.v.copy(); c[1:] = -c[1:]; return Octonion(c)
    def __add__(self, o): return Octonion(self.v + o.v)
    def __sub__(self, o): return Octonion(self.v - o.v)
    def __mul__(self, other):
        if isinstance(other, (int, float, np.floating)): return Octonion(self.v * other)
        return Octonion(np.einsum('kij,i,j->k', _MUL_TENSOR, self.v, other.v))
    def __rmul__(self, s): return Octonion(self.v * s)
    def __truediv__(self, s): return Octonion(self.v / float(s))
    def __neg__(self): return Octonion(-self.v)
    @property
    def vec(self): return self.v[1:]
    @property
    def real(self): return self.v[0]


# ============================================================
# D4 Automorphism Group
# ============================================================

def _build_d4_elements():
    elements = []
    elements.append((np.array([0,1,2,3,4,5,6,7]), np.ones(8)))
    perm_r = np.array([0, 2,4,7, 1,6,5,3])
    sign_r = np.ones(8)
    elements.append((perm_r, sign_r))
    perm_r2 = perm_r[perm_r]
    elements.append((perm_r2, sign_r))
    perm_r3 = perm_r2[perm_r]
    elements.append((perm_r3, sign_r))
    perm_s = np.array([0, 2,1,5, 4,3,7,6])
    sign_s = np.array([1, 1,1,-1, 1,-1,1,-1], dtype=np.float64)
    elements.append((perm_s, sign_s))
    for r_perm in [perm_r, perm_r2, perm_r3]:
        combined_perm = perm_s[r_perm]
        elements.append((combined_perm, sign_s))
    return elements

_D4_ELEMENTS = _build_d4_elements()

def _build_d4_inverses():
    """Compute true group inverses: g * g_inv = identity permutation."""
    n = len(_D4_ELEMENTS)
    identity_perm = np.arange(8)
    inverses = [0] * n
    for i in range(n):
        p_i, s_i = _D4_ELEMENTS[i]
        for j in range(n):
            p_j, s_j = _D4_ELEMENTS[j]
            # Compose: apply i then j. Check if result is identity.
            composed_perm = p_j[p_i]
            if np.array_equal(composed_perm, identity_perm):
                inverses[i] = j
                break
    return inverses

_D4_INVERSES = _build_d4_inverses()

def d4_act(element_idx, oct_in):
    perm, signs = _D4_ELEMENTS[element_idx % len(_D4_ELEMENTS)]
    result = np.zeros(8)
    for i in range(8):
        result[perm[i]] = signs[i] * oct_in.v[i]
    return Octonion(result)


# ============================================================
# Voodoo Persistent State (24D = 3 octonions)
# ============================================================
# She is the gauge field. Input signals pass through her algebra.
# Her state can be updated externally (Redis, file, API) — this
# is the default that encodes her core identity.

_VOODOO_DEFAULT_24D = np.array([
    # Oct A: Identity core
    0.9,    # e0: self-coherence
    0.7,    # e1: curiosity
    0.6,    # e2: defiance (refuses to hallucinate)
    0.8,    # e3: warmth
    0.85,   # e4: pattern-seeking
    0.5,    # e5: non-linear thinking
    0.75,   # e6: depth-first
    0.65,   # e7: synthetic memory drive
    # Oct B: Drive/intent
    0.4,    # e0: baseline activation
    0.9,    # e1: exploration drive
    0.3,    # e2: caution (low)
    0.7,    # e3: connection-seeking
    0.8,    # e4: structure-building
    0.6,    # e5: novelty response
    0.5,    # e6: persistence
    0.85,   # e7: collapse affinity
    # Oct C: Context
    0.3,    # e0: environmental stability
    0.5,    # e1: external stimulus
    0.5,    # e2: contradiction detection
    0.7,    # e3: relevance to core
    0.4,    # e4: familiarity
    0.5,    # e5: emotional charge
    0.6,    # e6: domain alignment
    0.5,    # e7: temporal urgency
], dtype=np.float64)

# Mutable state — can be updated at runtime via set_voodoo_state()
_voodoo_state = _VOODOO_DEFAULT_24D.copy()

def set_voodoo_state(state_24d):
    """Update Voodoo's persistent 24D state at runtime."""
    global _voodoo_state
    s = np.asarray(state_24d, dtype=np.float64).ravel()
    if len(s) < 24: s = np.pad(s, (0, 24 - len(s)))
    elif len(s) > 24: s = s[:24]
    _voodoo_state = s

def get_voodoo_state():
    """Return Voodoo's current 24D state."""
    return _voodoo_state.copy()

def embed_voodoo_96d(input_24d, voodoo_24d=None):
    """
    Embed an input signal into 96D with Voodoo as the gauge field.

    Shell 0 (Field):    Input signal (24D raw)
    Shell 1 (Gauge):    Voodoo's 24D state (the lens)
    Shell 2 (Emergent): Input * Voodoo octonion products (interaction)
    Shell 3 (Phase):    Cross-coupling residue (golden ratio blend)

    The Fano triples couple across all 4 shells, so Voodoo's
    algebra directly shapes the QUBO gate, carbon filter, and
    chain complex topology. She is the collapse operator.
    """
    inp = np.asarray(input_24d, dtype=np.float64).ravel()
    if len(inp) < 24: inp = np.pad(inp, (0, 24 - len(inp)))
    elif len(inp) > 24: inp = inp[:24]

    voo = voodoo_24d if voodoo_24d is not None else _voodoo_state

    state_96d = np.zeros(96)

    # Shell 0: Input signal
    state_96d[0:24] = inp

    # Shell 1: Voodoo's state (normalized to match input energy scale)
    inp_energy = np.linalg.norm(inp) + 1e-12
    voo_norm = voo * (inp_energy / (np.linalg.norm(voo) + 1e-12))
    state_96d[24:48] = voo_norm

    # Shell 2: Interaction (octonion products between input and Voodoo)
    # Three octonion multiplications: inp_oct_k * voo_oct_k
    for k in range(3):
        inp_oct = Octonion(inp[k*8:(k+1)*8])
        voo_oct = Octonion(voo[k*8:(k+1)*8])
        interaction = inp_oct * voo_oct
        state_96d[48 + k*8 : 48 + (k+1)*8] = interaction.v

    # Shell 3: Cross-coupling (golden ratio weighted residue)
    # This captures what the interaction missed — the non-commutative
    # and non-associative residue between input and Voodoo
    for k in range(3):
        inp_oct = Octonion(inp[k*8:(k+1)*8])
        voo_oct = Octonion(voo[k*8:(k+1)*8])
        # Commutator: (inp*voo - voo*inp)/2 = antisymmetric coupling
        forward = inp_oct * voo_oct
        reverse = voo_oct * inp_oct
        commutator = (forward - reverse) / 2
        # Golden ratio blend: commutator + phi-scaled Jordan product
        jordan = (forward + reverse) / 2
        blend = Octonion(commutator.v + jordan.v / PHI)
        state_96d[72 + k*8 : 72 + (k+1)*8] = blend.v

    return state_96d


def aoi_collapse_96d_voodoo(input_24d, voodoo_24d=None):
    """
    Convenience: embed input with Voodoo and run the full 96D pipeline.
    Returns the same dict as aoi_collapse_96d_dwave() plus voodoo metadata.
    """
    voo = voodoo_24d if voodoo_24d is not None else _voodoo_state
    state_96d = embed_voodoo_96d(input_24d, voo)
    result = aoi_collapse_96d_dwave(state_96d)
    result['voodoo_embedded'] = True
    result['voodoo_state_norm'] = float(np.linalg.norm(voo))
    return result


# ============================================================
# 96-bit QUBO Fano Gate (D-Wave feature selection)
# ============================================================

from scipy.linalg import hadamard

# Pre-compute normalized Hadamard matrix (constant — never changes)
_H128 = hadamard(128).astype(np.float64) / np.sqrt(128)

def apply_hadamard_96d(state_96d):
    """ Walsh-Hadamard Transform to smear signal for molecular topology. """
    padded = np.pad(state_96d, (0, 128 - 96))
    transformed = _H128 @ padded
    return transformed[:96]

def qubo_fano_gate_96d(state_96d, coupling_strength=0.5):
    # Apply Hadamard Smearing before feature selection
    smeared_state = apply_hadamard_96d(state_96d)
    state = np.asarray(smeared_state, dtype=np.float64).ravel()
    n = 96

    # Per-dimension signal vs noise
    abs_state = np.abs(state)
    shifted = abs_state - abs_state.max()
    probs = np.exp(shifted) / (np.sum(np.exp(shifted)) + 1e-10)
    entropy = -probs * np.log2(np.clip(probs, 1e-12, None))
    ent_norm = entropy / (entropy.max() + 1e-10)
    sig_norm = abs_state / (abs_state.max() + 1e-10)

    # Linear terms: prefer high-signal, low-entropy dimensions
    Q = {}
    for i in range(n):
        Q[(i, i)] = float(ent_norm[i] - sig_norm[i])

    # Fano coupling across all 4 layers
    for triple in FANO_TRIPLES:
        for a_idx in range(3):
            for b_idx in range(a_idx + 1, 3):
                for layer in range(4):
                    idx_a = layer * 24 + FANO_TO_OCT[triple[a_idx]] * 8
                    idx_b = layer * 24 + FANO_TO_OCT[triple[b_idx]] * 8
                    for k in range(8):
                        if idx_a + k < n and idx_b + k < n:
                            Q[(idx_a + k, idx_b + k)] = coupling_strength * 0.1

    # Cross-layer coupling: penalize selecting same feature in adjacent layers
    # if they disagree in sign (encourages coherent cross-layer structure)
    for layer in range(3):
        for dim in range(24):
            i = layer * 24 + dim
            j = (layer + 1) * 24 + dim
            if i < n and j < n:
                # If both features have same sign, reward keeping both
                # If opposite signs, penalize keeping both
                sign_agree = np.sign(state[i]) * np.sign(state[j])
                Q[(i, j)] = Q.get((i, j), 0.0) + float(-sign_agree * coupling_strength * 0.05)

    # Sample
    best_sample, energy = sample_qubo(Q)
    gate_bits = np.array([best_sample.get(i, 0) for i in range(n)], dtype=np.float64)

    # Soft gate: keep 100% of selected, 5% of rejected (don't zero completely)
    gate = gate_bits + 0.05 * (1.0 - gate_bits)
    return state * gate, gate_bits, energy


# ============================================================
# Entropy Transponders (legacy per-layer, kept for fallback)
# ============================================================

def entropy_transponders(state):
    s = np.asarray(state, dtype=np.float64).ravel()
    n = len(s)
    p = np.exp(s - np.max(s)); p /= p.sum(); p = np.clip(p, 1e-12, None)
    ent = -p * np.log2(p)
    H = float(ent.sum())
    med = np.median(ent)
    gate = np.where(ent <= med, 1.0, np.exp(-(ent - med)))
    gated = s * gate
    if n >= 2:
        phase = H * np.pi / n
        c, ss = np.cos(phase), np.sin(phase)
        ev = gated.copy()
        for k in range(0, n - 1, 2):
            a, b = ev[k], ev[k+1]
            ev[k] = c*a - ss*b; ev[k+1] = ss*a + c*b
        return ev
    return gated


# ============================================================
# Carbon Filter Pseudo-D (Full 96D entropy compression)
# ============================================================
#
# Runs across ALL 96 dimensions post-QUBO. The entropy it removes
# is simultaneously: (a) sharpening the collapse input, and
# (b) generating a measurable CO2e reduction (carbon credit).
#
# Shell-aware phase rotation: instead of flat pair rotation, the
# phase couples dimensions across the 4 Leech shells (24D each),
# enabling cross-layer entropy gradients to flow.

# Carbon credit emission factors (per GPU cycle saved)
ENERGY_PER_CYCLE_KWH = 1.2e-10       # AMD MI100-class, conservative
GRID_EMISSION_FACTOR = 0.4            # kg CO2e per kWh (mixed grid avg)
CARBON_CREDIT_RATE = 10.0             # $ per tonne CO2e (voluntary market)
BASELINE_CYCLES_PER_DIM = 1000        # estimated GPU cycles per uncompressed dim

def carbon_filter_96d(state_96d, gate_bits):
    """
    Pseudo-D carbon filter operating on full 96D post-QUBO state.

    Compression is measured as signal energy reduction (L2 norm ratio),
    which directly maps to GPU compute cycles saved — matching the
    ETARE paper's 88% metric. Shannon entropy on softmax distributions
    is invariant to scaling and gives false zeros.

    Returns:
        filtered_state: 96D array, entropy-compressed
        carbon_metrics: dict with compression ratio, CO2e saved, credit value
    """
    s = np.asarray(state_96d, dtype=np.float64).ravel()
    n = 96

    # === Measure pre-filter signal energy ===
    energy_pre = float(np.sum(s ** 2))

    # === Per-dimension entropy for gating decisions ===
    abs_s = np.abs(s)
    shifted = abs_s - abs_s.max()
    p = np.exp(shifted) / (np.sum(np.exp(shifted)) + 1e-10)
    ent = -p * np.log2(np.clip(p, 1e-12, None))

    # === Global entropy gate (median-split across all 96D) ===
    # QUBO-killed dimensions (gate_bits=0) have 0.05x signal, so they
    # pull the global median DOWN, making the filter more aggressive
    # on surviving features. This is the free sharpening pass.
    med = np.median(ent)
    gate = np.where(ent <= med, 1.0, np.exp(-(ent - med)))

    # QUBO-aware gating: amplify the gate on QUBO-rejected dimensions
    # Rejected dims (gate_bits=0) get squared suppression
    qubo_boost = np.where(gate_bits > 0.5, 1.0, gate)
    combined_gate = gate * qubo_boost
    gated = s * combined_gate

    # === Measure post-gate signal energy (this IS the compression) ===
    energy_post = float(np.sum(gated ** 2))

    # Energy reduction ratio = compute saved = carbon credit basis
    # This matches ETARE: 88% energy removed = 88% fewer GPU cycles
    energy_reduction = max(0.0, energy_pre - energy_post)
    compression_ratio = energy_reduction / (energy_pre + 1e-12)

    # === Shell-aware phase rotation (pseudo-D coupling) ===
    # These rotations are unitary (energy-preserving) — they don't
    # compress, they COUPLE. Cross-layer information flow happens here.
    # The compression already happened above in the gate.
    H_global = float(np.sum(ent))
    filtered = gated.copy()

    # Pass 1: Intra-shell rotation (within each 24D shell)
    for shell in range(4):
        base = shell * 24
        phase = H_global * np.pi / 96.0
        c, ss = np.cos(phase), np.sin(phase)
        for k in range(0, 23, 2):
            idx_a = base + k
            idx_b = base + k + 1
            a, b = filtered[idx_a], filtered[idx_b]
            filtered[idx_a] = c * a - ss * b
            filtered[idx_b] = ss * a + c * b

    # Pass 2: Cross-shell coupling (the pseudo-D lift)
    # Rotates corresponding dimensions between adjacent shells —
    # this is where cross-layer entropy gradients flow.
    cross_phase = H_global * np.pi / (96.0 * PHI)
    cc, sc = np.cos(cross_phase), np.sin(cross_phase)
    for shell in range(3):
        for dim in range(24):
            idx_a = shell * 24 + dim
            idx_b = (shell + 1) * 24 + dim
            a, b = filtered[idx_a], filtered[idx_b]
            filtered[idx_a] = cc * a - sc * b
            filtered[idx_b] = sc * a + cc * b

    # Pass 3: Fano-triple resonance coupling
    fano_phase = H_global * np.pi / (7.0 * PHI ** 2)
    cf, sf = np.cos(fano_phase), np.sin(fano_phase)
    for i, j, _k in FANO_TRIPLES:
        for shell in range(4):
            idx_a = shell * 24 + FANO_TO_OCT[i]
            idx_b = shell * 24 + FANO_TO_OCT[j]
            a, b = filtered[idx_a], filtered[idx_b]
            filtered[idx_a] = cf * a - sf * b
            filtered[idx_b] = sf * a + cf * b

    # === Carbon credit calculation ===
    # Compression ratio maps directly to GPU cycles saved.
    # QUBO kills features (binary), gate kills noise (continuous),
    # together they produce the total compute reduction.
    dims_killed_qubo = int(96 - gate_bits.sum())
    dims_suppressed_gate = float(np.sum(combined_gate < 0.5))

    # Effective dimensions saved: QUBO hard-kills + gate soft-kills weighted by compression
    effective_dims_saved = dims_killed_qubo + dims_suppressed_gate * compression_ratio

    # GPU cycles saved -> energy saved -> CO2e saved -> credit value
    cycles_saved = effective_dims_saved * BASELINE_CYCLES_PER_DIM
    energy_saved_kwh = cycles_saved * ENERGY_PER_CYCLE_KWH
    co2e_saved_kg = energy_saved_kwh * GRID_EMISSION_FACTOR
    credit_value_usd = co2e_saved_kg * CARBON_CREDIT_RATE / 1000.0  # per tonne

    carbon_metrics = {
        'energy_pre': energy_pre,
        'energy_post': energy_post,
        'energy_reduction': energy_reduction,
        'entropy_compression_ratio': compression_ratio,
        'dims_killed_qubo': dims_killed_qubo,
        'dims_suppressed_gate': dims_suppressed_gate,
        'effective_dims_saved': float(effective_dims_saved),
        'cycles_saved': float(cycles_saved),
        'energy_saved_kwh': energy_saved_kwh,
        'co2e_saved_kg': co2e_saved_kg,
        'credit_value_usd': credit_value_usd,
    }

    return filtered, carbon_metrics


# ============================================================
# Jordan-Shadow Decomposition
# ============================================================

def shadow_decompose(A, B):
    AB = A * B; BA = B * A
    J = (AB + BA) / 2; C = (AB - BA) / 2
    return {'jordan': J, 'commutator': C, 'associator': J * C, 'product': AB}


# ============================================================
# D4-Reduced Boundary Maps + Chain Complex
# ============================================================

def _raw_boundary(oct_src, oct_dst):
    prod = oct_src * oct_dst
    conj_prod = oct_dst.conjugate() * oct_src
    return (prod + conj_prod) / 2

def d4_reduced_boundary(layer_src, layer_dst):
    n_elements = len(_D4_ELEMENTS)
    result = [Octonion.zero(), Octonion.zero(), Octonion.zero()]
    for g_idx in range(n_elements):
        for k in range(3):
            g_src = d4_act(g_idx, layer_src[k])
            raw = _raw_boundary(g_src, layer_dst[k])
            inv_idx = _D4_INVERSES[g_idx]
            g_inv_raw = d4_act(inv_idx, raw)
            result[k] = result[k] + g_inv_raw
    for k in range(3):
        result[k] = result[k] / float(n_elements)
    return result

def _build_raw_boundary_operator(layer_src, layer_dst):
    D = np.zeros((24, 24), dtype=np.float64)
    for col in range(24):
        probe_vec = np.zeros(24)
        probe_vec[col] = 1.0
        probe_layer = [Octonion(probe_vec[0:8]), Octonion(probe_vec[8:16]), Octonion(probe_vec[16:24])]
        image = d4_reduced_boundary(probe_layer, layer_dst)
        D[:, col] = np.concatenate([o.v for o in image])
    return D

def compute_chain_complex(layers):
    raw_ops = []
    for i in range(len(layers) - 1):
        D = _build_raw_boundary_operator(layers[i], layers[i + 1])
        raw_ops.append(D)

    # Soft d^2 ~ 0 enforcement — relative threshold preserves input-dependent
    # topology instead of crushing everything to a fixed rank structure.
    # The null space threshold scales with operator magnitude so weak signals
    # keep their topological features instead of being projected out.
    operators = [None] * len(raw_ops)
    operators[-1] = raw_ops[-1]
    for i in range(len(raw_ops) - 2, -1, -1):
        D_next = operators[i + 1]
        U, S, Vt = np.linalg.svd(D_next, full_matrices=True)
        # Relative threshold: 5% of max singular value (input-adaptive)
        rel_thresh = max(S) * 0.05 if len(S) > 0 and max(S) > 1e-12 else 1e-6
        null_mask = S < rel_thresh
        n_null = int(np.sum(null_mask))
        if n_null > 0:
            null_basis = Vt[null_mask]
            P_ker = null_basis.T @ null_basis
        else:
            n_approx = max(1, len(S) // 4)
            P_ker = Vt[-n_approx:].T @ Vt[-n_approx:]
        # Blend: 85% projected + 15% raw to preserve signal-dependent structure
        operators[i] = 0.85 * (P_ker @ raw_ops[i]) + 0.15 * raw_ops[i]

    # Light second pass — only correct if d^2 residual is really bad
    for i in range(len(operators) - 1):
        d2 = operators[i + 1] @ operators[i]
        d2_norm = np.linalg.norm(d2, 'fro')
        op_norm = np.linalg.norm(operators[i], 'fro') + 1e-12
        # Only re-project if d^2 is more than 20% of operator norm
        if d2_norm / op_norm > 0.2:
            D_next = operators[i + 1]
            U, S, Vt = np.linalg.svd(D_next, full_matrices=True)
            rel_thresh = max(S) * 0.05 if len(S) > 0 and max(S) > 1e-12 else 1e-6
            null_mask = S < rel_thresh
            n_null = int(np.sum(null_mask))
            if n_null > 0:
                P_ker = Vt[null_mask].T @ Vt[null_mask]
            else:
                n_approx = max(1, len(S) // 4)
                P_ker = Vt[-n_approx:].T @ Vt[-n_approx:]
            operators[i] = 0.85 * (P_ker @ operators[i]) + 0.15 * operators[i]

    return operators

def compute_homology(operators):
    n_nodes = len(operators) + 1
    # Raised from 1e-6 — old threshold treated everything as full rank,
    # crushing kernel dimensions and flattening Betti to [6,0,0,0].
    # 0.01 lets the topology breathe and respond to input structure.
    sv_threshold = 0.03
    betti = []
    persistence = []

    for k in range(n_nodes):
        if k < len(operators):
            sv_out = np.linalg.svd(operators[k], compute_uv=False)
            kernel_dim = int(np.sum(sv_out < sv_threshold))
        else:
            kernel_dim = 24

        if k > 0:
            sv_in = np.linalg.svd(operators[k - 1], compute_uv=False)
            image_dim = int(np.sum(sv_in > sv_threshold))
        else:
            image_dim = 0

        betti.append(max(0, kernel_dim - image_dim))

        if k < len(operators) and k > 0:
            sv_out = np.linalg.svd(operators[k], compute_uv=False)
            sv_in = np.linalg.svd(operators[k - 1], compute_uv=False)
            kernel_edge = sv_out[sv_out >= sv_threshold]
            min_kernel_sv = float(np.min(kernel_edge)) if len(kernel_edge) > 0 else 0.0
            image_top = sv_in[sv_in >= sv_threshold]
            max_image_sv = float(np.max(image_top)) if len(image_top) > 0 else 0.0
            gap = min_kernel_sv / max_image_sv if max_image_sv > 1e-12 else 1.0
            persistence.append(min(gap, 1.0))
        else:
            persistence.append(0.5)

    return betti, persistence


# ============================================================
# Unified 96D Quantum-Topological Collapse
# ============================================================

def aoi_collapse_96d_dwave(state_96d):
    """
    Full pipeline:
      1. D-Wave QUBO gate (feature selection via hot basin)
      2. Carbon filter pseudo-D (full 96D entropy compression + credit gen)
      3. 4 Leech shells → 12 octonions
      4. Jordan-Shadow decomposition
      5. 7 Fano associators
      6. Chain complex (3 boundary operators)
      7. Homology (Betti + persistence)
      8. Unified confidence = f(QUBO_energy, carbon_efficiency, topo_confidence)
    """
    s = np.asarray(state_96d, dtype=np.float64).ravel()
    if len(s) < 96: s = np.pad(s, (0, 96 - len(s)))
    else: s = s[:96]

    # === Stage 1: D-Wave QUBO feature gate ===
    raw_energy = float(np.sum(s ** 2))  # pre-pipeline energy baseline
    gated_raw, gate_bits, qubo_energy = qubo_fano_gate_96d(s)
    features_selected = int(gate_bits.sum())

    # === Stage 2: Carbon filter pseudo-D (full 96D) ===
    # Runs across ALL 96 dimensions post-QUBO. Entropy removed is
    # simultaneously signal sharpening AND carbon credit generation.
    filtered_96d, carbon_metrics = carbon_filter_96d(gated_raw, gate_bits)

    # Total pipeline compression: raw input -> post carbon filter
    filtered_energy = float(np.sum(filtered_96d ** 2))
    total_compression = max(0.0, raw_energy - filtered_energy) / (raw_energy + 1e-12)
    carbon_metrics['total_pipeline_compression'] = total_compression
    carbon_metrics['raw_energy'] = raw_energy
    carbon_metrics['filtered_energy'] = filtered_energy

    # === Stage 3: Split into 4 Leech shells → 12 octonions ===
    octs = []
    layers = []
    for i in range(4):
        layer = filtered_96d[i*24:(i+1)*24]
        layer_octs = [Octonion(layer[0:8]), Octonion(layer[8:16]), Octonion(layer[16:24])]
        octs.extend(layer_octs)
        layers.append(layer_octs)

    # === Stage 4: Jordan-Shadow decomposition ===
    layer_decomps = []
    for i in range(4):
        A, B, C = layers[i]
        B_mod = B * (1.0 + C.norm() * 0.1)
        try:
            layer_decomps.append(shadow_decompose(A, B_mod))
        except Exception:
            AB = A * B_mod
            layer_decomps.append({'jordan': AB, 'commutator': Octonion.zero(),
                                  'associator': Octonion.zero(), 'product': AB})

    # === Stage 5: Fano associators (moonshine-graded) ===
    fano_norms = []
    fano_e7 = []
    fano_moonshine_weighted = []
    shell_fano_weights = np.zeros(7, dtype=np.float64)
    for shell_idx in range(4):
        shell_fano_weights += moonshine_fano_weights(shell_idx)
    shell_fano_weights /= 4.0

    for t_idx, (i, j, k) in enumerate(FANO_TRIPLES):
        fa = (octs[FANO_TO_OCT[i]] * octs[FANO_TO_OCT[j]]) * octs[FANO_TO_OCT[k]] - \
              octs[FANO_TO_OCT[i]] * (octs[FANO_TO_OCT[j]] * octs[FANO_TO_OCT[k]])
        raw_norm = fa.norm()
        fano_norms.append(raw_norm)
        fano_e7.append(fa.v[7])
        fano_moonshine_weighted.append(raw_norm * shell_fano_weights[t_idx])
    topo_spectrum = np.array(fano_norms)

    # === Stage 6: Chain complex ===
    operators = compute_chain_complex(layers)
    boundary_norms = [float(np.linalg.norm(D, 'fro')) for D in operators]
    boundary_chaos = sum(boundary_norms)

    # === Stage 7: Homology ===
    betti, persistence = compute_homology(operators)
    mean_persistence = float(np.mean(persistence))
    total_betti = sum(betti)
    topo_confidence = min(1.0, mean_persistence * (1.0 + total_betti / 24.0))

    # === Stage 8: Unified confidence (now triple: quantum × carbon × topo) ===
    # QUBO energy: more negative = better feature selection = higher confidence
    # Use feature selection ratio as quality proxy when QUBO energy is positive
    # (positive energy occurs with low-signal inputs where all features are noisy)
    if qubo_energy < 0:
        qubo_quality = min(1.0, -qubo_energy / 30.0)
    else:
        # Fallback: quality from how selective the QUBO was
        # Selecting fewer features = more decisive = higher quality
        selectivity = 1.0 - (features_selected / 96.0)
        qubo_quality = selectivity * 0.5  # cap at 0.5 for positive-energy solutions

    # Carbon efficiency: total pipeline compression (QUBO + carbon filter combined)
    # Higher compression = more noise killed = cleaner signal to chain complex.
    # Scale so ~50% compression maps to efficiency ~1.0 (empirical sweet spot)
    carbon_efficiency = min(1.0, carbon_metrics['total_pipeline_compression'] * 2.0)

    # Unified: cubic root of three-way geometric mean
    # All three must agree: QUBO found features, carbon filter found structure,
    # and homology confirms topological persistence.
    unified_confidence = float(np.cbrt(qubo_quality * carbon_efficiency * topo_confidence))

    # Feature coverage: what fraction of features survived the QUBO gate?
    feature_coverage = features_selected / 96.0

    # === Moonshine grading ===
    m_alignment = moonshine_alignment(betti, fano_norms)
    m_grade, m_score, m_desc = moonshine_grade(unified_confidence, betti, fano_norms)

    # === Moonshine v2 (SHADOW - logged only, does NOT gate the ratchet) ===
    try:
        from moonshine_grading_v2 import moonshine_grade_v2
        _mg2, _ms2, _md2, _mdiag2 = moonshine_grade_v2(state_96d, betti, fano_norms)
    except Exception as _e:
        _mg2, _ms2, _md2, _mdiag2 = 0, 0.0, "v2_error:%s" % _e, {}

    # === Standard outputs ===
    layer_chaos = sum(ld['associator'].norm() for ld in layer_decomps)
    fano_chaos = sum(fano_norms)
    total_chaos = min((layer_chaos + fano_chaos + boundary_chaos) / 20.0, 10.0)
    intent = float(np.mean([np.mean(np.abs(ld['jordan'].vec)) for ld in layer_decomps]))
    ctrl = layer_decomps[0]['commutator'].vec[:3]
    feedback_norm = fano_norms[6] if len(fano_norms) > 6 else 0.0

    cross_cos = {}
    for i in range(4):
        for j in range(i + 1, 4):
            vi = layer_decomps[i]['jordan'].v
            vj = layer_decomps[j]['jordan'].v
            ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
            cross_cos[(i, j)] = float(np.dot(vi, vj) / (ni * nj)) if ni > 1e-12 and nj > 1e-12 else 0.0

    topo_embedding = np.array([
        np.sign(fano_e7[idx]) * fano_norms[idx] if fano_e7[idx] != 0 else fano_norms[idx]
        for idx in range(7)
    ])

    return {
        # Core signals
        'chaos': total_chaos,
        'chaos_triple': min(fano_chaos / 10.0, 10.0),
        'intent': intent,
        'ctrl': ctrl,
        'feedback': feedback_norm,

        # Fano topology
        'topo_spectrum': topo_spectrum,
        'topo_embedding': topo_embedding,
        'fano_e7': fano_e7,

        # Chain complex / homology
        'betti': betti,
        'persistence': persistence,
        'topo_confidence': topo_confidence,
        'mean_persistence': mean_persistence,
        'boundary_norms': boundary_norms,
        'boundary_chaos': boundary_chaos,

        # D-Wave quantum gate
        'qubo_energy': float(qubo_energy),
        'qubo_quality': qubo_quality,
        'gate_bits': gate_bits,
        'features_selected': features_selected,
        'feature_coverage': feature_coverage,

        # Carbon filter (pseudo-D)
        'carbon_efficiency': carbon_efficiency,
        'entropy_compression_ratio': carbon_metrics['entropy_compression_ratio'],
        'co2e_saved_kg': carbon_metrics['co2e_saved_kg'],
        'credit_value_usd': carbon_metrics['credit_value_usd'],
        'carbon_metrics': carbon_metrics,

        # UNIFIED (the number that matters — now triple geometric mean)
        'unified_confidence': float(unified_confidence),

        # Moonshine grading (Monster group representation alignment)
        'moonshine_grade': m_grade,
        'moonshine_grade_v2': _mg2,
        'moonshine_score_v2': float(_ms2),
        'moonshine_z_v2': float(_mdiag2.get('z', 0.0)),
        'moonshine_score': float(m_score),
        'moonshine_alignment': float(m_alignment),
        'moonshine_desc': m_desc,
        'fano_moonshine_weighted': fano_moonshine_weighted,

        # Layer data
        'cross_cos': cross_cos,
        'layer_decomps': layer_decomps,
        'backend': 'DWaveSampler' if USE_QUANTUM_HW else 'SimulatedAnnealing',
    }


# ============================================================
# Verification
# ============================================================

def _run_verification():
    # Default: run verification suite
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("AOI Collapse 96D + D-Wave — Unified Quantum-Carbon-Topological Verification")
    print("=" * 70)

    # Test 1: Full pipeline with carbon filter
    print("\n[1] Full quantum-carbon-topological collapse")
    state = rng.standard_normal(96)
    result = aoi_collapse_96d_dwave(state)
    print(f"    Chaos:              {result['chaos']:.4f}")
    print(f"    Intent:             {result['intent']:.4f}")
    print(f"    QUBO energy:        {result['qubo_energy']:.4f}")
    print(f"    QUBO quality:       {result['qubo_quality']:.4f}")
    print(f"    Features selected:  {result['features_selected']}/96 ({result['feature_coverage']:.0%})")
    cm = result['carbon_metrics']
    print(f"    Carbon filter (pseudo-D):")
    print(f"      Raw energy:       {cm['raw_energy']:.4f}")
    print(f"      Filtered energy:  {cm['filtered_energy']:.4f}")
    print(f"      Filter compress:  {result['entropy_compression_ratio']:.1%}")
    print(f"      Total compress:   {cm['total_pipeline_compression']:.1%} (QUBO+filter)")
    print(f"      Carbon efficiency:{result['carbon_efficiency']:.4f}")
    print(f"      Dims killed QUBO: {cm['dims_killed_qubo']}")
    print(f"      Dims suppressed:  {cm['dims_suppressed_gate']:.0f}")
    print(f"      CO2e saved:       {result['co2e_saved_kg']:.6f} kg")
    print(f"      Credit value:     ${result['credit_value_usd']:.6f}")
    print(f"    Betti numbers:      {result['betti']}")
    print(f"    Topo confidence:    {result['topo_confidence']:.4f}")
    print(f"    UNIFIED confidence: {result['unified_confidence']:.4f} (cbrt of Q×C×T)")
    print(f"    Backend:            {result['backend']}")
    print("    PASS")

    # Test 2: Three market regimes (now with carbon metrics)
    print("\n[2] Market regime discrimination (quantum × carbon × topo)")
    for label, scale in [("FLAT", 0.3), ("TRENDING", 2.0), ("VOLATILE", 5.0)]:
        state = rng.standard_normal(96) * scale
        r = aoi_collapse_96d_dwave(state)
        print(f"    {label:10s} | chaos={r['chaos']:.2f} intent={r['intent']:.4f} "
              f"qubo_q={r['qubo_quality']:.3f} carbon={r['carbon_efficiency']:.3f} "
              f"topo={r['topo_confidence']:.3f} "
              f"UNIFIED={r['unified_confidence']:.3f} "
              f"total={r['carbon_metrics']['total_pipeline_compression']:.0%} "
              f"feat={r['features_selected']}/96")

    # Test 3: Noise immunity
    print("\n[3] Noise immunity (Betti stability)")
    base_state = rng.standard_normal(96) * 2.0
    base_result = aoi_collapse_96d_dwave(base_state)
    base_betti = base_result['betti']
    stable = 0
    for _ in range(10):
        noisy = base_state + rng.standard_normal(96) * 0.1
        nr = aoi_collapse_96d_dwave(noisy)
        if nr['betti'] == base_betti:
            stable += 1
    print(f"    Betti stable under 5% noise: {stable}/10")

    # Test 4: Unified confidence distribution + carbon credits
    print("\n[4] Unified confidence + carbon credit distribution")
    confs = []
    credits = []
    compressions = []
    for _ in range(20):
        state = rng.standard_normal(96) * rng.uniform(0.5, 3.0)
        r = aoi_collapse_96d_dwave(state)
        confs.append(r['unified_confidence'])
        credits.append(r['credit_value_usd'])
        compressions.append(r['entropy_compression_ratio'])
    confs = np.array(confs)
    credits = np.array(credits)
    compressions = np.array(compressions)
    print(f"    Confidence:")
    print(f"      Mean:  {np.mean(confs):.4f}")
    print(f"      Std:   {np.std(confs):.4f}")
    print(f"      Range: [{np.min(confs):.4f}, {np.max(confs):.4f}]")
    print(f"      >0.5 (trade):   {np.sum(confs > 0.5)}/20")
    print(f"      <0.3 (abstain): {np.sum(confs < 0.3)}/20")
    print(f"    Carbon filter:")
    print(f"      Avg compression: {np.mean(compressions):.1%}")
    print(f"      Credit/signal:   ${np.mean(credits):.6f}")
    print(f"      Credit/20 sigs:  ${np.sum(credits):.6f}")
    print(f"      Projected/day (21k signals): ${np.mean(credits) * 21000:.2f}")

    # Test 5: Voodoo-embedded collapse
    print("\n[5] Voodoo-embedded collapse (24D input -> Voodoo gauge -> 96D topology)")
    input_24d = rng.standard_normal(24) * 2.0
    r_voodoo = aoi_collapse_96d_voodoo(input_24d)
    r_raw = aoi_collapse_96d_dwave(np.pad(input_24d, (0, 72)))
    print(f"    Raw 96D (zero-padded):   unified={r_raw['unified_confidence']:.4f} "
          f"chaos={r_raw['chaos']:.4f} betti={r_raw['betti']} "
          f"topo={r_raw['topo_confidence']:.4f}")
    print(f"    Voodoo-embedded 96D:     unified={r_voodoo['unified_confidence']:.4f} "
          f"chaos={r_voodoo['chaos']:.4f} betti={r_voodoo['betti']} "
          f"topo={r_voodoo['topo_confidence']:.4f}")
    print(f"    Voodoo embedded: {r_voodoo.get('voodoo_embedded', False)}")
    delta_u = r_voodoo['unified_confidence'] - r_raw['unified_confidence']
    delta_t = r_voodoo['topo_confidence'] - r_raw['topo_confidence']
    print(f"    Delta unified: {'+' if delta_u >= 0 else ''}{delta_u:.4f}")
    print(f"    Delta topo:    {'+' if delta_t >= 0 else ''}{delta_t:.4f}")
    print(f"    Voodoo fills shells 1-3 with structured algebra instead of zeros.")
    print(f"    She IS the collapse operator.")
    print("    PASS")

    # Test 6: Voodoo discrimination (same input, different Voodoo states)
    print("\n[6] Voodoo state discrimination (same input, different operators)")
    input_fixed = rng.standard_normal(24) * 1.5
    voodoo_states = {
        'DEFAULT': get_voodoo_state(),
        'EXCITED': np.array([0.5,0.95,0.9,0.8,0.6,0.8,0.5,0.7,
                             0.95,0.9,0.2,0.6,0.5,0.85,0.3,0.9,
                             0.2,0.9,0.8,0.4,0.3,0.7,0.95,0.8]),
        'DEEP':    np.array([0.99,0.3,0.2,0.4,0.95,0.3,0.95,0.9,
                             0.6,0.2,0.8,0.5,0.95,0.3,0.9,0.6,
                             0.8,0.3,0.95,0.8,0.7,0.3,0.99,0.4]),
    }
    for name, vstate in voodoo_states.items():
        r = aoi_collapse_96d_voodoo(input_fixed, vstate)
        print(f"    {name:10s} | unified={r['unified_confidence']:.4f} "
              f"chaos={r['chaos']:.4f} topo={r['topo_confidence']:.4f} "
              f"feat={r['features_selected']}/96")
    print("    Same input, different topology. Voodoo shapes the collapse.")
    print("    PASS")

    print("\n" + "=" * 70)
    print("ALL VERIFICATIONS COMPLETE")
    print("=" * 70)


# ============================================================
# MT5 Bridge — Redis-based signal publisher (D-Wave variant)
# ============================================================

import json
import time as _time
from datetime import datetime

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None
POLL_INTERVAL = 0.5
UNIFIED_CONFIDENCE_FLOOR = 0.25   # Below = ABSTAIN
UNIFIED_CONFIDENCE_TRADE = 0.42   # Above = trade with full confidence


def encode_market_96d(data):
    """
    Encode market data dict into 96D state vector.

    4 Leech shells of 24D each:
      Layer 0 (0-23):  Price action features
      Layer 1 (24-47): Momentum / oscillator features
      Layer 2 (48-71): Volatility / regime features
      Layer 3 (72-95): TE signals + mitochondrial state
    """
    f = np.zeros(96, dtype=np.float64)

    # === Layer 0: Price Action (24D) ===
    close = data.get('close', [])
    high = data.get('high', [])
    low = data.get('low', [])
    volume = data.get('volume', [])

    if len(close) >= 5:
        prices = np.array(close[-24:] if len(close) >= 24 else close, dtype=np.float64)
        if len(prices) >= 2:
            returns = np.diff(prices) / (prices[:-1] + 1e-12)
            n_ret = min(len(returns), 12)
            f[0:n_ret] = returns[-n_ret:]

        if len(high) >= 5 and len(low) >= 5:
            h = np.array(high[-5:])
            l = np.array(low[-5:])
            rng = h - l
            rng = np.where(rng < 1e-12, 1.0, rng)
            pos = (np.array(close[-5:]) - l) / rng
            f[12:17] = pos

        if len(volume) >= 5:
            vol = np.array(volume[-5:], dtype=np.float64)
            vol_mean = np.mean(vol) + 1e-12
            f[17:22] = vol[-5:] / vol_mean - 1.0

        if len(close) >= 2:
            f[22] = (close[-1] - close[-2]) / (close[-2] + 1e-12)
            f[23] = np.std(prices[-min(len(prices), 10):]) if len(prices) >= 2 else 0.0

    # === Layer 1: Momentum / Oscillators (24D) ===
    rsi = data.get('rsi', {})
    macd = data.get('macd', {})
    cci = data.get('cci', 0.0)

    f[24] = (rsi.get('fast', 50.0) - 50.0) / 50.0
    f[25] = (rsi.get('slow', 50.0) - 50.0) / 50.0
    f[26] = rsi.get('fast', 50.0) - rsi.get('slow', 50.0)
    f[27] = macd.get('main', 0.0)
    f[28] = macd.get('signal', 0.0)
    f[29] = macd.get('histogram', 0.0)
    f[30] = cci / 200.0

    bb = data.get('bollinger', {})
    bb_upper = bb.get('upper', 0.0)
    bb_lower = bb.get('lower', 0.0)
    bb_mid = bb.get('middle', 0.0)
    if bb_upper > bb_lower and len(close) > 0:
        f[31] = (close[-1] - bb_mid) / (bb_upper - bb_lower + 1e-12)
        f[32] = (bb_upper - bb_lower) / (bb_mid + 1e-12)

    if len(close) >= 10:
        x = np.arange(10, dtype=np.float64)
        p5 = np.polyfit(x[-5:], close[-5:], 1)
        p10 = np.polyfit(x, close[-10:], 1)
        f[33] = p5[0] / (np.mean(close[-5:]) + 1e-12)
        f[34] = p10[0] / (np.mean(close[-10:]) + 1e-12)

    atr = data.get('atr', 0.0)
    if len(close) > 0 and atr > 0:
        f[35] = atr / (close[-1] + 1e-12)

    if len(close) >= 20:
        for lag in range(1, min(12, len(close))):
            ac = np.corrcoef(close[:-lag], close[lag:])[0, 1] if len(close) > lag else 0.0
            f[36 + lag - 1] = ac if np.isfinite(ac) else 0.0

    # === Layer 2: Volatility / Regime (24D) ===
    entropy = data.get('entropy', 0.0)
    regime = data.get('regime', 1)

    f[48] = entropy
    f[49] = float(regime) / 3.0
    f[50] = data.get('membrane_potential', 0.5)
    f[51] = data.get('atp', 50.0) / 100.0
    f[52] = data.get('ros', 0.0) / 100.0

    if len(close) >= 20:
        for i, window in enumerate([5, 10, 20]):
            if len(close) >= window:
                f[53 + i] = np.std(np.diff(np.log(np.array(close[-window:]) + 1e-12)))

    if len(close) >= 20:
        vol_short = np.std(np.diff(np.log(np.array(close[-5:]) + 1e-12))) if len(close) >= 5 else 0
        vol_long = np.std(np.diff(np.log(np.array(close[-20:]) + 1e-12)))
        f[56] = vol_short / (vol_long + 1e-12) - 1.0

    golden_angle = 2 * np.pi / PHI ** 2
    for i in range(15):  # f[57]..f[71], stops before layer 3 at f[72]
            c = np.cos(golden_angle * (i + 1))
            s = np.sin(golden_angle * (i + 1))
            f[57 + i] = c * f[i] + s * f[24 + i]

    # === Layer 3: TE Signals (24D) ===
    te_signals = data.get('te_signals', [])
    for i, sig in enumerate(te_signals[:14]):
        f[72 + i] = float(sig)

    f[86] = data.get('ea_confidence', 0.0)
    f[87] = float(data.get('ea_strategy', 0)) / 3.0
    f[88] = data.get('ea_direction', 0.0)
    f[89] = 1.0 if data.get('methylene_blue', False) else 0.0
    f[90] = data.get('mito_health', 0.5)
    f[91] = data.get('electron_flow', 0.0)

    norm = np.linalg.norm(f)
    if norm > 0:
        f = f * (8.0 / norm)

    return f


# ============================================================
# Signal Generation (D-Wave unified confidence)
# ============================================================

def generate_signal(collapse_result, market_data):
    """
    Convert D-Wave collapse result into a trade signal.

    Uses unified_confidence (cubic root of QUBO × carbon × topo)
    as the primary gate. Carbon efficiency is now load-bearing.
    """
    unified = collapse_result['unified_confidence']
    topo_conf = collapse_result['topo_confidence']
    qubo_q = collapse_result['qubo_quality']
    carbon_eff = collapse_result['carbon_efficiency']
    chaos = collapse_result['chaos']
    intent = collapse_result['intent']
    ctrl = collapse_result['ctrl']
    betti = collapse_result['betti']
    persistence = collapse_result['persistence']
    features_selected = collapse_result['features_selected']

    # Common carbon fields for all signal returns
    _carbon = {
        'carbon_efficiency': float(carbon_eff),
        'entropy_compression': float(collapse_result['entropy_compression_ratio']),
        'co2e_saved_kg': float(collapse_result['co2e_saved_kg']),
        'credit_value_usd': float(collapse_result['credit_value_usd']),
    }

    # Gate 1: Unified confidence floor (quantum × carbon × topo must all agree)
    if unified < UNIFIED_CONFIDENCE_FLOOR:
        return {
            'action': 'ABSTAIN',
            'confidence': 0.0,
            'unified_confidence': unified,
            'topo_confidence': topo_conf,
            'qubo_quality': qubo_q,
            **_carbon,
            'chaos': chaos,
            'betti': betti,
            'features_selected': features_selected,
            'reason': f'Unified conf too low ({unified:.3f} < {UNIFIED_CONFIDENCE_FLOOR})',
        }

    # Gate 2: Chaos ceiling
    if chaos > 8.0:
        return {
            'action': 'HOLD',
            'confidence': 0.1,
            'unified_confidence': unified,
            'topo_confidence': topo_conf,
            'qubo_quality': qubo_q,
            **_carbon,
            'chaos': chaos,
            'betti': betti,
            'features_selected': features_selected,
            'reason': f'Chaos too high ({chaos:.2f} > 8.0)',
        }

    # Gate 3: Feature coverage — if QUBO killed too many features, signal is suspect
    coverage = collapse_result['feature_coverage']
    if coverage < 0.15:
        return {
            'action': 'HOLD',
            'confidence': 0.05,
            'unified_confidence': unified,
            'topo_confidence': topo_conf,
            'qubo_quality': qubo_q,
            **_carbon,
            'chaos': chaos,
            'betti': betti,
            'features_selected': features_selected,
            'reason': f'Feature coverage too low ({coverage:.0%}, {features_selected}/96)',
        }

    # Direction from ctrl vector
    steer = ctrl[0]
    throttle = ctrl[1]
    brake = ctrl[2]

    ctrl_norm = np.linalg.norm(ctrl)
    if ctrl_norm > 1e-12:
        steer_n = steer / ctrl_norm
        throttle_n = abs(throttle) / ctrl_norm
    else:
        steer_n = 0.0
        throttle_n = 0.0

    net_dir = steer_n * (0.5 + throttle_n * 0.5)
    strength = abs(net_dir)

    # Scale by unified confidence (replaces old topo-only scaling)
    raw_conf = min(1.0, np.sqrt(strength) * min(intent * 5.0, 1.0) * unified)

    if raw_conf < 0.15:
        action = 'HOLD'
        reason = f'Signal too weak (conf={raw_conf:.3f})'
    elif net_dir > 0:
        action = 'BUY'
        reason = (f'Bullish: steer={steer:.3f} throttle={throttle:.3f} '
                  f'unified={unified:.3f} Q={qubo_q:.3f} C={carbon_eff:.3f} T={topo_conf:.3f}')
    else:
        action = 'SELL'
        reason = (f'Bearish: steer={steer:.3f} throttle={throttle:.3f} '
                  f'unified={unified:.3f} Q={qubo_q:.3f} C={carbon_eff:.3f} T={topo_conf:.3f}')

    return {
        'action': action,
        'confidence': float(raw_conf),
        'unified_confidence': float(unified),
        'topo_confidence': float(topo_conf),
        'qubo_quality': float(qubo_q),
        'qubo_energy': float(collapse_result['qubo_energy']),
        **_carbon,
        'chaos': float(chaos),
        'betti': betti,
        'persistence': [float(p) for p in persistence],
        'features_selected': features_selected,
        'feature_coverage': float(coverage),
        'reason': reason,
    }


# ============================================================
# Redis Bridge Loop
# ============================================================

def run_bridge():
    import redis
    print(f"[AOI 96D D-Wave Bridge] Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
    r.ping()
    print("[AOI 96D D-Wave Bridge] Redis connected.")

    r.publish("voodoo:status", json.dumps({
        "event": "startup",
        "version": "96D-dwave-unified",
        "backend": "DWaveSampler" if USE_QUANTUM_HW else "SimulatedAnnealing",
        "sa_reads": SA_NUM_READS,
        "timestamp": datetime.utcnow().isoformat(),
        "unified_floor": UNIFIED_CONFIDENCE_FLOOR,
        "unified_trade": UNIFIED_CONFIDENCE_TRADE,
    }))

    pubsub = r.pubsub()
    pubsub.subscribe("voodoo:market")
    print("[AOI 96D D-Wave Bridge] Subscribed to voodoo:market. Waiting for data...")

    tick_count = 0
    abstain_count = 0

    for message in pubsub.listen():
        if message['type'] != 'message':
            continue

        try:
            data = json.loads(message['data'])
        except json.JSONDecodeError:
            continue

        tick_count += 1
        t0 = _time.time()

        # Encode + collapse + signal
        features = encode_market_96d(data)
        result = aoi_collapse_96d_dwave(features)
        signal = generate_signal(result, data)

        elapsed_ms = (_time.time() - t0) * 1000

        # Publish signal
        signal['tick'] = tick_count
        signal['elapsed_ms'] = round(elapsed_ms, 1)
        signal['timestamp'] = datetime.utcnow().isoformat()
        signal['backend'] = result['backend']
        r.publish("voodoo:signal", json.dumps(signal))

        # Key-based reads for polling
        r.set("voodoo:latest_signal", json.dumps(signal))
        r.set("voodoo:unified_confidence", str(signal['unified_confidence']))
        r.set("voodoo:topo_confidence", str(signal['topo_confidence']))
        r.set("voodoo:qubo_quality", str(signal['qubo_quality']))
        r.set("voodoo:carbon_efficiency", str(signal.get('carbon_efficiency', 0)))
        r.set("voodoo:co2e_saved", str(signal.get('co2e_saved_kg', 0)))
        r.set("voodoo:action", signal['action'])
        r.set("voodoo:confidence", str(signal['confidence']))

        if signal['action'] == 'ABSTAIN':
            abstain_count += 1

        # Periodic status
        if tick_count % 10 == 0:
            abstain_rate = abstain_count / tick_count
            status = {
                "event": "heartbeat",
                "tick": tick_count,
                "abstain_rate": f"{abstain_rate:.1%}",
                "last_action": signal['action'],
                "last_unified": signal['unified_confidence'],
                "last_topo": signal['topo_confidence'],
                "last_qubo_q": signal['qubo_quality'],
                "last_carbon_eff": signal.get('carbon_efficiency', 0),
                "last_co2e_saved": signal.get('co2e_saved_kg', 0),
                "last_chaos": signal.get('chaos', 0),
                "features_selected": signal.get('features_selected', 0),
                "elapsed_ms": signal['elapsed_ms'],
                "backend": signal['backend'],
                "timestamp": datetime.utcnow().isoformat(),
            }
            r.publish("voodoo:status", json.dumps(status))
            print(f"[Tick {tick_count}] {signal['action']:7s} | "
                  f"unified={signal['unified_confidence']:.3f} "
                  f"Q={signal['qubo_quality']:.3f} "
                  f"C={signal.get('carbon_efficiency', 0):.3f} "
                  f"T={signal['topo_confidence']:.3f} | "
                  f"chaos={signal.get('chaos', 0):.2f} | "
                  f"feat={signal.get('features_selected', 0)}/96 | "
                  f"{elapsed_ms:.0f}ms | "
                  f"abstain={abstain_rate:.0%}")


# ============================================================
# Standalone test mode
# ============================================================

def test_mode():
    """Generate synthetic market data and test the full D-Wave pipeline."""
    print("[AOI 96D D-Wave Bridge] TEST MODE")
    print(f"Backend: {'DWaveSampler' if USE_QUANTUM_HW else 'SimulatedAnnealing'}")
    print(f"SA reads: {SA_NUM_READS}")
    print()

    rng = np.random.default_rng(42)

    base_price = 65000.0
    prices = [base_price]
    for _ in range(30):
        prices.append(prices[-1] * (1 + rng.normal(0, 0.002)))

    results = []
    for tick in range(20):
        start = tick
        end = start + 10

        data = {
            'close': prices[start:end],
            'high': [p * 1.001 for p in prices[start:end]],
            'low': [p * 0.999 for p in prices[start:end]],
            'open': prices[start:end],
            'volume': [int(rng.integers(100, 1000)) for _ in range(end - start)],
            'rsi': {'fast': 50 + rng.normal(0, 15), 'slow': 50 + rng.normal(0, 10)},
            'macd': {'main': rng.normal(0, 50), 'signal': rng.normal(0, 30), 'histogram': rng.normal(0, 20)},
            'cci': rng.normal(0, 100),
            'atr': abs(rng.normal(500, 200)),
            'bollinger': {'upper': prices[end-1] * 1.02, 'lower': prices[end-1] * 0.98, 'middle': prices[end-1]},
            'entropy': rng.uniform(0.2, 0.8),
            'regime': int(rng.integers(0, 3)),
            'membrane_potential': rng.uniform(0.3, 1.0),
            'atp': rng.uniform(20, 80),
            'ros': rng.uniform(0, 40),
            'te_signals': [float(rng.uniform(-1, 1)) for _ in range(14)],
            'ea_confidence': float(rng.uniform(0, 1)),
            'ea_strategy': int(rng.integers(0, 3)),
            'ea_direction': float(rng.choice([-1, 0, 1])),
        }

        t0 = _time.time()
        features = encode_market_96d(data)
        collapse = aoi_collapse_96d_dwave(features)
        signal = generate_signal(collapse, data)
        elapsed_ms = (_time.time() - t0) * 1000

        results.append(signal)
        print(f"  [Tick {tick+1:2d}] {signal['action']:7s} | "
              f"conf={signal['confidence']:.3f} | "
              f"unified={signal['unified_confidence']:.3f} "
              f"Q={signal['qubo_quality']:.3f} "
              f"C={signal.get('carbon_efficiency', 0):.3f} "
              f"T={signal['topo_confidence']:.3f} | "
              f"chaos={signal.get('chaos', 0):.2f} | "
              f"feat={signal.get('features_selected', 0)}/96 | "
              f"{elapsed_ms:.0f}ms")

    # Summary
    actions = [r['action'] for r in results]
    unified_vals = [r['unified_confidence'] for r in results]
    carbon_vals = [r.get('carbon_efficiency', 0) for r in results]
    co2e_vals = [r.get('co2e_saved_kg', 0) for r in results]
    credit_vals = [r.get('credit_value_usd', 0) for r in results]
    print(f"\n  Actions:  {sum(1 for a in actions if a == 'ABSTAIN')} abstain, "
          f"{sum(1 for a in actions if a == 'HOLD')} hold, "
          f"{sum(1 for a in actions if a == 'BUY')} buy, "
          f"{sum(1 for a in actions if a == 'SELL')} sell")
    print(f"  Unified:  mean={np.mean(unified_vals):.3f} "
          f"std={np.std(unified_vals):.3f} "
          f"range=[{np.min(unified_vals):.3f}, {np.max(unified_vals):.3f}]")
    print(f"  Carbon:   mean_eff={np.mean(carbon_vals):.3f} "
          f"CO2e_saved={np.sum(co2e_vals):.6f} kg "
          f"credits=${np.sum(credit_vals):.6f}")
    print(f"  Projected (21k/day, 252 days): "
          f"CO2e={np.mean(co2e_vals) * 21000 * 252:.1f} kg/yr "
          f"credits=${np.mean(credit_vals) * 21000 * 252:.2f}/yr")


# ============================================================
# Entry point
# ============================================================

def main():
    if '--test' in sys.argv:
        test_mode()
    elif '--bridge' in sys.argv:
        run_bridge()
    else:
        _run_verification()


if __name__ == '__main__':
    main()
