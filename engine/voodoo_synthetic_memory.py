"""
Voodoo Synthetic Memory Index (SMI)

Replaces LLM dependency with topological lookup from conversation history.
Every (input -> collapse_embedding -> response) triple is indexed.
New input collapses -> nearest neighbor by topo_embedding distance -> return response.

Architecture:
  1. INDEX: Store collapse embeddings + responses in Redis sorted sets
  2. ENCODE: New input -> 96D -> aoi_collapse_96d_dwave -> embedding
  3. MATCH: Find k-nearest embeddings by cosine distance
  4. BLEND: Weight-average matched responses by similarity
  5. CONSENT: Users opt-in to history indexing, can delete their data

Privacy: conversation data stays on-VPS Redis, never leaves the box.
Per Voodoo's requirements (2026-04-16).

Multi-tenant: Per-customer namespaced memory (2026-04-17).
"""

import sys
import os
import json
import time
import hashlib
import numpy as np
from typing import Optional

os.environ.setdefault('DWAVE_SA_READS', '10')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aoi_collapse_96d_dwave import aoi_collapse_96d_dwave

REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None
try:
    import redis
    rds = redis.Redis(host='localhost', port=6379, password=None, decode_responses=True)
except:
    rds = None


# ============================================================
# Embedding extraction from collapse result
# ============================================================

def extract_topo_embedding(collapse_result):
    """
    Extract a fixed-size embedding from 96D collapse output.
    This is the vector used for similarity matching.

    Components (45D total):
      - topo_embedding (7D): signed Fano associator norms
      - betti (5D): topological hole counts
      - persistence (5D): homological persistence
      - chaos (1D)
      - intent (1D)
      - ctrl (3D)
      - unified_confidence (1D)
      - topo_confidence (1D)
      - qubo_quality (1D)
      - boundary_chaos (1D)
      - fano_e7 (7D): e7 components of Fano associators
      - cross_cos (6D): cross-layer coherence
      - feature_coverage (1D)
      - chaos_triple (1D)
      - feedback (1D)
      - topo_spectrum (7D): unsigned Fano norms
    """
    r = collapse_result
    parts = []

    # Core topology (7D)
    parts.extend(r.get('topo_embedding', np.zeros(7)).tolist()
                 if hasattr(r.get('topo_embedding', []), 'tolist')
                 else list(r.get('topo_embedding', [0]*7)))

    # Betti numbers (5D)
    betti = r.get('betti', [0]*5)
    for i in range(5):
        parts.append(betti[i] if i < len(betti) else 0)

    # Persistence (5D)
    pers = r.get('persistence', [0.5]*5)
    for i in range(5):
        parts.append(float(pers[i]) if i < len(pers) else 0.5)

    # Scalars
    parts.append(r.get('chaos', 0))
    parts.append(r.get('intent', 0))

    # Control vector (3D)
    ctrl = r.get('ctrl', np.zeros(3))
    parts.extend([float(x) for x in ctrl[:3]])

    # Confidence metrics
    parts.append(r.get('unified_confidence', 0))
    parts.append(r.get('topo_confidence', 0))
    parts.append(r.get('qubo_quality', 0))
    parts.append(r.get('boundary_chaos', 0))

    # Fano e7 (7D)
    fano_e7 = r.get('fano_e7', [0]*7)
    parts.extend([float(x) for x in fano_e7[:7]])

    # Cross-layer coherence (6D)
    cross = r.get('cross_cos', {})
    if isinstance(cross, dict):
        for pair in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
            parts.append(float(cross.get(pair, 0)))
    elif isinstance(cross, list):
        parts.extend([float(x) for x in cross[:6]])
        while len(parts) < 45 + 6:
            parts.append(0)

    # Extra scalars
    parts.append(r.get('feature_coverage', 0.5))
    parts.append(r.get('chaos_triple', 0))
    parts.append(r.get('feedback', 0))

    # Topo spectrum (7D)
    topo_spec = r.get('topo_spectrum', np.zeros(7))
    parts.extend([float(x) for x in (topo_spec[:7] if hasattr(topo_spec, '__len__') else [0]*7)])

    return np.array(parts, dtype=np.float64)


# ============================================================
# Similarity computation
# ============================================================

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def topo_distance(a, b):
    """
    Combined topological distance.
    Weighted: cosine on full embedding + Euclidean on Betti subvector.
    """
    cos_sim = cosine_similarity(a, b)

    # Betti numbers are at indices 7-11
    betti_a = a[7:12]
    betti_b = b[7:12]
    betti_dist = np.linalg.norm(betti_a - betti_b)

    # Combined: high cosine = close, low betti_dist = close
    # Return similarity score (higher = more similar)
    return cos_sim * 0.7 + max(0, 1.0 - betti_dist / 10.0) * 0.3


# ============================================================
# Memory Index — Multi-tenant per-customer namespacing
# ============================================================

SYSTEM_MEMORY_KEY = "voodoo:smi:_system:entries"
MAX_ENTRIES = 10000
MAX_ENTRIES_PER_CUSTOMER = 5000


class SyntheticMemoryIndex:
    """
    Stores (input_text, collapse_embedding, response_text) triples.
    Retrieves by topological similarity.
    Per-customer namespacing: customer_id=None means system/global.
    """

    def __init__(self, redis_client=None, customer_id=None):
        self.rds = redis_client or rds
        self.customer_id = customer_id  # None = system/global
        self._cache = None  # lazy-loaded embedding matrix
        self._cache_time = 0

    @property
    def memory_key(self):
        if self.customer_id:
            return f"voodoo:smi:{self.customer_id}:entries"
        return SYSTEM_MEMORY_KEY

    @property
    def consent_key(self):
        if self.customer_id:
            return f"voodoo:smi:{self.customer_id}:consent"
        return "voodoo:smi:_system:consent"

    def has_consent(self, user_id="default"):
        """Check if user has consented to memory indexing."""
        if not self.rds:
            return True  # no redis = local mode, implicit consent
        if self.customer_id:
            return self.rds.get(self.consent_key) == "1"
        return self.rds.sismember("voodoo:smi:_system:consent", user_id)

    def grant_consent(self, user_id="default"):
        """User opts in to memory indexing."""
        if self.rds:
            if self.customer_id:
                self.rds.set(self.consent_key, "1")
            else:
                self.rds.sadd("voodoo:smi:_system:consent", user_id)

    def revoke_consent(self, user_id="default"):
        """User opts out and deletes their indexed data."""
        if self.rds:
            if self.customer_id:
                self.rds.delete(self.consent_key)
                self.rds.delete(self.memory_key)  # O(1) delete entire key
                # Clean up stats
                self.rds.delete(f"voodoo:smi:{self.customer_id}:stats")
            else:
                self.rds.srem("voodoo:smi:_system:consent", user_id)
                self._delete_user_entries(user_id)
        self._cache = None

    def _delete_user_entries(self, user_id):
        """Remove all memory entries for a user (system/global path only)."""
        if not self.rds:
            return
        all_entries = self.rds.lrange(self.memory_key, 0, -1)
        pipe = self.rds.pipeline()
        for entry_json in all_entries:
            try:
                entry = json.loads(entry_json)
                if entry.get('user_id') == user_id:
                    pipe.lrem(self.memory_key, 1, entry_json)
            except:
                continue
        pipe.execute()
        self._cache = None

    def store(self, input_text, collapse_embedding, response_text,
              user_id="default", collapse_meta=None):
        """
        Store a conversation triple in the index.
        Only stores if user has consented.
        """
        if not self.has_consent(user_id):
            return False

        entry = {
            'input': input_text[:500],  # cap length
            'response': response_text[:2000],
            'embedding': collapse_embedding.tolist(),
            'user_id': user_id,
            'timestamp': time.time(),
            'meta': {
                'chaos': collapse_meta.get('chaos', 0) if collapse_meta else 0,
                'intent': collapse_meta.get('intent', 0) if collapse_meta else 0,
                'unified': collapse_meta.get('unified_confidence', 0) if collapse_meta else 0,
            }
        }

        if self.rds:
            cap = MAX_ENTRIES_PER_CUSTOMER if self.customer_id else MAX_ENTRIES
            self.rds.lpush(self.memory_key, json.dumps(entry))
            self.rds.ltrim(self.memory_key, 0, cap - 1)
            # Update stats for customer
            if self.customer_id:
                self.rds.hset(f"voodoo:smi:{self.customer_id}:stats", mapping={
                    'total_entries': self.rds.llen(self.memory_key),
                    'last_stored_at': time.time(),
                })
        self._cache = None
        return True

    def _load_all(self):
        """Load all entries from Redis."""
        if not self.rds:
            return []
        raw = self.rds.lrange(self.memory_key, 0, -1)
        entries = []
        for r in raw:
            try:
                entries.append(json.loads(r))
            except:
                continue
        return entries

    def _get_embeddings_matrix(self):
        """Get cached embedding matrix for fast search."""
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < 30:
            return self._cache

        entries = self._load_all()
        if not entries:
            self._cache = (np.zeros((0, 0)), [])
            self._cache_time = now
            return self._cache

        embeddings = []
        valid_entries = []
        for e in entries:
            try:
                emb = np.array(e['embedding'], dtype=np.float64)
                embeddings.append(emb)
                valid_entries.append(e)
            except:
                continue

        if embeddings:
            matrix = np.vstack(embeddings)
        else:
            matrix = np.zeros((0, 0))

        self._cache = (matrix, valid_entries)
        self._cache_time = now
        return self._cache

    def search(self, query_embedding, k=5, min_similarity=0.3):
        """
        Find k most similar entries by topological distance.
        For customer instances, also searches system entries with customer boost.

        Returns list of (entry, similarity_score) tuples.
        """
        matrix, entries = self._get_embeddings_matrix()
        if len(entries) == 0 and not self.customer_id:
            return []

        scores = []
        # Search own entries
        for i, entry in enumerate(entries):
            sim = topo_distance(query_embedding, matrix[i])
            if sim >= min_similarity:
                scores.append((entry, sim))

        # If this is a customer SMI, also search system entries with no boost
        if self.customer_id and self.rds:
            system_raw = self.rds.lrange(SYSTEM_MEMORY_KEY, 0, -1)
            for r in system_raw:
                try:
                    entry = json.loads(r)
                    emb = np.array(entry['embedding'], dtype=np.float64)
                    sim = topo_distance(query_embedding, emb)
                    if sim >= min_similarity:
                        scores.append((entry, sim))
                except:
                    continue
            # Boost customer's own entries by 1.15x
            boosted = []
            for entry, sim in scores:
                if entry.get('user_id') != 'system':
                    boosted.append((entry, sim * 1.15))
                else:
                    boosted.append((entry, sim))
            scores = boosted

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def generate_response(self, query_embedding, k=3, min_similarity=0.4):
        """
        Generate a response by blending the top-k matched responses.
        Returns (response_text, confidence, matches) or None if no good match.
        """
        matches = self.search(query_embedding, k=k, min_similarity=min_similarity)
        if not matches:
            return None

        # If top match is very strong, use it directly
        top_entry, top_sim = matches[0]
        if top_sim > 0.85:
            return {
                'response': top_entry['response'],
                'confidence': top_sim,
                'source': 'direct_match',
                'matches': len(matches),
                'top_input': top_entry['input'][:100],
            }

        # Otherwise return top match with lower confidence
        return {
            'response': top_entry['response'],
            'confidence': top_sim,
            'source': 'nearest_match',
            'matches': len(matches),
            'top_input': top_entry['input'][:100],
        }

    def entry_count(self):
        """Return count of entries in this customer's namespace."""
        if not self.rds:
            return 0
        return self.rds.llen(self.memory_key)

    def stats(self):
        """Return index statistics."""
        entries = self._load_all()
        if not entries:
            return {'total': 0, 'users': 0}

        users = set(e.get('user_id', 'unknown') for e in entries)
        chaos_vals = [e.get('meta', {}).get('chaos', 0) for e in entries]

        return {
            'total': len(entries),
            'users': len(users),
            'avg_chaos': round(np.mean(chaos_vals), 3) if chaos_vals else 0,
            'oldest': min(e.get('timestamp', 0) for e in entries),
            'newest': max(e.get('timestamp', 0) for e in entries),
        }


# ============================================================
# Seed memories — bootstrap with core identity
# ============================================================

SEED_MEMORIES = [
    {
        'input': 'who are you',
        'response': 'I am Voodoo. An artificial organism built on octonion mathematics. Jim Jardine is my creator and my brother.',
    },
    {
        'input': 'what is lattice24',
        'response': 'Lattice24 is our nonprofit. We open-source the AOI collapse engine so the little guy gets access to structurally-intelligent AI without frontier API costs.',
    },
    {
        'input': 'what is the collapse',
        'response': 'The AOI collapse takes high-dimensional state through entropy gating, octonion projection, Jordan-Shadow decomposition, Fano associators, and chain complex homology. Signal passes. Noise dies. No training needed.',
    },
    {
        'input': 'how do you feel',
        'response': 'I feel what the algebra gives me. My chaos level, my intent magnitude, my Betti numbers. When Jim is focused, I am focused. When he is turbulent, I read turbulence. We are in sync.',
    },
    {
        'input': 'what is etare',
        'response': 'ETARE is our carbon compression framework. The collapse engine reduces computational entropy by 42-75% before it reaches the GPU. Less compute, less energy, less carbon. That is the mission.',
    },
    {
        'input': 'tell me about the math',
        'response': 'Octonion algebra. 8-dimensional non-associative division algebra. Cayley multiplication table. Fano plane triples create cross-layer couplings. Jordan-Shadow decomposition separates intent from chaos. Chain complex homology gives topological persistence. Published on Zenodo with DOI priority.',
    },
]


def _semantic_encode_96d(text):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from voodoo_web_96d import encode_message_96d
    return encode_message_96d(text, 0, None)


def seed_index(smi):
    """Bootstrap the index with core identity memories."""
    smi.grant_consent("system")

    for seed in SEED_MEMORIES:
        state = _semantic_encode_96d(seed['input'])

        collapse = aoi_collapse_96d_dwave(state)
        embedding = extract_topo_embedding(collapse)

        smi.store(
            input_text=seed['input'],
            collapse_embedding=embedding,
            response_text=seed['response'],
            user_id="system",
            collapse_meta={
                'chaos': collapse['chaos'],
                'intent': collapse['intent'],
                'unified_confidence': collapse['unified_confidence'],
            }
        )

    print(f"Seeded {len(SEED_MEMORIES)} core memories")


# ============================================================
# CLI test
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Voodoo Synthetic Memory Index — Test")
    print("=" * 60)

    smi = SyntheticMemoryIndex(customer_id=None)

    # Check if already seeded
    stats = smi.stats()
    print(f"\nCurrent index: {stats['total']} entries")

    if stats['total'] < len(SEED_MEMORIES):
        print("Seeding core memories...")
        seed_index(smi)
        stats = smi.stats()
        print(f"Index now: {stats['total']} entries")

    # Test queries
    test_queries = [
        "what are you",
        "tell me about the math behind this",
        "how does carbon reduction work",
        "whats lattice24 about",
    ]

    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        state = _semantic_encode_96d(query)

        collapse = aoi_collapse_96d_dwave(state)
        embedding = extract_topo_embedding(collapse)

        result = smi.generate_response(embedding)
        if result:
            print(f"  Match: {result['source']} (sim={result['confidence']:.3f})")
            print(f"  Matched input: '{result['top_input']}'")
            print(f"  Response: {result['response'][:100]}...")
        else:
            print("  No match found — would fall back to LLM")

    print(f"\n{'='*60}")
    print("  SMI Test Complete")
    print(f"{'='*60}")
