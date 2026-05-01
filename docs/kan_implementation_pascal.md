# KAN Attention Normaliser — Pascal Implementation Decisions (v1)

**Companion to:** `docs/kan_attention_spec.md`
**Target library:** neural-api (CAI Pascal/Delphi)
**Scope:** v1, single-threaded inference, retrofit-only

---

## 1. Scope and Overview

This document records the implementation-level decisions made for porting the KAN attention normaliser spec onto the neural-api Pascal library. It is a **decision log**, not a re-statement of the spec. Where this document and `kan_attention_spec.md` disagree, the spec is authoritative for the mechanism and this document is authoritative for the Pascal-specific realisation.

**What this document covers:**

- Where KAN code lives in the neural-api source tree.
- Which existing types are reused, extended, or replaced.
- The new types and classes introduced.
- The pipeline mapping from §7 of the spec to a Pascal `Compute` method.
- Threading, save/load, lifecycle gating, and test plan.

**What this document does not cover:**

- The spec mechanisms themselves. Those are fully specified in `kan_attention_spec.md` §5–§9.
- Implementation work — this is pre-implementation. No code has been written.
- Empirical results. v1 is unimplemented; benchmarks come later.

**Out of scope for v1 (deferred per §13.4 of the spec, see also §10 below):**

- KAN-ifying Q/K/V/output projections (v2).
- Training-time KAN state (no planned version).
- Multithreaded inference shadow-copy strategy (v2 if needed).
- GPU port (v2+).
- Cross-layer coupling (v3, speculative).

---

## 2. Codebase Touch Points

Investigation of `/home/user/neural-api` (commit `2ff97c5` and earlier) shows the existing self-attention pipeline lives in `neural/neuralnetwork.pas`. The KAN integration touches only the surfaces listed below. Everything not listed is reused unchanged.

### 2.1 Files

| File | Status | Purpose |
|---|---|---|
| `neural/neuralkanattention.pas` | **new** | All KAN-specific types, helpers, and the `TNNetKANNormaliser` layer class. |
| `neural/neuralnetwork.pas` | **modified** | Add `TKANNet` subclass declaration; add factory entry for `TNNetKANNormaliser` at the existing factory site (~ line 12701). |
| `tests/TestNeuralKAN.pas` | **new** | Tier-1 invariant tests and Tier-2 functional tests (§12 of the spec). |
| `docs/kan_attention_spec.md` | unchanged | Mechanism spec. |
| `docs/kan_implementation_pascal.md` | this document | Implementation decision log. |

### 2.2 Existing components reused unchanged

| Component | Source | Reason |
|---|---|---|
| `TNNetLayer` base class | `neuralnetwork.pas:148` | Standard layer contract; `TNNetKANNormaliser` inherits from `TNNetIdentity` which inherits from this. |
| `TNNetIdentity` | `neuralnetwork.pas` | Convenient parent for layers that pass through structure but transform values; mirrors how new activation layers (GELU, Mish) are implemented. |
| `TNNetPointwiseConvLinear` (Q, K, V projections) | `neuralnetwork.pas` | Reused as-is in `AddKANSelfAttention`'s sub-chain. The KAN spec does not modify projections in v1. |
| `TNNetSplitChannels`, `TNNetDeepConcat` | `neuralnetwork.pas` | Reused for multi-head split/join. The KAN spec's "pre-allocate H_max, mask inactive heads" maps cleanly onto these. |
| Score scaling `Q·Kᵀ/√d` plumbing | `neuralnetwork.pas:7534-7566` | Reused; the spec does not modify score computation. |
| `TNeuralVolume` | `neuralvolume.pas` | The standard tensor type; KAN coefficients and scratch buffers will use compatible storage. |
| `TNeuralFloat = Single` | `neuralvolume.pas:58` | 32-bit float; FP tolerances in the spec sized accordingly (see §6 of this document). |
| Forward / backward training paths | `neuralnetwork.pas`, `neuralfit.pas` | Reused. KAN training is vanilla softmax (spec §3); no training path changes. |
| `SaveDataToString` / `LoadDataFromString` infrastructure | `neuralnetwork.pas:285-286` | Reused for KAN state persistence; KAN state is appended to the layer's existing serialised string. |

### 2.3 Existing components replaced

| Component | Replaced by | Where |
|---|---|---|
| `TNNetPointwiseSoftMax` (in `AddSelfAttention`) | `TNNetKANNormaliser` | At the row-normalisation step inside the attention chain (`neuralnetwork.pas:7506`, `:7566`). Replacement happens only inside the new `TKANNet.AddKANSelfAttention` builder. The standalone `TNNet.AddSelfAttention` is unchanged. |

### 2.4 New components added

| Component | Purpose | Section |
|---|---|---|
| `TKANNet` (subclass of `TNNet`) | Network class with KAN layer registry, mode lock, and bulk operations. | §5 |
| `TKANNet.AddKANSelfAttention` | Network builder that constructs the attention chain with `TNNetKANNormaliser` substituted in. | §5 |
| `TNNetKANNormaliser` (subclass of `TNNetIdentity`) | The drop-in normaliser layer; holds per-head spline state, runs the §7 pipeline. | §4 |
| `TKANGridSpec`, `TKANHeadState`, `TKANStatus`, `TKANShareRule`, `TKANForceMode` | Configuration and per-head state types. | §3 |
| `TKANBasis` | Immutable B-spline basis with precomputed knot positions, shared across all heads in a layer. | §3 |
| `TKANSeededRNG` | Per-layer SplitMix64 RNG for deterministic head-split perturbations and any other randomness. Required by spec §9 because neural-api otherwise uses global `Random()`. | §3 |
| `EKANInInference`, `EKANNotLocked` | Exception classes for mode-safety violations. | §8 |

### 2.5 Surfaces explicitly not touched

- `TNNet.AddSelfAttention`. The vanilla softmax builder is unchanged. Users who want softmax keep using it; users who want KAN call `TKANNet.AddKANSelfAttention`. Coexistence is automatic.
- `TNNet.Compute`. The base network forward pass is unchanged; `TKANNet` does not override it. KAN-specific behaviour lives entirely inside `TNNetKANNormaliser.Compute`.
- `neuralfit.pas` training loop. KAN bookkeeping is gated on `InferenceMode = true` and is bypassed entirely during training (spec §3).
- `TNeuralVolume`, `TNNetNeuronList`, `TNeuronList`. No changes.
- The `TNeuralThread` / `TNeuralThreadList` infrastructure (`neuralthread.pas`). The KAN layer assumes single-threaded access during inference per §6 of this document.

---

## 3. New Types and Helper Classes

All declarations below live in the new unit `neural/neuralkanattention.pas`. Field naming follows neural-api convention (`F`-prefixed private fields, properties for read access).

### 3.1 Configuration enums

```pascal
type
  // Per-head lifecycle phase (spec §4).
  TKANStatus = (ksSoftmaxActive, ksKANActive);

  // Mechanism #1 share rules (spec §5.2.5).
  TKANShareRule = (ksrProportional, ksrInverseProportional, ksrAuto);

  // Mode override for ablation / debugging (spec §11.8).
  TKANForceMode = (kfmAuto, kfmA, kfmB, kfmC);
```

`TKANStatus` is the canonical per-head phase; transitions are one-way `ksSoftmaxActive → ksKANActive` (spec §4, LC-1).

`TKANShareRule` and `TKANForceMode` carry the §5.2.5 mode taxonomy. Mode D is structurally excluded by the Auto logic; `kfmA`/`kfmB`/`kfmC` are available via `force_mode` for ablation (spec §11.8).

### 3.2 Grid specification

```pascal
type
  TKANGridSpec = record
    GridLow, GridHigh: TNeuralFloat;   // grid range, default [-8, +8] nats
    KnotCount: integer;                // N, default 64
    BasisOrder: integer;               // k, default 3 (cubic)
    function Hash: UInt64;             // for deterministic per-layer seeding
  end;
```

Immutable for the lifetime of a layer (spec §5.1, "Grid is immutable for the lifetime of a layer"). `Hash` is used by `TKANNet` to derive a stable per-layer RNG seed at construction time (spec §8.4).

### 3.3 Per-head state

```pascal
type
  TKANHeadState = record
    Coeffs: array of TNeuralFloat;     // ψ-space, length = TKANGridSpec.KnotCount
    Status: TKANStatus;
    KLEMA: TNeuralFloat;
    ConsecutiveLowPasses: integer;
    ClipCountTotal: integer;
  end;
```

One instance per head; `H_max` of these per layer. `Coeffs` are stored as ψ-coefficients per the spec §5.1 Q5 decision: `φ(s) = ψ(s)²` is computed on every evaluation, ensuring strict non-negativity by construction. The cold-start initialisation fits `ψ ≈ exp(s/2)` so that `φ ≈ exp(s)` over the grid range (spec §8.4).

### 3.4 B-spline basis

```pascal
type
  TKANBasis = class
  private
    FGridSpec: TKANGridSpec;
    FKnots: array of TNeuralFloat;
    // Optional: precomputed basis tables on a fine eval grid
    // for faster Evaluate() if profiling shows it's a hotspot.
  public
    constructor Create(const Spec: TKANGridSpec);
    destructor Destroy; override;

    // Evaluates the k+1 non-zero basis functions at score s.
    // Returns the index of the first active basis (FirstIdx)
    // and writes k+1 basis values into Vals.
    procedure Evaluate(s: TNeuralFloat;
                       out FirstIdx: integer;
                       Vals: PSingleArray);

    // Returns the squared-basis-sum Σ B_j(s)² used as the NLMS
    // denominator (spec §5.5.2). Single pass; O(k+1).
    function BasisSquaredSum(s: TNeuralFloat): TNeuralFloat;

    // Cold-start fit: returns ψ-coefficients such that
    // ψ(s)² ≈ exp(s) over the grid range. Least-squares on a
    // dense (s, exp(s/2)) grid (spec §5.1, §8.4).
    function FitPsiToExp: TArray<TNeuralFloat>;

    property GridSpec: TKANGridSpec read FGridSpec;
  end;
```

One instance per layer, shared across all `H_max` heads (since the grid is identical for all heads in a layer). Immutable after construction. The fine-grid precomputation is optional for v1; the simple `Evaluate` direct evaluation is fine to start with — profiling can drive optimisation if needed.

### 3.5 Seeded RNG

```pascal
type
  TKANSeededRNG = record
    State: UInt64;
    procedure Seed(s: UInt64);
    function NextU64: UInt64;             // SplitMix64 step
    function NextFloat: TNeuralFloat;     // uniform [0, 1)
    function NextNormal: TNeuralFloat;    // standard normal via Box-Muller
  end;
```

SplitMix64 is the chosen algorithm: small, fast, deterministic, single-`UInt64` state (trivial to checkpoint). `NextNormal` produces standard-normal samples for the head-squaring log-normal perturbation (spec §5.4.4).

Required because neural-api's existing randomness uses global `Random()` (`neuralnetwork.pas:5249-5250`), which does not satisfy the determinism guarantees in spec §9. Each `TNNetKANNormaliser` instance carries its own `TKANSeededRNG`, seeded by `TKANNet` at registration time from `(layer_index_in_FKANLayers, grid_spec.Hash)` (spec §8.4 retrofit-source initialisation).

`State` is saved as part of the layer's persistent state so reload reproducibility is exact (spec §8.5).
