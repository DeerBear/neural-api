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

---

## 4. Lifecycle and Mode Safety

### 4.1 Premise — the bad state must be unreachable

The spec mandates that training is vanilla softmax with no KAN bookkeeping (§3) and that inference uses the §7 pipeline. Mixing the two — running KAN bookkeeping during a training step, or backpropagating through a KAN-active forward path — is undefined behaviour in the spec and silently corrupts both the model weights and the KAN state.

Rather than asking "what should happen if the modes are mixed", v1's stance is **make the mixed state unreachable through normal API usage**. Training and inference are mutually exclusive at the network level, gated by an explicit one-way transition.

### 4.2 The two-state lifecycle

A `TKANNet` instance is, at any moment, in exactly one of two states:

| State | `InferenceLocked` | What's allowed | What's forbidden |
|---|---|---|---|
| **Training** | `false` | `Compute` (vanilla softmax forward), `Backpropagate`, `Fit*`, weight updates | `LockToInference` is the only way out; `InferenceForward` raises |
| **Inference** | `true` | `Compute` (full §7 pipeline), `InferenceForward` | `Backpropagate`, `Fit*`, weight updates all raise `EKANInInference` |

**Initial state:** Training. A freshly-constructed `TKANNet` is unlocked.

**Transition:** `LockToInference` is the single, **one-way** transition to inference. There is no `UnlockToTraining`. The network commits to inference at the moment the operator calls this method, and every subsequent call respects that commitment.

The one-way property mirrors the per-head lifecycle's `SoftmaxActive → KANActive` transition (spec §4, LC-1) and the squaring monotonicity (spec HS-5). Same epistemic shape applied at a different scale: once the system has committed to a more-evolved state, it does not revert.

### 4.3 `LockToInference` — what it does

```pascal
procedure TKANNet.LockToInference;
begin
  if FInferenceLocked then exit;             // idempotent
  FInferenceLocked := true;
  // Propagate to every registered KAN layer.
  for layer in FKANLayers do
    TNNetKANNormaliser(layer).EnterInferenceMode;
end;
```

`EnterInferenceMode` on the layer:

- Sets `FInferenceMode := true`.
- If the layer is at cold-start (no prior KAN state, e.g. fresh retrofit), runs `ColdStartHead` for every head per spec §8.4.
- Validates that `FRNG.State` is non-zero (a sanity check against forgotten seeding).

Once propagation completes, the network is fully in inference mode. Subsequent forward passes run the §7 pipeline; subsequent training calls raise.

### 4.4 Training-method gating

Every method on `TKANNet` that performs a weight update or invokes a backward pass calls `AssertNotLocked` as its first action:

```pascal
procedure TKANNet.AssertNotLocked(const opName: string);
begin
  if FInferenceLocked then
    raise EKANInInference.CreateFmt(
      '%s is not permitted: network is locked to inference. ' +
      'Construct a new TKANNet from a checkpoint to resume training.',
      [opName]);
end;
```

Methods that gate on this:

- `TKANNet.Backpropagate` (override of `TNNet.Backpropagate`).
- Any `Fit*` / `Train*` entry point inherited from `TNNet` that the subclass exposes (overridden to gate before delegating).
- Any direct weight-mutation entry point.

The error message is intentionally specific — it tells the operator both what failed and how to recover ("construct a new `TKANNet` from a checkpoint"). Lost training time is the most expensive operator mistake; the message catches it loudly.

### 4.5 Inference-method gating

Inference-only methods (currently just `InferenceForward`) assert the opposite:

```pascal
procedure TKANNet.InferenceForward(pInput: TNNetVolume);
begin
  if not FInferenceLocked then
    raise EKANNotLocked.Create(
      'InferenceForward requires LockToInference to have been called first.');
  inherited Compute(pInput);
end;
```

This catches the inverse mistake — calling the inference path while still in training mode would skip the §7 pipeline silently because the per-layer `FInferenceMode` flags would be `false`. Operators get a clear error instead of a quietly broken serving deployment.

### 4.6 Per-layer enable: `KANEnabled`

Distinct from the network-level lock, each `TNNetKANNormaliser` carries a per-layer `FKANEnabled: boolean` (default `true`). This is the spec's `disable_kan_at_inference` flag (§11.8), repackaged with positive polarity for natural API readability.

Behaviour table (locked = `true`):

| `Layer.KANEnabled` | Layer `Compute` behaviour |
|---|---|
| `true` (default) | Full §7 pipeline runs. KAN may take over after Phase M convergence. |
| `false` | Pure softmax forward; no spline evaluation, no NLMS, no mech#1, no gauge. **Zero KAN overhead** on this layer. |

Bulk operations on `TKANNet`:

```pascal
procedure TKANNet.DisableKANForLayer(idx: integer);
procedure TKANNet.EnableKANForLayer(idx: integer);
procedure TKANNet.DisableAllKAN;
procedure TKANNet.EnableAllKAN;
function  TKANNet.KANEnabledMask: TArray<boolean>;
```

The intended deployment workflow (selective per-layer enable) is laid out in §1: train, retrofit with KAN on every layer, evaluate per-layer, disable KAN on layers where it doesn't pay, ship the subset. `KANEnabled = false` is genuinely free overhead-wise, so leaving every layer KAN-capable by construction and toggling at runtime costs nothing on disabled layers.

### 4.7 Layer-level backstop

`TNNetKANNormaliser.Backpropagate` is the last line of defence. Even if some code path bypasses `TKANNet`'s gating (e.g. a test directly invoking a layer method, or a caller that constructed a normaliser without registering it):

```pascal
procedure TNNetKANNormaliser.Backpropagate;
begin
  if FInferenceMode then
    raise EKANInInference.Create(
      'Backpropagate called on KAN normaliser layer in inference mode.');
  inherited Backpropagate;   // standard softmax backprop path
end;
```

In training mode the layer falls through to the standard softmax backprop (since forward was vanilla softmax, the gradient computation is unchanged). In inference mode it raises immediately.

### 4.8 Save/load preserves the lock

Persistence rules:

- A network checkpointed while locked **reloads locked**. `FInferenceLocked` is part of `TKANNet`'s persisted state.
- A network checkpointed while unlocked reloads unlocked. The operator can then either continue training or call `LockToInference` to enter serving.
- A retrofit-source checkpoint (a vanilla softmax `TNNet` checkpoint loaded into a `TKANNet`) loads **unlocked** with cold-start KAN state per spec §8.4. The operator decides whether to lock immediately or warm up Phase M against representative data first.

The lock state is a single boolean field; no special handling beyond inclusion in the standard persistence stream.

### 4.9 Exception types

```pascal
type
  EKANInInference = class(Exception);   // raised when training-only ops attempted while locked
  EKANNotLocked   = class(Exception);   // raised when inference-only ops attempted while unlocked
  EKANBadState    = class(Exception);   // raised on impossible internal states (defensive)
```

All three derive from the standard Pascal `Exception`. They are caught by the standard `try/except` block and surface to the operator with diagnostic messages.

### 4.10 Workflow summary

```pascal
// --- Training ---
NN := TKANNet.Create;
// build network with AddKANSelfAttention(...)
NN.Fit(...);                       // training, FInferenceLocked = false
NN.SaveToFile('checkpoint.dat');   // saved unlocked

// --- Serving ---
NN := TKANNet.LoadFromFile('checkpoint.dat');
NN.LockToInference;                // one-way commit
NN.InferenceForward(input);        // serving, full §7 pipeline runs
// ... indefinitely ...
NN.SaveToFile('post-deployment.dat');  // saved locked, with evolved KAN state

// --- Resume serving ---
NN := TKANNet.LoadFromFile('post-deployment.dat');  // reloads locked
NN.InferenceForward(input);        // resumes immediately, no relock needed
```

The bad state (mixing training and inference) is unreachable through this API. The only way to attempt it is to bypass `TKANNet` entirely and directly invoke layer methods — and the layer-level backstop catches that.

---

## 5. `TNNetKANNormaliser` Layer Class

The single new layer class. Implements the §7 pipeline. Lives in `neural/neuralkanattention.pas`.

### 5.1 Inheritance and rationale

```pascal
TNNetKANNormaliser = class(TNNetIdentity)
```

Inherits from `TNNetIdentity`, mirroring the convention used by recently-added activation layers (`TNNetGELU`, `TNNetMish`). `TNNetIdentity` provides standard pass-through semantics for output shape and gradient routing; the KAN normaliser overrides `Compute` and `Backpropagate` to inject its own forward and backward behaviour while reusing identity's plumbing for everything else.

The layer is *not* a multi-head container. It is a per-row normaliser instance that sits in one head's sub-path between `TNNetSplitChannels` (upstream) and `TNNetDeepConcat` (downstream). Per-head state is held inside the layer because the layer only ever services one head at a time. `H_max` parallel `TNNetKANNormaliser` instances are constructed by `TKANNet.AddKANSelfAttention`, each with `HeadIndex` set to its position in the layer's KAN registry.

Wait — this needs a clarification. Re-reading: per-layer head state was meant to mean *all heads' state lives on a single normaliser instance*. Re-aligning with the multi-head architecture decision in this document:

**Option taken:** one `TNNetKANNormaliser` per head sub-path. Each instance holds **one head's worth** of B-spline state (`Coeffs`, `Status`, `KLEMA`, etc.) — a single `TKANHeadState`, not an array. `H_max` instances are constructed at network-build time, registered in `FKANLayers` in head-index order. Per-layer aggregates (`ClipRateEMA`, `ConsecutiveHighSweeps`, squaring decisions) are coordinated at the `TKANNet` level by walking the registry and grouping by attention layer.

This matches neural-api's existing channel-split pattern (one sublayer per head) and keeps each layer instance simple. Head squaring becomes a `TKANNet`-level operation that flips `FKANEnabled` on previously-inactive head sub-paths, since pre-allocation already created their sublayers at `H_max` count.

### 5.2 Class declaration

```pascal
TNNetKANNormaliser = class(TNNetIdentity)
private
  // --- Identity (set at construction) ---
  FAttentionLayerId: integer;        // groups normalisers belonging to one attention layer
  FHeadIndex: integer;               // 0 .. H_max - 1 within the attention layer
  FBasis: TKANBasis;                 // shared with the layer's other heads (managed by TKANNet)

  // --- Per-head mutable state ---
  FHead: TKANHeadState;              // ψ-coeffs, status, KL EMA, counters

  // --- Per-head per-layer-step counters ---
  FCascadeCapHits: integer;          // diagnostic (spec §5.2.6)

  // --- RNG (one per layer; this normaliser inherits a reference) ---
  FRNG: ^TKANSeededRNG;              // pointer; TKANNet owns the storage

  // --- Configuration (cached at construction; immutable thereafter) ---
  FGridSpec: TKANGridSpec;
  FHighClipShare, FLowLiftShare: TKANShareRule;
  FForceMode: TKANForceMode;
  FSharpenAlpha, FEpsilonKL, FLambdaKL: TNeuralFloat;
  FNConfirm, FCascadeMaxIter: integer;
  FTauCalm, FTauStressed: TNeuralFloat;
  FNlmsEta, FNlmsDelta: TNeuralFloat;

  // --- Operational flags ---
  FInferenceMode: boolean;           // set by TKANNet.LockToInference
  FKANEnabled: boolean;              // per-layer toggle (spec §11.8)
  FFreezeAfterTakeover: boolean;
  FSkipRedundantSoftmax: boolean;

  // --- Per-pass scratch (alloc once in constructor, reuse) ---
  FPhiRow: array of TNeuralFloat;    // φ values for the current row
  FWSoftmaxRow: array of TNeuralFloat;
  FWKANRow: array of TNeuralFloat;
  FWSharpRow: array of TNeuralFloat; // Phase D snapshot (spec §5.5.3)
  FTargetPreRow: array of TNeuralFloat;

  // --- §7 pipeline helpers (private) ---
  procedure EvaluateSplineRow(const scoresRow: TNeuralVolume; rowIdx, rowLen: integer);
  procedure ComputeWeightsRow(rowLen: integer;
                              const scoresRow: TNeuralVolume; rowIdx: integer);
  function  ComputeKLRow(rowLen: integer): TNeuralFloat;
  procedure NLMSPhaseM(rowLen: integer;
                       const scoresRow: TNeuralVolume; rowIdx: integer);
  procedure NLMSPhaseD(rowLen: integer;
                       const scoresRow: TNeuralVolume; rowIdx: integer);
  procedure Mechanism1Sweep;
  procedure GaugeRenormalise;
  procedure CheckHandover;
  procedure ColdStartHead;
  procedure ZeroOutputForInactive;

public
  constructor Create(const GridSpec: TKANGridSpec;
                     AttentionLayerId, HeadIndex: integer;
                     SharedBasis: TKANBasis;
                     SharedRNG: PKANSeededRNG); reintroduce;
  destructor Destroy; override;

  procedure Compute; override;
  procedure Backpropagate; override;

  function  SaveDataToString: string; override;
  procedure LoadDataFromString(strData: string); override;
  function  SaveStructureToString: string; override;

  // Mode control (called by TKANNet)
  procedure EnterInferenceMode;

  // Telemetry (spec §14.5)
  property HeadIndex: integer read FHeadIndex;
  property AttentionLayerId: integer read FAttentionLayerId;
  property Status: TKANStatus read FHead.Status;
  property KLEMA: TNeuralFloat read FHead.KLEMA;
  property ClipCountTotal: integer read FHead.ClipCountTotal;
  property CascadeCapHits: integer read FCascadeCapHits;

  // Per-layer toggles
  property KANEnabled: boolean read FKANEnabled write FKANEnabled;
  property FreezeAfterTakeover: boolean read FFreezeAfterTakeover write FFreezeAfterTakeover;
end;
```

### 5.3 `Compute` — pipeline mapping

`Compute` is the §7 pipeline. Pseudocode showing the structure; actual implementation will use `TNeuralVolume` accessors and pointer arithmetic for hot paths.

```
procedure TNNetKANNormaliser.Compute;
begin
  // Training mode: pure softmax, no bookkeeping.
  if not FInferenceMode then begin
    SoftmaxComputeAsParent;     // call inherited softmax-equivalent forward
    exit;
  end;

  // Per-layer disable: also pure softmax, zero KAN overhead.
  if not FKANEnabled then begin
    SoftmaxComputeAsParent;
    exit;
  end;

  FCascadeCapHits := 0;          // reset per Compute (or per-batch — TBD)

  for rowIdx in 0 .. FOutput.Depth - 1 do begin
    // §7 step 1: scores already computed upstream and presented in input.
    // §7 step 2: evaluate spline at every score in this row.
    EvaluateSplineRow(FPrevLayer.Output, rowIdx, rowLen);

    // §7 step 3: compute candidate weights.
    //   Skip w_softmax if KANActive and FSkipRedundantSoftmax (spec §11.8).
    ComputeWeightsRow(rowLen, FPrevLayer.Output, rowIdx);

    // §7 steps 4-5: select forward weights based on status; write to FOutput.
    case FHead.Status of
      ksSoftmaxActive: WriteRowToOutput(rowIdx, FWSoftmaxRow);
      ksKANActive:     WriteRowToOutput(rowIdx, FWKANRow);
    end;

    // §7 step 6: KL update (always computed for diagnostics, even post-takeover).
    UpdateKLEMA(ComputeKLRow(rowLen));

    // §7 step 7: NLMS local rule.
    if not (FFreezeAfterTakeover and (FHead.Status = ksKANActive)) then begin
      case FHead.Status of
        ksSoftmaxActive: NLMSPhaseM(rowLen, FPrevLayer.Output, rowIdx);
        ksKANActive:     NLMSPhaseD(rowLen, FPrevLayer.Output, rowIdx);
      end;
    end;

    // §7 step 8: mechanism #1 cascade.
    Mechanism1Sweep;

    // §7 step 9: GM=1 gauge renormalisation.
    GaugeRenormalise;
  end;

  // §7 step 10: handover check (per-head).
  if FHead.Status = ksSoftmaxActive then CheckHandover;

  // §7 step 11 (squaring) is per-attention-layer, not per-normaliser.
  // It is invoked by TKANNet after all normalisers in the same attention
  // layer have completed Compute for this batch. See §6.

  // HS-3: zero output if this head is currently inactive.
  if not IsActiveHead then ZeroOutputForInactive;
end;
```

Per-row processing matches the row-coupled Phase D rule's snapshot semantics (spec §5.5.3 fix). The row sum is computed inside `EvaluateSplineRow` and reused by `NLMSPhaseD`'s target derivation.

### 5.4 `Backpropagate`

```pascal
procedure TNNetKANNormaliser.Backpropagate;
begin
  if FInferenceMode then
    raise EKANInInference.Create(
      'Backpropagate called on KAN normaliser layer in inference mode.');

  // Training mode: forward was pure softmax, backward is pure softmax.
  inherited Backpropagate;
end;
```

The layer's training-mode backward is the inherited softmax backprop. No KAN state is touched during training.

### 5.5 Helper method behaviour summary

Each helper realises a single section of the spec; bodies are not detailed here (they are direct translations of the spec algorithms):

| Method | Spec section | One-line behaviour |
|---|---|---|
| `EvaluateSplineRow` | §5.1, §5.5.3 | Compute `ψ(s)` then `φ = ψ²` for every score in the row; cache in `FPhiRow`. Compute and cache the row sum for Phase D's snapshot. |
| `ComputeWeightsRow` | §7 step 3 | Row-normalise `FPhiRow` into `FWKANRow`; compute `FWSoftmaxRow` from raw scores (skip if `FSkipRedundantSoftmax` and `KANActive`). |
| `ComputeKLRow` | §6.1 | KL of the softmax row from the KAN row, in nats. Reverse-direction `KL(softmax ‖ KAN)`. |
| `NLMSPhaseM` | §5.5.2 | Per-score NLMS toward `exp(s)` over the `k+1` active basis functions, in ψ-space (since coeffs are ψ). |
| `NLMSPhaseD` | §5.5.3 | Per-score NLMS toward the row-snapshot sharpened target; `α`-power renormalisation per spec. |
| `Mechanism1Sweep` | §5.2 | Window GM, high-clip + low-lift cascade with Auto / forced share rule selection, increments `FCascadeCapHits` if cap reached. |
| `GaugeRenormalise` | §5.3 | Log-space mean computation; uniform rescale by `1/G`. |
| `CheckHandover` | §6.3 | If `KLEMA < ε_KL` for ≥ `N_confirm` passes, atomically flip `Status := ksKANActive`. |
| `ColdStartHead` | §8.4 | Fit `ψ ≈ exp(s/2)`; reset all per-head counters; mark `Status := ksSoftmaxActive`. |
| `ZeroOutputForInactive` | HS-3 | Write zeros to `FOutput` if this head's index ≥ the attention layer's `ActiveHeads`. |

### 5.6 Per-pass scratch buffers

Allocated once in the constructor (sized by `FOutput.Depth` and `KnotCount`); reused across rows and across calls to `Compute`. Total per-instance scratch: `~5 × max_row_length × sizeof(Single)` for the row buffers, plus `KnotCount × sizeof(Single)` for the spline scratch. For typical `L = 256`, `N = 64`: ~6 KB per normaliser instance. With `H_max = 64` heads per layer and 12 layers: ~5 MB total. Negligible relative to model weights.

### 5.7 Construction parameters

`TKANNet.AddKANSelfAttention` constructs `H_max` instances of `TNNetKANNormaliser`, passing each:

- The shared `TKANGridSpec` for the attention layer.
- A unique `(AttentionLayerId, HeadIndex)` pair so per-layer aggregation can find them later.
- A pointer to the shared `TKANBasis` (one per attention layer).
- A pointer to the shared `TKANSeededRNG` (one per attention layer; head-squaring uses it deterministically).
- Cached configuration values (sharpening α, KL threshold, etc.) — passed by value because they are immutable after construction.

The shared `TKANBasis` and `TKANSeededRNG` are owned by the attention layer's metadata held in `TKANNet`, not by any individual normaliser; the normalisers hold typed pointers and do not free them.
