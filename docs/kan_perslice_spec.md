# KAN Per-Slice Augmentation — Design Specification (v0 draft)

> **Status: v0 working draft.** Sections marked **[OPEN]** still need design
> decisions; defaults proposed below are reasonable starting points to be
> sharpened against implementation reality. Closed sections reflect explicit
> decisions made during the design conversation.

## 1. Abstract

This specification extends the KAN-attention design (`kan_attention_spec.md`)
to the rest of the transformer. A transformer's non-attention weights compute
a sequence of fixed scalar projections; once trained, these become static. We
augment that static computation with a **per-slice KAN corrector** — a small,
mathematically-bounded, online-adapting B-spline-parameterised module that
sits *in parallel* with each slice of weights and adds a learned correction
to the slice's output.

Each KAN corrector starts at zero contribution (bit-identical fallback),
evolves during training by observation only (no backprop through splines),
graduates to influencing the forward pass once a convergence criterion fires,
and then **continues to evolve at inference time** to track the deployed data
distribution. All four safety mechanisms from the attention design — GM=1
gauge fixing, energy-spreading cascade, edge-squaring capacity growth,
phased graduation — apply directly with no re-derivation, because the KAN's
contribution is added linearly to the slice output and therefore inherits
every coefficient bound verbatim.

The architectural target: a transformer whose deployed function is the
trained base model plus a fleet of small, continuously-adapting per-slice
correctors. Training cost is essentially unchanged; deployed parameter
overhead is a small fraction per slice; deployed behaviour adapts to actual
data without retraining.

## 2. Design Goals

Inherited from the attention spec:

- **Bit-identical fallback.** A non-graduated KAN contributes exactly zero;
  the augmented forward pass is mathematically equal to the unaugmented one.
- **No backprop through splines.** K's coefficients are updated by NLMS
  observation only. Backprop through the slice itself is unchanged.
- **Online learnable.** K evolves on every forward pass during training,
  and continues to evolve on every forward pass at inference.
- **Structural safety, not engineered safety.** Every bound on K's behaviour
  comes from a mathematical invariant (GM gauge, M1-1 product preservation,
  HMax cap), not from an ε floor or clamp.
- **Deterministic.** Per-slice KAN state is seeded deterministically from
  the slice's identity hash (analogous to attention's per-layer seeding).

New to per-slice:

- **Slicing rule independence.** The slice boundaries (§3) are an
  implementation choice; the rest of the spec must not depend on slice
  internals.
- **Inference-time stability.** The continued evolution at inference must
  not destabilise an already-deployed model. The existing GM=1 + spreading
  bounds carry that load; no new mechanism required.
- **Compositionality with attention KAN.** Slices that overlap an attention
  layer's normaliser are independent: the attention KAN handles the
  per-head softmax replacement; the per-slice KAN handles the weight
  augmentation. They observe different signals and update independently.

## 3. Slicing Rule

The transformer's non-attention parameters are sliced uniformly by parameter
count. Each contiguous block of approximately `SliceSize` parameters is one
slice and gets one KAN corrector.

- **Default `SliceSize`:** 250,000 parameters.
- **Boundary policy:** slices respect *layer* boundaries (a slice does not
  cross between two distinct neural-network layers). Within a layer, the
  slice boundaries can fall anywhere; the KAN's input/output topology
  adapts to whatever the slice's I/O is (§4).
- **Number of slices** at deployment is `total_non_attention_params /
  SliceSize`, rounded to respect layer boundaries.

Worked example for a 1M-parameter non-attention transformer: 4 slices, each
~250K, each with one KAN corrector.

`SliceSize` is a deployment-time hyperparameter:
- Smaller `SliceSize` → more KANs, finer adaptive resolution, more per-KAN
  overhead.
- Larger `SliceSize` → fewer KANs, coarser correction, less per-KAN
  overhead.

## 4. Per-Slice K Architecture **[OPEN — proposed default]**

Each per-slice KAN is structured as a **low-rank B-spline corrector**:

```
   x ∈ R^D_in  ─[linear A]→  z ∈ R^r  ─[KAN edges]→  z' ∈ R^r  ─[linear B]→  K(x) ∈ R^D_out
```

- **Down-projection A** (`D_in × r`): linear, learnable via NLMS observation.
- **KAN edge layer** (`r → r`): r independent B-spline edges, each a 1-D
  spline parameterised by `KnotCount` coefficients in nat-domain (per
  attention spec §5.1). Edge `i` consumes `z[i]` and produces `z'[i]`.
- **Up-projection B** (`r × D_out`): linear, learnable via NLMS observation.

**Capacity:** `r` is the KAN's "rank" parameter, mirroring the role of
`Heads` in attention. It starts at `InitialRank = 2` and grows under
edge-squaring pressure (§8) up to a cap `MaxRank`.

**Proposed defaults** (drawn to mirror attention spec proportions):
- `InitialRank = 2`
- `MaxRank = 32`
- `KnotCount = 64`
- `BasisOrder = 3` (cubic)
- Initial parameter cost per slice (`D_in = D_out = 512`):
  `2 × KnotCount + 2 × 512 × 2 = 128 + 2048 = ~2.2K`, about 0.9% of slice.
- Steady-state at full `MaxRank`: `32 × 64 + 2 × 512 × 32 = 2048 + 32768 =
  ~35K`, about 14% of slice.

This gives the corrector a small starting footprint and bounded growth
under demand. Open question: whether `A` and `B` should be linear matrices
trained via NLMS, or themselves KAN-parameterised. The simpler linear
choice composes cleanly with the spline gauge story; the KAN-parameterised
choice would double the spline learning surface. v0 picks linear.

## 5. Influence Mechanism

Closed.

The per-slice KAN contributes additively to the slice's output:

```
   slice_out := S(x) + K(x)
```

- `S(x)`: the slice's normal forward computation (linear projection plus
  whatever activation is part of that slice).
- `K(x)`: the KAN corrector's output, computed via §4.

`K` starts at zero (all spline coefficients cold-started to produce zero
output) and grows under graduation. Before graduation, `K(x) ≡ 0` exactly,
yielding bit-identical fallback by construction.

## 6. Observation and NLMS Update **[OPEN]**

Each forward pass:

1. Slice computes `S(x)`. K computes `K(x)`. Output `S(x) + K(x)` proceeds.
2. The pair `(x, S(x))` is observed by K's NLMS rule.
3. K's spline coefficients update toward minimising the equivalent of "the
   correction K *should have* applied for this input to make the model's
   downstream behaviour better."

**Open: the observation target.** Two candidate signals:
- *Output-only*: target derived from `(x, S(x))` and possibly downstream
  layer outputs. No backprop coupling.
- *Gradient-informed*: the gradient `∂L/∂slice_out` (already computed by
  backprop in training) is used as a target signal for K — "the model
  wished slice_out had been pushed in this direction." K then learns to
  push it in that direction proactively. This is *gradient-as-target*,
  not *gradient-as-update*: backprop still doesn't flow into K.

Gradient-informed is strictly more informative but couples K's learning to
backprop's signal in a non-trivial way. v0 proposes starting with
output-only and adding gradient-informed in a v1 if needed.

## 7. Graduation Criterion **[OPEN]**

Mirroring the attention spec's KL-EMA mechanism. Each slice's K is
considered to have *graduated* once its predicted correction has been
stable and useful for `N_confirm_slice` consecutive observation passes.

Candidate metrics for "stable and useful":
- **Reconstruction error**: how well K's output, applied additively,
  reduces some downstream loss proxy compared to the unaugmented slice.
- **Self-consistency**: K's outputs on similar inputs are stable across
  passes (low EMA of pass-to-pass variance).
- **Coefficient stability**: K's coefficients are no longer changing
  significantly between updates (EMA of update magnitudes below threshold).

v0 proposes a hybrid: graduation requires *both* coefficient stability
(K has converged) and a downstream-quality metric. Specifics TBD.

Once graduated, K influences the forward pass per §5; pre-graduation,
K's contribution is gated to zero regardless of its internal state.

## 8. Edge Squaring (Capacity Growth) **[OPEN — mirror attention]**

Per-slice analogue of head squaring (attention spec §5.4). When K's clip
rate (the fraction of B-spline coefficients firing clip-or-lift in
mechanism #1) stays above `TauSquaring` for `KSquaring` consecutive
sweeps, the slice's rank `r` grows by one (or doubles, TBD), capped at
`MaxRank`.

This is the per-slice version of "add capacity when measured pressure
demands it." The trigger mechanism, EMA computation, and atomic activation
all carry over directly from attention's spec §5.4. The *effect* (grow
KAN rank vs. allocate a new attention head) is the per-slice equivalent.

Open: whether the new rank is initialised by perturbation of an existing
rank (analogous to head-squaring's log-normal perturbation) or cold-started
to zero contribution.

## 9. Inference-Time Continuation

Closed.

After training and `LockToInference` is called (per attention spec §6.5),
all per-slice K continue to evolve via the same NLMS rule used during
training. The graduation criterion remains in force — newly-graduated
slices begin influencing the forward pass; already-graduated slices
continue refining.

**Safety bound.** K's coefficients remain bounded by:
- GM=1 gauge per slice (attention spec §5.3 applied to slice coefficients).
- Mechanism #1 per slice (attention spec §5.2 applied to slice coefficients).
- HMax-equivalent cap on rank (§8).

The deployed model therefore cannot destabilise through unbounded
adaptation. The KAN can re-shape its correction to track drift but cannot
inflate or oscillate.

## 10. Composition with Attention KAN **[OPEN — proposed default]**

Per-slice KANs and attention KANs are **independent**: they observe
different signals, update independently, and graduate independently.

- A slice that contains parts of an attention layer's QKV projections
  augments those projections via §5; the per-head softmax is still
  handled by the attention spec's TNNetKANNormaliser.
- The two KANs do not share state, do not coordinate, and do not require
  any synchronisation.

This independence is the v0 default. If empirical work reveals interaction
effects (e.g., a per-slice K augmenting the V projection should "know"
that the per-head attention KAN has graduated), a v1 coordination
mechanism can be added.

## 11. Invariants **[OPEN — to be enumerated]**

Inherited from attention spec, applied per-slice:
- **(PS-1)** Per-slice per-window coefficient product preservation.
- **(PS-2)** Per-slice GM=1 after gauge step.
- **(PS-3)** Sign preservation of every coefficient.
- **(PS-4)** Pre-graduation K(x) ≡ 0 exactly.
- **(PS-5)** Inference-time evolution preserves PS-1 through PS-3.

To be sharpened against implementation.

## 12. Parameters and Defaults **[OPEN — consolidated table TBD]**

Proposed defaults gathered from above:
- `SliceSize = 250000`
- `InitialRank = 2`, `MaxRank = 32`
- `KnotCount = 64`, `BasisOrder = 3`
- Mechanism #1 thresholds: inherited from attention spec §11.2.
- Graduation: `N_confirm_slice = 100` (mirroring attention's `N_confirm`).

To be tuned during implementation.

## 13. Open Questions Summary

- §4: Whether A/B projections are linear or KAN-parameterised.
- §6: Observation signal choice (output-only vs. gradient-informed).
- §7: Exact graduation criterion (stability + downstream metric form).
- §8: Squaring increment (rank+1 vs. rank×2) and initialisation policy
  for newly-activated rank dimensions.
- §10: Whether composition with attention KAN ever requires explicit
  coordination.
- §11/§12: Full enumeration of invariants and parameter defaults table.

These are tractable in subsequent iterations of this document and do not
block initial implementation against the closed sections.
