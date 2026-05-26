# KAN Attention v1.1 Backlog

Status snapshot for the next iteration of the KAN attention library. v1.0
ships the spec's core mechanism end-to-end: full lifecycle (training →
LockToInference → Phase M → Phase D), mechanism #1 (clip+lift cascade with
GM=1 gauge), per-head telemetry, persistent state via save/load, plug-in
factory registration for cloned worker threads, plateau-based early
stopping in the SimpleTransformer1M example, and a continuous EMA-driven
SharpenAlpha auto-tuner on top.

This document tracks what's outstanding, grouped by impact and coupling.

---

## Priority 1 — Safety net (failure modes)

Removes a hard crash path. Half-day of focused work; the four items are
coupled and ship as one bundle.

### S4: `DoSquaring` implementation
**Location:** `delphi/neural/neuralkanattention.pas:272`
**Spec:** §5.4.4, §7 step 12
**Effort:** L (~3-4 hours)

`TKANAttentionLayerInfo.CheckSquaring` currently raises
`EKANBadState('DoSquaring not implemented')` if the squaring trigger ever
fires. Implementation needs to:

- Apply log-normal coefficient perturbation to the new head's spline
  coefficients (seeded from `TKANSeededRNG`)
- Inherit the parent head's coefficient vector as the starting point
  per §5.4.4
- Bump `FActiveHeads` (capped at `FHMax`)
- Reset per-head counters on the new head
- Increment `FCascadeCapHits` accounting if appropriate

**Failure mode without this:** any dataset complex enough to push clip
rates past `τ_squaring` crashes during inference. For TinyStories
char-level this may never fire; on richer data it almost certainly does.

### P1/P2/P3 bundle: normaliser → layer-info backref
**Location:** `delphi/neural/neuralkannormaliser.pas` (Compute orchestrator
+ several helpers), `delphi/neural/neuralkantypes.pas` (forward decl)
**Effort:** M (~2-3 hours, blocking dependency for S4 to be reachable from real data)

Three coupled compromises that resolve together once the backref lands.

**P1:** `LayerClipRate := 0` hardcoded at `neuralkannormaliser.pas:360` →
`ResolveShareMode` always picks Mode A (proportional/proportional, gentle
preserve-shape). Real mode selection requires reading
`Info.FClipRateEMA`.

**P2:** `ZeroOutputForInactive` (`neuralkannormaliser.pas:997`) declared
but never called. Invariant HS-3 requires it when `HeadIndex >=
ActiveHeads`. Without it, inactive heads contribute non-zero noise to
downstream layers.

**P3:** Per-head clip counts from `Mechanism1Sweep` / `SweepOnce` never
reported back to `Info.RecordSweepClipRate`, so `FClipRateEMA` stays 0 →
the squaring trigger can't fire from real data even after S4 lands.

**Unblock path:** add a backref from `TNNetKANNormaliser` to its owning
`TKANAttentionLayerInfo`. Cleanest implementation: forward-declared
opaque pointer in `neuralkantypes.pas`, populated by
`AddKANSelfAttention` when the normaliser is registered.

---

## Priority 2 — Enhancements (likely measurable wins)

Build on v1.0 mechanisms; each is independently shippable.

### Per-head adaptive α
**Spec:** extends v1.1 auto-tuner (not in original spec)
**Effort:** M (~2-3 hours)
**Depends on:** nothing (v1.0 auto-tuner already exists)

The v1.0 auto-tuner sets a single α uniformly across all normalisers.
Different heads serve different attention roles; uniform α leaves
expressive capacity on the table.

Add `TKANNet.SetAlphaForHead(LayerIdx, HeadIdx, Value)` and extend
`CalibrateAlpha` with a per-head mode that varies one head at a time
while holding the rest at their current value. Local search per head; α
trajectory EMA-smoothed independently per head.

Cost scales as `Iterations × HeadsTotal × SamplesPerIter × 3` forwards
— for the SimpleTransformer1M (32 heads total) that's ~30k forwards,
~20 minutes on the reference hardware.

### Rescale-and-spread for Phase M `Exp(0.5*S)` overflow
**Location:** `delphi/neural/neuralkannormaliser.pas:585` (current
short-circuit), `~533-544` (Phase M target computation)
**Effort:** S (~1-2 hours)

Currently guarded by a no-support short-circuit: if `FBasis.Evaluate`
returns zero basis values, Phase M skips the row entirely. Cleaner
approach: detect approaching overflow, rescale the row, and spread the
overshoot across neighbouring positions — same energy-conservation
pattern mechanism #1 uses for excess clip energy.

Makes Phase M robust on out-of-grid scores without dropping the NLMS
signal entirely.

### Training stabilisation (urgent v1.0 fix surfaced by initial run)
**Location:** `delphi/examples-kan/SimpleNLP/SimpleTransformer1M.dpr`
**Status:** prototype landed (`ClipAndSpreadWeights` + OnAfterStep hook)
**Effort:** S (extension)

First long-run training collapsed at epoch 28 (ValLoss 1.37 → 1.98 → 3.21
in three epochs, generation went to character soup). Root cause: the
example disabled every training stabiliser (`InitialLearningRate=0.01`,
`Inertia=0`, `LearningRateDecay=0`, `L2Decay=0`). With nothing damping
the optimiser, the 2nd attention block's Q/K projections entered a
positive-feedback weight-growth regime around epoch 27 and saturated
the dot-product output range. Once attention went uniform, the FFN
couldn't compensate and generation collapsed into unigram noise.

Prototype fix landed in the example: per-neuron energy-conserving
weight clipping in `OnAfterStep`. For each neuron, weights with
|w| > `csWeightClipMax` are clipped to ±`csWeightClipMax` and the
excess magnitude is redistributed across the remaining weights in
the same neuron, proportional to their existing magnitude. Preserves
L1 norm per neuron while bounding max weight magnitude.

This is the Mechanism-#1 design philosophy applied to network weights.
The optimiser can't drop excess into the void — it has to spread,
which disrupts winner-take-all dynamics that drive late-training
runaway.

v1.1 extensions to evaluate:
- Iterate to convergence (currently single-pass; redistributed weights
  may slightly exceed threshold)
- Per-layer (rather than per-neuron) variant to compare
- Threshold scheduling (relaxed early in training, tightened late)
- Companion fixes: add `LearningRateDecay := 0.97`, `Inertia := 0.9`,
  `L2Decay := 1e-4` — any one of these alone would also prevent the
  observed failure mode, but the energy-conserving variant is the
  novel contribution worth profiling

### Self-supervised continuous adaptation during generation
**Effort:** L (~half-day)
**Depends on:** per-head adaptive α (logical follow-on)

The v1.0 auto-tuner adapts α using labelled validation data, then locks
the calibrated value. During free generation there's no label signal, so
α stays frozen. Self-supervised proxies:

- **Output entropy:** lower entropy = more confident final softmax
- **Top-k confidence margin:** gap between top-1 and top-2 logits
- **Attention entropy:** distribution sharpness at attention layers

EMA-driven slow drift on α using one of these as the loss signal.
Allows α to track regime changes during long generation runs (different
context lengths, topic shifts, etc.).

Research-y — ship after the others have validated.

---

## Priority 3 — Polish

Quality improvements with no failure mode.

### P5: LS-fit for `FitPsiToExp`
**Location:** `delphi/neural/neuralkanbasis.pas:173`
**Effort:** S (~1 hour)

Simple collocation instead of tridiagonal least-squares solve. ~1% error
at non-knot points. Phase M NLMS refines from here so impact on
end-to-end quality is small. Pure quality-of-fit improvement; nice to
have for clean cold-start curves.

---

## Spec maintenance (ambiguities to tighten)

Documented in code where decisions were made; spec should be updated to
match.

### §5.5.2 vs §5.1: Phase M target
**Location:** `delphi/neural/neuralkannormaliser.pas:533-544`

§5.5.2 says target is `exp(s)`. §5.1 defines `phi = psi²`, which makes
the consistent target `exp(s/2)` (so that `psi² = exp(s)` matches
post-square). Code uses `Exp(0.5*S)` per the §5.1-consistent reading.

Spec should pick one and edit the other to match.

### §5.5.3: NLMS chain-rule factor
**Location:** `delphi/neural/neuralkannormaliser.pas:620-626`

§5.5.3 NLMS step omits the chain-rule `2·psi` factor that would appear
in a strict gradient derivation. Code follows spec literally. Direction
is correct, magnitude is auto-tuned by the NLMS denominator, so empirics
are fine — but the derivation in the spec is incomplete.

Either add the factor to §5.5.3 with a re-derivation, or add a
justification note explaining why the simplified form is sufficient.

---

## Out of scope (already deferred per spec)

Listed for completeness; not v1.1 work.

- **§6.4:** Optional task-loss parity check — reserved for supervised-eval
  deployments.
- **§9.3.1/9.3.2:** Batch-boundary thread-safety strategy — performance
  optimisation; v1 runs serially per the spec.
- **§8.6:** Cross-version basis upgrade — explicitly out of scope per spec.

---

## Suggested execution order

1. **Safety net (P1+P2+P3+S4 as one bundle)** — half-day, removes the
   only hard crash path. Ship as one PR.
2. **Per-head adaptive α** — natural extension of the v1.0 auto-tuner,
   biggest expressive win after the safety net lands.
3. **Rescale-and-spread for Phase M** — makes the NLMS path robust on
   pathological scores.
4. **P5 LS-fit** — optional polish.
5. **Self-supervised continuous adaptation** — research-y; ship last
   after the others have validated.

Spec maintenance items can land any time alongside whichever PR is in
flight.
