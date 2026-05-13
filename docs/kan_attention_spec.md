# KAN Attention Normaliser — Design Specification

**Version:** v1 (mechanism-locked, pre-implementation)
**Scope:** language-independent design document

---

## 1. Abstract

A drop-in replacement for the softmax operator inside multi-head self-attention. The layer is trained entirely as standard softmax attention; at inference time a per-head B-spline normaliser (a "KAN edge") autonomously learns to mimic softmax, takes over the forward path per-layer when converged, and thereafter evolves to surface information that a fixed-shape softmax cannot express. Retrofits onto any pre-trained softmax transformer with no retraining.

**Non-goals:** KAN-ifying Q, K, V or output projections; replacing any part of training; supporting training-time gradient flow through the KAN normaliser.

---

## 2. Design Goals

The design is a layered response to a single overarching goal: **make more information available at inference than a fixed-shape softmax can express, without retraining and without compromising deployment safety.** Each mechanism in the spec exists to discharge a specific sub-goal:

| Goal | Mechanism delivering it |
|---|---|
| Sharper attention on fewer heads | B-spline shape freedom beats fixed `exp` |
| Safe deployment of novel machinery | Shadow-mimic-then-takeover; softmax is the known baseline |
| No gradient machinery at inference | Normalised LMS (autonomous local rule, no autograd) |
| Surface information a small model would lose | Post-takeover self-distillation + product-preserving budget |
| Self-regulating capacity | Head squaring with self-limiting dynamics |
| Full reproducibility | Deterministic-given-state; all state checkpointable |
| Retrofit on existing models | Training is untouched; any pre-trained softmax model plugs in |

**Explicit non-goals:**

- Replacing softmax during training. Training stays standard.
- Replacing Q/K/V or output projections in v1. They contribute no nonlinearity, and they already receive full task-loss gradient during training; KAN-ifying them would be cargo-culting.
- Supporting backpropagation through the KAN at inference. All inference-time evolution is driven by autonomous local rules.
- Maintaining backwards-bit-compatibility with the original softmax forward path after a layer hands over. The whole point is that the KAN diverges from softmax post-takeover.

---

## 3. Training Phase

Training is **completely vanilla softmax attention**. Q, K, V and output projections remain standard linear layers. The softmax operator is the standard softmax. **No KAN state exists during training.**

Consequences of this choice:

- **Indistinguishable trained model.** A model trained against this specification's attention layer is bit-identical (up to RNG seeding) to one trained against the host library's existing softmax attention. Standard checkpoints work.
- **No new gradient paths.** Backpropagation flows through the standard softmax exactly as it always has. No new gradient machinery, no second optimiser, no auxiliary loss.
- **No new hyperparameters during training.** Learning rate schedules, batch sizes, regularisation, etc. are unchanged.
- **Retrofit is trivial.** Any pre-trained softmax transformer (regardless of when, where or how it was trained) can be loaded into this layer. The KAN state is initialised at first forward pass, not at training time.
- **Failure mode is graceful.** If the KAN machinery is disabled at inference (for whatever reason), the layer is functionally identical to standard softmax attention. There is no path by which "training broke because the KAN was being trained."

**Implementation note (abstract).** The new layer class must, in its training-time forward and backward implementation, be byte-identical in behaviour to the host library's existing softmax attention layer. The KAN-specific state and operations are gated by an `inference-mode` flag that is `false` during training and only flips `true` once the network enters inference. This gating is the single point of departure from the existing layer.

---

## 4. Inference Lifecycle

At inference time, each head of each attention layer progresses independently through three phases. Layers do not have to be in the same phase as one another; per-head state is independent.

### 4.1 Phase M — Mimicry (default on fresh deploy)

- Forward path uses **softmax**.
- A per-head B-spline `φ(s) = Σ_j c_j · B_j(s)` runs in **shadow** alongside the softmax. It receives every score the softmax sees but its output does not affect the layer's attention weights.
- An **autonomous local rule** updates the spline coefficients toward the target `exp(s)` (see §6.4 for the rule, §5.1 for the spline parametrisation).
- A convergence metric `KL(softmax_weights ‖ KAN_weights)` is tracked as an exponential moving average (`KL_ema`).
- The phase is observation-only from the model's behaviour standpoint: the deployed model's attention outputs are exactly those of the standard softmax model. **There is zero behavioural risk during this phase.**

### 4.2 Phase T — Takeover (atomic transition)

- Triggered when `KL_ema < ε_KL` for at least `N_confirm` consecutive forward passes (see §6 for criteria and defaults).
- Performed atomically per head per layer:
  1. The forward path switches from `softmax(s)` to `φ(s) / Σφ(s)`.
  2. The Phase-M mimicry rule retires.
  3. The Phase-D divergence rule activates.
  4. The per-head `status` flag flips from `SoftmaxActive` to `KANActive`.
- **One-way.** No mechanism reverts a head from `KANActive` back to `SoftmaxActive`. Stability of the post-takeover dynamics is the responsibility of mechanism #1 (§5.2) and head squaring (§5.5).
- **Per-head independence.** Different heads of the same layer may take over at different times. Different layers progress entirely independently.

### 4.3 Phase D — Divergence (post-takeover)

- Forward path uses the **KAN**.
- The local rule is no longer mimicry; it is **self-distillation toward a sharpened version of the head's own current attention distribution** (§6.4).
- Combined with mechanism #1's product-preserving redistribution (§5.2), the sharpening rule cannot grow mass — it can only *redistribute* mass within the head's GM=1 budget. This is the mechanism by which information that softmax would have averaged-out gets concentrated into peaks at score regions that consistently carry signal.
- Phase D runs **indefinitely**. There is no terminal state; the head keeps adapting to the inference data distribution as long as inference continues.

### 4.4 Lifecycle State Diagram

```
                          per head, per layer

   ┌──────────────────┐      KL_ema < ε_KL       ┌──────────────────┐
   │                  │      sustained over      │                  │
   │     Phase M      │  ──── N_confirm ───▶     │     Phase D      │
   │     (Mimicry)    │       inferences         │   (Divergence)   │
   │                  │                          │                  │
   │ status =         │                          │ status =         │
   │   SoftmaxActive  │                          │   KANActive      │
   │ forward = softmax│                          │ forward = KAN    │
   │ rule = NLMS→exp  │                          │ rule = self-dist │
   │                  │                          │                  │
   └──────────────────┘                          └──────────────────┘
                                                         │
                                                         │ no transition
                                                         ▼
                                                    (terminal)
```

### 4.5 Non-Convergent Heads

If a head's `KL_ema` never satisfies the convergence criterion (e.g. score distribution is non-stationary or the spline grid is misconfigured for the observed range), the head simply remains in Phase M indefinitely. The layer continues to serve correctly via softmax. This is graceful degradation, not failure.

A head that remains in Phase M at checkpoint time is saved with `status = SoftmaxActive`; on reload, it resumes mimicry from its current coefficient state.

---

## 5. Core Components

The KAN attention normaliser is composed of five interacting components. Each is described in its own subsection.

### 5.1 B-Spline Normaliser (per head)

**Parametrisation.**

- Each head owns one B-spline `φ : ℝ → ℝ` defined on a fixed grid of knot positions covering the expected range of the attention scores `s_ij = (Q · Kᵀ)_{ij} / √d`.
- The spline is `φ(s) = Σ_{j=1}^{N} c_j · B_j(s)`, where:
  - `B_j(·)` are the (immutable) B-spline basis functions of order `k`.
  - `c_j ∈ ℝ` are the learnable coefficients (the entire mutable state of the head's normaliser).
  - `N` is the number of knots (basis count).

**Locality.** B-spline basis functions of order `k` have **local support over `k+1` adjacent knot intervals**. Consequently each coefficient `c_j` only influences `φ(s)` for `s` in a small neighbourhood around its knot. This locality is load-bearing for the entire design:

- It defines the natural **window** for mechanism #1 (§5.2): the window of `c_i` is the set of coefficients whose support overlaps `c_i`'s support, naturally `2k+1` coefficients on a uniform grid.
- It allows the autonomous local learning rule (§6.4) to update only the small subset of coefficients whose basis functions are non-zero at the observed score, keeping per-update cost O(`k+1`) rather than O(`N`).

**Output and normalisation.**

- The forward attention weights are computed by applying `φ` pointwise to every score in the row-major attention matrix and then row-normalising:
  ```
  w_ij = φ(s_ij) / Σ_j' φ(s_ij')
  ```
- Because the output is row-normalised, **multiplying every coefficient `c_j` by any positive constant leaves `w_ij` unchanged**. This scale-invariance is the gauge freedom exploited by the GM=1 fixing (§5.3).
- For the output to constitute a valid attention distribution it must be non-negative. A non-negativity constraint on `φ(s)` is required (see §5.1 Constraints below).

**Constraints.**

| Constraint | Reason | Enforcement |
|---|---|---|
| `φ(s) ≥ 0` for all `s` in the grid range | Attention weights must be non-negative | Either (a) parametrise the spline as `φ(s) = ψ(s)²` with an underlying spline `ψ`, or (b) accept occasional negative outputs and clip to zero with a small floor `ε_φ`. **Default: option (a).** |
| Coefficients sign-preserving across rebalance | Stability of the spline shape under mechanism #1 | Mechanism #1's multiplicative redistribution always uses positive factors. |
| GM=1 per head | Canonical gauge; meaningful thresholds | Renormalisation step in the per-inference pipeline (§7). |

**Initialisation.**

- At layer construction (or first retrofit load), coefficients are fitted such that `φ(s) ≈ exp(s)` over the grid range. Because the non-negativity constraint is satisfied via the `φ = ψ²` parametrisation (see Constraints above), the fit is performed **in ψ-space**: `ψ(s) ≈ exp(s/2)`, so that `φ = ψ² ≈ exp(s)`. A simple least-squares fit on a dense grid of `(s, exp(s/2))` pairs suffices. The stored coefficients `c_j` are ψ's, not φ's.
- Phase M therefore starts with the KAN already approximating softmax. `KL_ema` begins near zero rather than at random, dramatically shortening Phase M for retrofit deployments.

**Grid configuration.**

- Grid range: covers the expected `Q·Kᵀ/√d` range. Default `[−8, +8]` nats; tunable per layer if scores are known to live in a different range.
- Knot count `N`: trades expressivity for memory. Default `N = 64`. Each head holds `N` floats of state.
- Basis order `k`: cubic (`k = 3`) by default, giving window size `2k+1 = 7`.
- Grid is **immutable** for the lifetime of a layer. (A v3 extension may add online grid refinement; not in v1.)

**Memory footprint per head.**

- `N` floats for coefficients.
- One `N`-sized scratch buffer for evaluation (shared across heads at runtime if memory is tight).
- Negligible compared to Q/K/V projection weights, which dominate per-layer memory.

### 5.2 Mechanism #1 — GM-Based Coefficient Rebalance

The central regulator of the spline. Runs every inference step on every head, after the local learning rule (§6.4) has applied its update. Iterates to fixed point with a hard safety cap on iterations.

#### 5.2.1 Window

For each coefficient `c_i`, its **window** is the set of coefficients whose B-spline supports overlap `c_i`'s support. On a uniform grid with order-`k` basis this is the `2k+1` coefficients centred on `c_i` (clamped near boundaries). Groups self-form from the basis structure; no manual group definition is required.

The window is the unit on which all of mechanism #1's invariants are defined.

#### 5.2.2 Window Statistic — Geometric Mean

For each window, compute:
```
g = exp( mean_{j ∈ window}( log |c_j| ) )
```

Computed in log-space for numerical safety. No `+ε` floor is required because the low-lift rule (§5.2.4) keeps `|c_j|` strictly bounded away from zero.

The geometric mean is the right statistic for a multiplicative regime: it is scale-equivariant (`GM(α·X) = α·GM(X)`), it is naturally robust to a single dominating outlier (the GM is barely affected by extreme single values, unlike the arithmetic mean), and it is the correct centre of mass under the GM=1 gauge (§5.3).

#### 5.2.3 High Clip

If `|c_i| > 3 · g`:

1. Clip the offending coefficient: `c_i ← sign(c_i) · 3 · g`.
2. Compute the multiplicative excess: `r = |c_i_old| / |c_i_new| > 1`.
3. Redistribute `r` across the remaining `N−1 = 2k` window coefficients by multiplicative factors `f_j` satisfying:
   ```
   c_j ← c_j · f_j        for j ≠ i
   Π_{j ≠ i} f_j  =  r
   ```
   The factors are positive (sign-preserving). The exact distribution rule depends on the share mode (§5.2.5).

By construction, the **window product `Π_{j ∈ window} |c_j|` is preserved exactly** by this operation:
```
Π_new = (|c_i| / r) · (Π_{j ≠ i} |c_j| · f_j)
      = (|c_i| / r) · r · Π_{j ≠ i} |c_j|
      = Π_old
```

#### 5.2.4 Low Lift (Productivity Rule)

If `|c_i| < g / 2`:

1. Lift the deficient coefficient toward the window's GM: `c_i ← sign(c_i) · g`.
2. Compute the multiplicative deficit: `q = |c_i_new| / |c_i_old| > 1`.
3. Shrink the remaining window coefficients by multiplicative factors `f_j` satisfying:
   ```
   c_j ← c_j · f_j        for j ≠ i
   Π_{j ≠ i} f_j  =  1 / q
   ```

Same product-preservation property as the high clip.

**Threshold rationale.** The low threshold is `g / 2`, not the symmetric `g / 3`. The asymmetry is intentional: the low-lift rule is doing **productivity** work (no dead weight), not merely numerical safety. Keeping every coefficient contributing maximises the per-head representational budget, which makes the head squaring (§5.5) trigger a meaningful signal of genuine capacity demand rather than wasted-budget noise. See §5.5 for the coupling.

#### 5.2.5 Share Rule for Redistribution

The factors `f_j` are computed as `f_j = r^{s_j}` (high clip) or `f_j = (1/q)^{s_j}` (low lift), where the **share weights `s_j` sum to 1** over the redistributed coefficients.

Two share rules are supported, controlled per-event:

- **Proportional:** `s_j ∝ |c_j| / Σ_{k ≠ i} |c_k|`. Larger neighbours absorb more growth (high-clip) or shrinkage (low-lift).
- **InverseProportional:** `s_j ∝ (1 / |c_j|) / Σ_{k ≠ i} (1 / |c_k|)`. Smaller neighbours absorb more.

Two flags select rules independently:
```
HighClipShare ∈ { Proportional, InverseProportional, Auto }
LowLiftShare  ∈ { Proportional, InverseProportional, Auto }
```

##### Mode taxonomy and stability

| Combination | Name | High-clip behaviour | Low-lift behaviour | Stability |
|---|---|---|---|---|
| Prop, Prop | Mode A — *preserve shape* | Big neighbours grow more (creates outliers) | Big neighbours shrink more (flattens) | Mixed: bad on high-clip, good on low-lift |
| Inv, Inv | Mode B — *reverse* | Small neighbours grow more (flattens) | Small neighbours shrink more (creates new zeros) | Mixed: good on high-clip, bad on low-lift |
| Inv, Prop | **Mode C — *always-flatten*** | Flattens (good half of B) | Flattens (good half of A) | **Stable; converges fast** |
| Prop, Inv | Mode D — *anti-flatten* | Creates outliers + creates zeros | — | **Pathological; oscillates. Excluded.** |

Mode C isolates the stabilising half of each pure mode and is strictly the safest. Mode A is useful when sharpness preservation is desired and clip pressure is low. Mode B is rarely useful in isolation. Mode D must never be selected.

##### Auto mode (default for both flags)

`Auto` selects the mode based on the layer's current `clip_rate` (defined in §5.5):

```
if clip_rate < τ_calm:                          -- e.g. 2 %
    HighClipShare := Proportional               -- Mode A: preserve shape
    LowLiftShare  := Proportional
elif clip_rate > τ_stressed:                    -- e.g. 10 %
    HighClipShare := InverseProportional        -- Mode C: always-flatten
    LowLiftShare  := Proportional
else:
    -- hysteresis band: keep current mode
```

Mode D is structurally excluded by the Auto logic.

#### 5.2.6 Cascade to Fixed Point

After processing all coefficients in one sweep, a coefficient that was a *recipient* of redistribution may now itself violate the high-clip or low-lift threshold (because its window's `g` has shifted). Mechanism #1 therefore iterates:

```
for iter in 1 .. max_iter:
    fired = 0
    for i in 1 .. N:
        recompute g for c_i's window
        if |c_i| > 3·g: high-clip; fired += 1
        elif |c_i| < g/2: low-lift; fired += 1
    if fired == 0: break       -- fixed point reached
```

**Convergence.** Empirically the cascade terminates in 1–3 iterations under default parameters and Mode C share rules; formal convergence is **not proven in v1** and is not relied upon by any other part of the spec. The hard cap `max_iter = 16` is a safety guard against pathological inputs.

**Behaviour at the cap.** If the cascade hits `max_iter` without `fired == 0`, the sweep terminates with the current state and the per-layer counter `cascade_cap_hits` (§8.1) is incremented. The partially-converged state still satisfies invariant **M1-1** (every individual high-clip and low-lift step preserves window products by construction), so energy conservation is intact. Only invariant **M1-3** (bounded coefficients) may be violated for that pass; the next pass's sweep will continue the relaxation. Sustained nonzero `cascade_cap_hits` across many sweeps is a misconfiguration signal (e.g. score range mismatched to grid, pathological input); it does not by itself break the spec's safety floor.

#### 5.2.7 Counters Updated

Each invocation of mechanism #1 updates two counters used elsewhere in the spec:

- `clip_rate` (per layer): exponential moving average of the fraction of coefficients firing clip-or-lift in the most recent sweep. Used by Auto mode (§5.2.5) and by head squaring (§5.5).
- `clip_count_total` (per head, per phase): cumulative count of clips and lifts since checkpoint load. Diagnostic only.

#### 5.2.8 Invariants Maintained

After each invocation of mechanism #1, on every head:

- **(M1-1)** Per-window product: `Π_{j ∈ window} |c_j|` is unchanged from before the sweep, up to floating-point tolerance.
- **(M1-2)** Sign of every coefficient is unchanged.
- **(M1-3)** No coefficient is below `g/2` or above `3·g` of its own window's `g`, after cascade termination.
- **(M1-4)** Scale-equivariance: applying mechanism #1 to coefficients `(c_j)` then multiplying by `α > 0` produces the same result as multiplying by `α` first then applying mechanism #1.

### 5.3 GM=1 Gauge Fixing (per head)

#### 5.3.1 Premise — The Gauge Freedom

Because the attention weights are computed by row-normalisation:
```
w_ij = φ(s_ij) / Σ_j' φ(s_ij')
```
multiplying every coefficient `c_j` by any positive scalar `α > 0` leaves `w_ij` **identically unchanged**. The KAN therefore has a **gauge freedom**: a one-parameter family of coefficient configurations that all produce the same forward output.

This freedom is the only direction in which the coefficient vector can drift without changing the model's behaviour. Pinning that direction is free — it costs the model nothing in expressive power — and pays off in stability, interpretability, and the meaning of mechanism #1's thresholds.

#### 5.3.2 The Gauge Choice — `GM = 1`

The chosen gauge is **per-head geometric mean equal to 1**:
```
G(head) := exp( mean_j( log |c_j| ) )  =  1
```

Equivalently: the sum of log-magnitudes of all coefficients in a head is zero. Some coefficients have `|c_j| < 1`, some have `|c_j| > 1`, and they multiplicatively balance.

**Why per-head, not per-layer:** each head is an independent representational budget (§5.5). Per-head gauge gives each head its own canonical scale; per-layer gauge would tie heads together and dilute the "more heads = more budget" interpretation.

#### 5.3.3 The Renormalisation Operation

After mechanism #1 has converged in a given inference step (§5.2.6), per head:

```
G   := exp( mean_j( log |c_j| ) )         -- in log-space, O(N)
c_j ← c_j / G                              -- uniform rescale, O(N)
```

That is the entire operation. One pass to compute `G`, one pass to apply.

**Numerical implementation note.** Use log-space throughout: maintain or recompute `Σ_j log|c_j|`, divide by `N` to get `log G`, then `c_j ← c_j · exp(−log G)`. This avoids the `Π|c_j|` underflow/overflow that direct GM computation would suffer for `N` in the dozens.

#### 5.3.4 Why It Costs Nothing in Forward Output

Before renormalisation, the spline outputs `φ(s) = Σ_j c_j · B_j(s)`. After renormalisation by uniform factor `1/G`:
```
φ_new(s) = Σ_j (c_j / G) · B_j(s) = (1/G) · Σ_j c_j · B_j(s) = φ_old(s) / G
```
And the row-normalised attention weight:
```
w_new = φ_new(s) / Σ φ_new = (φ_old(s)/G) / (Σ φ_old/G) = φ_old(s) / Σ φ_old = w_old
```

The renormalisation is a pure gauge transformation: the attention output is **mathematically equal** before and after, with the only difference being the floating-point rounding introduced by the per-coefficient rescale `c_j ← c_j / G`. The resulting per-weight error is bounded by `O(N · ε_machine)`, well below `ε_KL` and below any threshold the rest of the spec relies on.

#### 5.3.5 Composition with Mechanism #1

Mechanism #1 preserves **per-window products** (M1-1). The GM=1 renormalisation applies a uniform multiplicative rescaling, which **changes** every window product by `(1/G)^(2k+1)`. There is no contradiction: mechanism #1 preserves whatever window products it finds at the start of its sweep. The renormalisation simply resets the "preserved targets" to the new gauge.

The locked ordering (also encoded in §7) is:

```
1. local learning update (additive, perturbs both window products and G)
2. mechanism #1 sweep (cascades to fixed point; preserves whatever products it sees)
3. GM = 1 renormalisation (uniform rescale; resets gauge)
```

After step 3, the head is in a canonical state: GM=1 and all window products consistent with that gauge.

#### 5.3.6 What the Gauge Buys

- **Meaningful thresholds.** Mechanism #1's `3·g` and `g/2` thresholds are now expressed against a stable reference. With GM=1, the typical window `g` is approximately 1, so coefficients live (approximately) in `[1/6, 6]`. Thresholds remain comparable across layers, across heads, and across time.
- **No coefficient-magnitude drift.** Without the gauge, the additive learning updates would slowly inflate or deflate the overall coefficient scale (depending on the data distribution). The gauge eliminates this nuisance dynamic entirely.
- **Numerical stability.** `log |c_j|` is bounded; `Σ log |c_j|` cannot underflow or overflow; the GM computation is well-conditioned regardless of `N`.
- **Foundation for head squaring.** With the gauge in place, each head's representational budget is well-defined (the bounded coefficient interval), making "this head has saturated its budget" a meaningful statement (§5.5).

#### 5.3.7 Invariants Maintained

After step 7 of the per-inference pipeline (§7), on every head:

- **(GM-1)** `|G − 1| < ε_gauge` where `ε_gauge` is at machine-precision tolerance for the head's coefficient count.
- **(GM-2)** The forward attention weights computed before and after renormalisation are mathematically equal; the per-weight FP error from the rescale is bounded by `O(N · ε_machine)`.
- **(GM-3)** Every coefficient sign is preserved.

### 5.4 Multi-Head Dynamics — Head Squaring

#### 5.4.1 Premise — Each Head Is a Bounded Budget

After mechanism #1 + GM=1 gauge, each head's coefficients live in a bounded interval (approximately `[1/6, 6]` in absolute value, depending on window size). The spline `φ` constructed from those coefficients can express only patterns that fit within this **multiplicatively balanced budget**:

- Cannot have all coefficients large.
- Cannot have all coefficients tiny.
- Must distribute "strength" unevenly with geometric balance around 1.

If the data calls for an attention pattern that cannot fit within one head's budget — for example, multiple sharp peaks across distinct score regions, each requiring high-magnitude coefficients — the head will repeatedly bump against mechanism #1's clip thresholds. **Head squaring is the relief valve: when one head's budget saturates, allocate more heads.**

#### 5.4.2 Allocation — Pre-Allocate, Mask the Rest

At layer construction:

```
H_max := largest_power_of_2_leq( ⌊ embedding_dim / 2 ⌋ )
                                  -- architectural ceiling: ≥ 2 channels per head,
                                  -- and a power of 2 so the squaring progression
                                  -- 2 → 4 → 16 → 256 → … fits cleanly
ActiveHeads := 2                  -- specification-mandated initial value
```

For typical `embedding_dim ∈ {128, 256, 512, 768, 1024, ...}` the rounding-down to a power of 2 changes `H_max` by at most a factor of 2 and is almost always the identity. The restriction is essential for invariant **HS-2** (power-of-2 progression) to hold without exceptions: every squaring event must satisfy `spawn_per_parent = H_new / H_old ∈ ℕ`, which requires `H_max` to itself be a power of 2.

All `H_max` heads are **physically allocated** at construction time (coefficients, RNG state, counters). Heads with index `≥ ActiveHeads` are **inactive**:

- Their KAN coefficients are not updated.
- Their forward contribution is **masked to zero** in the multi-head concat step.
- They consume memory but no meaningful compute.

Pre-allocation eliminates runtime tensor resize, structural network mutation, and the entire class of bugs that come with growing data structures during inference. Squaring is implemented as a single integer assignment plus the per-head copy/perturb of §5.4.4.

#### 5.4.3 Squaring Trigger

Per layer, two counters drive the trigger:

- `clip_rate_ema`: per-layer EMA of the fraction of all coefficients (across all *active* heads) that fired clip-or-lift in the most recent mechanism-#1 sweep.
- `consecutive_high_sweeps`: count of consecutive sweeps where `clip_rate_ema > τ_squaring`.

Trigger condition:

```
if ActiveHeads < H_max
   and consecutive_high_sweeps ≥ K_squaring:
       fire_squaring()
       consecutive_high_sweeps := 0
```

Defaults: `τ_squaring = 0.10` (10 %), `K_squaring = 64` consecutive sweeps. The latter prevents single-batch noise from triggering squaring.

#### 5.4.4 The Squaring Operation

```
H_old := ActiveHeads
H_new := min(H_old · H_old, H_max)        -- square, clamp at ceiling
spawn_per_parent := H_new / H_old          -- integer

for parent_idx in 0 .. H_old - 1:
    for child_local in 1 .. spawn_per_parent - 1:    -- skip 0 (parent keeps slot)
        child_idx := H_old + parent_idx · (spawn_per_parent - 1) + (child_local - 1)
        copy parent's B-spline coefficients into child slot
        for each c_j in child's coefficients:
            c_j ← c_j · exp( N(0, σ²) )    -- log-normal perturbation, seeded RNG
        -- inherit full per-head bookkeeping from parent:
        status[child_idx]                 := status[parent_idx]
        KL_ema[child_idx]                 := KL_ema[parent_idx]
        consecutive_low_passes[child_idx] := consecutive_low_passes[parent_idx]
        clip_count_total[child_idx]       := clip_count_total[parent_idx]

ActiveHeads := H_new
```

Properties:

- Each existing head retains its slot and its coefficients unchanged.
- `H_new − H_old` new heads are populated as perturbed clones of the existing heads.
- The perturbation is **multiplicative log-normal** with `σ ≈ 0.01`. Multiplicative is consistent with the GM=1 gauge: it preserves expected `log |c_j|` (so the new head also satisfies GM=1 to first order) and breaks symmetry without large jumps.
- **Children inherit the parent's `status`.** A child of a `KANActive` parent starts in `KANActive` (the small log-normal perturbation keeps it within the parent's converged basin; restarting in Phase M would be incoherent because the spline shape is no longer a softmax mimic). A child of a `SoftmaxActive` parent starts in `SoftmaxActive` and inherits the parent's `KL_ema` and `consecutive_low_passes`, so it can reach handover quickly if its own converged dynamics permit.
- The RNG used is the **per-layer seeded RNG**, so the operation is deterministic given the saved state (§9).
- Because GM=1 is approximately preserved by the perturbation, the new heads' coefficients land in the same canonical range; no re-gauge is needed at squaring time (the next per-inference step's gauge fixing will tidy any first-order drift).

#### 5.4.5 Growth Pattern and Self-Limitation

Squaring produces **super-exponential capacity growth**:

```
ActiveHeads:  2 → 4 → 16 → 256 → 65536 → ...   (clamped at H_max)
```

Per-head clip pressure drops **roughly linearly in head count** (each head specialises to a smaller fraction of the input distribution). The asymmetry between super-exponential capacity growth and linear pressure relief produces **inherent self-limitation**:

- One squaring event typically reduces `clip_rate_ema` by a factor of `H_new / H_old`.
- After one or two events, `clip_rate_ema < τ_squaring`, the trigger goes silent, and `ActiveHeads` freezes.

**No adaptive threshold is required.** The dynamics produce stable equilibrium on their own. Different layers settle at different `ActiveHeads` values based on their data complexity.

#### 5.4.6 Coupling with the Productivity Low-Lift

The aggressive low-lift threshold (`g/2`, §5.2.4) is essential to the meaning of the squaring trigger:

- **Without aggressive low-lift:** some coefficients drift toward zero and become dead weight. A head's effective capacity is `M < N`, and the active `M` coefficients overload, raising `clip_rate` for reasons that have nothing to do with genuine capacity demand. Squaring fires prematurely on wasted-budget noise.
- **With aggressive low-lift:** every coefficient stays productive. Effective capacity equals nominal capacity. `clip_rate` rises only when the head genuinely needs more representational room. Squaring fires only when capacity expansion is actually warranted.

The two mechanisms together promote head squaring from "noisy event" to "meaningful diagnostic signal."

#### 5.4.7 One-Way Only — No Head Merging

Heads never merge. `ActiveHeads` is monotonic non-decreasing across a deployment run.

Rationale: merging two heads' splines requires choosing how to combine their coefficients. Any combination (average, weighted average, etc.) destroys learned structure that the two heads had specialised for. Mechanism #1's low-lift already handles coefficient-level rescue; head-level merging would risk catastrophic forgetting for no clear gain.

If a deployment scenario requires reducing `ActiveHeads` (e.g. reducing serving cost), the only supported approach is **load a checkpoint from before the squaring event** — not in-place merge.

#### 5.4.8 Failure Mode — Hitting the Ceiling

If `clip_rate_ema` remains above `τ_squaring` even with `ActiveHeads = H_max`, squaring is impossible. The layer enters a **degraded-but-stable regime**:

- Mechanism #1 fires more often than typical (high clip rate).
- The Auto mode (§5.2.5) keeps the layer in Mode C (always-flatten) most of the time.
- The forward output remains correct; only the per-head splines are operating at capacity.

This regime is the right behaviour: it tells the operator "this layer is under-provisioned for the data at the architecture's resolution." The diagnostic value of `ActiveHeads = H_max ∧ clip_rate ≫ τ_squaring` is high; it is a clear, observable signal that the model architecture itself is the bottleneck.

#### 5.4.9 Diagnostic Value of Final `ActiveHeads`

After a deployment has settled, the per-layer `ActiveHeads` distribution is a free readout of the layer's informational complexity on the observed data:

- Layers stuck at 2: simple, uniform attention patterns. Probably could have been smaller in the original architecture.
- Layers at 4–16: moderate complexity, multi-modal attention where it matters.
- Layers at 64+ or hitting `H_max`: doing genuinely complex discrimination work.

This is information not available from any standard transformer at any inspection level. It is a side-effect of the spec, not a designed feature, but it is genuinely useful.

#### 5.4.10 Invariants Maintained

After every per-inference pipeline step (§7), per layer:

- **(HS-1)** `ActiveHeads ∈ ℕ`, `2 ≤ ActiveHeads ≤ H_max`, monotonic non-decreasing across the run.
- **(HS-2)** `ActiveHeads` is a power of 2 at all times. The progression `2 → 4 → 16 → 256 → …` is preserved by the spec's requirement that `H_max` is itself a power of 2 (§5.4.2).
- **(HS-3)** Inactive heads (index `≥ ActiveHeads`) contribute zero to the layer's forward output.
- **(HS-4)** Squaring transitions are deterministic given the per-layer RNG state.
- **(HS-6)** A child head spawned at squaring inherits its parent's `status`, `KL_ema`, `consecutive_low_passes`, and `clip_count_total` as starting state; only the spline coefficients are perturbed.

### 5.5 Autonomous Local Learning

#### 5.5.1 Premise — No Autograd at Inference

Inference must not require backpropagation, gradient tapes, or any optimiser machinery. The KAN evolves entirely through **autonomous local rules**: each spline coefficient updates from quantities locally observable at the layer (the score being processed, the basis activations, and the local residual). No external loss function is constructed; no gradient flows from anywhere outside the layer.

This constraint is satisfied by exploiting a structural property of B-splines: **`φ` is linear in its coefficients**. For linear-in-parameter models the locally optimal least-squares fit step has a closed form per observation (Normalised LMS / Widrow-Hoff), which can be expressed as a per-coefficient additive update with no gradient machinery.

Mathematically, this rule *is* equivalent to SGD-on-MSE for the linear regression problem; structurally, it is a **cellular update rule each coefficient applies to itself**, with no tape, no graph, and no external orchestrator. That structural distinction is what makes it "autonomous" rather than "SGD in disguise" — it slots into a pure-forward inference pass with no machinery extension.

Two rules exist, one per phase. The *form* is identical (NLMS); only the **target** differs.

#### 5.5.2 Phase M Rule — NLMS Toward `exp(s)`

For each attention score `s` evaluated in a forward pass, while the head's `status = SoftmaxActive`:

```
target := exp(s)                                          -- pre-normalisation softmax kernel
y_hat  := φ(s) = Σ_j c_j · B_j(s)
err    := target − y_hat

denom  := Σ_j B_j(s)² + δ                                  -- δ for stability, e.g. 1e-6
for each j with B_j(s) ≠ 0:                                -- only k+1 coefficients in support
    c_j ← c_j + η · B_j(s) · err / denom
```

**Why fit `exp(s)` rather than `softmax(s)`:** `softmax` involves the row-wise normalisation `/ Σ`, which is non-linear in `φ`'s coefficients (because the denominator depends on all coefficients). Fitting `exp(s)` decouples the local update from the row context — each `(s, exp(s))` pair is independently informative. After takeover, `φ(s) / Σφ(s)` then automatically matches softmax, since `exp(s) / Σexp(s) ≡ softmax(s)`.

**Why NLMS rather than plain LMS:** the denominator `Σ B_j(s)² + δ` makes the update **scale-invariant in the basis activations** at the observed score. This auto-tunes the per-step magnitude — a region of the spline with a high-magnitude basis hit gets the same effective step size as a low-magnitude one. Eliminates the need for a per-region learning-rate schedule.

**Locality of compute:** B-spline locality means only `k+1 = 4` coefficients have non-zero basis activation at any single score `s`. Each NLMS update touches only those `k+1` coefficients. Cost per score: O(`k+1`).

#### 5.5.3 Phase D Rule — Self-Distillation Sharpening

The Phase D rule is **row-coupled**: it depends on the full row's pre-normalisation spline outputs, not just the per-score `(s, φ(s))` pair. To keep the rule deterministic and order-independent within a row, the row sum is **snapshotted once at the start of the row** and reused for every per-score update inside that row. Per-score updates within a row do not see each other's perturbations of the spline; the next row recomputes the snapshot from the now-updated coefficients.

For the row containing scores `s_i_·`, while the head's `status = KANActive`:

```
-- snapshot once per row, before any per-score updates --
phi_row[j]      := φ(s_i_j)              for all j in the row
S_row           := Σ_j phi_row[j]
w[j]            := phi_row[j] / S_row
w_sharp[j]      := w[j]^α / Σ_k (w[k]^α)   -- power-α renormalisation, α slightly > 1
target_pre[j]   := w_sharp[j] · S_row      -- back into pre-normalisation space

-- per-score NLMS, using the snapshotted target --
for each score s = s_i_j in the row:
    err   := target_pre[j] − phi_row[j]
    denom := Σ_m B_m(s)² + δ
    for each m with B_m(s) ≠ 0:           -- only k+1 coefficients in support
        c_m ← c_m + η · B_m(s) · err / denom
```

Same NLMS form as Phase M; only the target changes. The target is now derived from the head's **own current attention distribution**, sharpened by raising to a power `α` slightly greater than 1 and re-normalising.

**Why the row snapshot.** Without the snapshot, `Σ φ` would have to be recomputed per score after the previous score's update, which (a) makes the dynamics depend on the within-row score iteration order, breaking determinism guarantees in §9, and (b) couples the per-score updates non-linearly through the changing denominator, making the rule no longer locally analysable. The snapshot is the simplest fix that keeps the rule fully deterministic and per-coefficient local while preserving the row-level mass-budget interpretation.

**Default `α = 1.1`.** Each step is a gentle multiplicative emphasis of larger weights at the expense of smaller ones. Mechanism #1 + GM=1 enforce the budget; only the *redistribution* of mass is free.

**Why power-α rather than temperature-scaled softmax:** the temperature variant `softmax(log(w) / τ)` blows up numerically when any `w_i ≈ 0` (because `log w → −∞`). Power-α (`w^α / Σ w^α`) is well-defined for any non-negative `w`, including zeros. Since the spline output enforces `φ ≥ 0` (§5.1), `w` may legitimately be very small but is never undefined.

#### 5.5.4 Why Phase D Surfaces Information Rather Than Collapsing

The Phase D rule, run on its own without any constraint, would push the spline toward a one-hot distribution at the highest-scoring position — losing all attention information. **It does not, because the system has two conserved quantities that together forbid the one-hot configuration.**

##### Energy-conservation argument

Treat each window's product `E_W := Π_{j ∈ W} |c_j|` as the **multiplicative energy** of that window, and the per-head GM as the head's **total energy**. Two invariants pin these:

- **(M1-1) Per-window energy conservation.** Mechanism #1's redistribution operations are constructed so that every individual high-clip and low-lift step preserves `E_W` exactly (§5.2.3, §5.2.4). The cascade is a sequence of energy-preserving operations on overlapping windows.
- **(GM-1) Per-head energy normalisation.** The gauge step pins the head's total energy `Π_j |c_j|^{1/N} = 1` (i.e. GM = 1) at the end of every inference step (§5.3).

A one-hot attention output requires the spline to peak so sharply that, after row-normalisation, all but one weight is negligible. In coefficient space, this means concentrating effectively all of the head's multiplicative budget into the small number of coefficients (`≤ k+1` per support) whose basis functions activate at the dominant score region. The remaining coefficients would have to be vanishingly small.

But the per-window conservation law forbids this concentration:

- Each window has fixed `E_W`, set by the previous pass's gauge step at `E_W ≈ 1` (because GM=1 implies `mean log |c_j| = 0`, so `mean E_W ≈ 1` window-by-window after a few iterations).
- Driving any one coefficient toward `|c_j| → ∞` while keeping `E_W` fixed requires driving its window-mates toward `|c_j| → 0` in a precise compensating ratio.
- But windows overlap (every coefficient sits in `2k+1` windows). Driving a coefficient to zero in one window drives it to zero in all `2k+1` windows it participates in, each of which then demands compensating growth in *its* window-mates.
- The compensating-growth chains propagate through the overlap graph and meet themselves: a coefficient on the far side of the head, eventually reached by the chain, is required to grow by a factor that contradicts the chain's earlier demand for it to shrink.
- The only configuration consistent with all `N` simultaneous per-window conservation laws is one where every `|c_j|` is finite, bounded, and strictly positive. The one-hot configuration violates this and is therefore **dynamically inaccessible**.

The Phase D sharpening rule can only redistribute mass *within* the conservation-law-respecting manifold. The manifold is bounded and excludes singular configurations.

##### What the rule actually does

Within the allowed manifold, the rule still has a strong preference: it pulls the spline shape toward configurations where the per-row sharpened distribution `w^α / Σw^α` is the steady-state of the local update. This selects for spline shapes that make consistent score regions sharper at the expense of inconsistent ones:

- At score values that consistently carry signal (the same row positions repeatedly land in this region with similar relative magnitudes), the per-row updates reinforce — `target_pre[j] − φ(s)` has a consistent sign — and the local NLMS rule grows `φ` there.
- At score values that carry inconsistent signal (the row position lands here with random relative magnitude), the updates cancel — `target_pre[j] − φ(s)` has random sign — and `φ` does not grow.
- The growth at consistent regions is paid for, via the conservation laws, by flattening at inconsistent regions.

**Net effect:** the spline learns a shape with multi-modal peaks at score values that consistently carry signal in the actual inference data distribution. Score regions that softmax would have crushed (low-score tokens carrying real information) get carved out as new peaks, drawing mass from regions of the spline that were previously redundant. The conservation laws guarantee this redistribution is bounded; the sharpening rule guarantees it is selective.

**This is the design's answer to "surface information that softmax would lose in a small model."** The combination of (a) shape freedom from B-spline parametrisation, (b) bounded budget from per-window energy conservation + gauge normalisation, (c) selective sharpening pressure from self-distillation, and (d) gradient-free local update produces a system that cannot grow without bound but will preferentially concentrate its mass on whatever consistently informative structure the data exhibits.

#### 5.5.5 Cost Analysis

Per forward pass with sequence length `L`, head count `H`, B-spline order `k`:

- **Spline evaluations:** `L²` per head (one per attention score), each O(`k+1`). Total `H · L² · (k+1)` multiply-adds.
- **NLMS updates:** `L²` per head, each O(`k+1`). Total `H · L² · (k+1)` multiply-adds. Same asymptotic cost as evaluation.
- **Mechanism #1 sweep:** `H · N · cascade_iters · (2k+1)` multiply-adds. Typically `cascade_iters ≤ 3`. Independent of `L`.
- **GM=1 gauge:** `H · N` log/exp operations. Independent of `L`.

For typical sequence lengths (`L ≫ N`), the per-pass cost is **dominated by the spline evaluations and updates, both linear in the existing attention's score-matrix cost**. The non-evaluation overhead (mechanism #1 + gauge) is O(`H · N`) per pass, asymptotically free relative to the attention's O(`L² · d`).

#### 5.5.6 Invariants Maintained

- **(LL-1)** Each NLMS update touches at most `k+1` coefficients per score — the ones with non-zero basis activation at `s`.
- **(LL-2)** The update is deterministic given `(s, target, denom, η, current coefficients)`.
- **(LL-3)** The update is sign-preserving in expectation (single-step updates can flip sign in pathological cases; mechanism #1 immediately corrects via low-lift).
- **(LL-4)** No gradient state, no optimiser state, no autograd graph is constructed at any point.

---

## 6. Convergence and Handover Criterion

The transition from Phase M (Mimicry, softmax in forward path) to Phase D (Divergence, KAN in forward path) is governed by a per-head convergence test. The test answers a single question: **"Has this head's KAN learned to reproduce softmax closely enough that swapping it into the forward path is safe?"**

### 6.1 Convergence Metric

The metric is **Kullback–Leibler divergence** of the softmax distribution from the KAN distribution, computed per attention row, averaged over the row's positions:

```
KL_row = Σ_j softmax(s_·j) · ( log softmax(s_·j) − log (φ(s_·j) / Σ_j' φ(s_·j')) )
```

**Direction:** `KL(softmax ‖ KAN)`. Reverse KL — zero-avoiding — penalises the KAN for assigning low probability anywhere softmax assigns mass. This produces a smoother fit during Phase M than forward KL would (forward KL is mode-seeking and can leave undershoots).

**Log base:** **natural log (nats)**, matching the convention used by the host model's training cross-entropy loss. Mixing log bases makes ε thresholds non-comparable across the loss landscape.

### 6.2 EMA Aggregation

`KL_row` is computed for every attention row in every forward pass, then aggregated per head:

```
KL_pass := mean_rows(KL_row)              -- per forward pass
KL_ema  := (1 − λ) · KL_ema  +  λ · KL_pass     -- EMA, λ ≈ 0.01
```

The EMA smooths the per-pass noise. Default decay `λ = 0.01` corresponds to an effective horizon of ~100 passes.

### 6.3 Handover Criterion

A head transitions from `SoftmaxActive` to `KANActive` when:

```
KL_ema < ε_KL    sustained for at least N_confirm consecutive passes
```

Defaults:

- `ε_KL = 0.01 nats`. A KL below this corresponds to attention distributions that differ by less than ~1 % in any individual weight (rule of thumb).
- `N_confirm = 100 consecutive passes`. Prevents handover on a single lucky batch.

### 6.4 Optional Task-Loss Parity Check

If a labelled evaluation stream is available at deployment (e.g. shadow-traffic replay, A/B testing, periodic offline eval batches), an additional safety check can be enabled:

```
L_task(KAN) ≤ L_task(softmax) + ε_task    over the eval stream, every K_eval batches
```

When enabled, both conditions (KL below threshold AND task parity) must hold for handover. When no labels are available (the typical inference deployment), only the KL condition applies.

Defaults if enabled: `ε_task = max(0.005, 0.01 · L_task)`, `K_eval = 100 batches`.

### 6.5 Atomic Handover Action

When the criterion fires for a head:

```
status[head]      ← KANActive
forward_path[head] ← KAN_normaliser     -- φ(s) / Σφ(s)
local_rule[head]  ← SelfDistillation     -- §5.5.3
KL_ema[head]      ← unchanged             -- continues to be tracked for diagnostics
```

The transition is **atomic per head per layer**. Each head's flip is observable in the very next forward pass.

### 6.6 One-Way Property

There is no rule that reverts a head from `KANActive` to `SoftmaxActive`. Reversion would create oscillation under any noise in the post-takeover dynamics. Stability of `KANActive` heads is guaranteed by:

- Mechanism #1's clip and lift, which bound coefficient magnitudes (§5.2).
- The GM=1 gauge, which bounds overall scale drift (§5.3).
- Auto mode in mechanism #1, which switches to flatten-only behaviour under high clip pressure (§5.2.5).
- Head squaring, which adds capacity if the per-head budget is genuinely insufficient (§5.4).

If post-takeover dynamics become pathological in a way these mechanisms cannot recover, the correct action is **operator intervention** (load an earlier checkpoint, reduce the deployment's input distribution shift, etc.) — not automatic reversion.

### 6.7 Initialisation State at First Inference

At the first forward pass after a model loads (whether retrofit or resumed):

```
status[head]           = SoftmaxActive
KL_ema[head]           = +∞               -- will descend over the first ~λ⁻¹ passes
consecutive_low_passes = 0
coefficients           = exp-fit on grid (§5.1, retrofit) OR loaded checkpoint state
```

The forward path produces standard softmax output on the first pass, identical to the original trained model. Phase M begins from there.

---

## 7. Per-Inference Pipeline (Locked Ordering)

For each forward pass through an attention layer using this spec, the operations execute in this exact order. The ordering is **load-bearing** — earlier sections rely on properties that hold only because the operations execute in this sequence.

```
For each attention layer:
  For each active head (h ∈ 0 .. ActiveHeads − 1):

    -- Forward computation --
    1. Compute the head's attention scores
       s = Q_h · K_hᵀ / √d

    2. Evaluate the spline at every score
       φ_h(s_ij) for all (i, j)

    3. Compute both candidate weight matrices
       w_softmax = row_softmax(s)
       w_KAN     = row_normalise(φ_h(s))   -- φ_h(s_ij) / Σ_j' φ_h(s_ij')

    4. Select the forward-path weights according to status
       w = (status[h] == KANActive) ? w_KAN : w_softmax

    5. Compute the attended output for this head
       attended_h = w · V_h

    -- Post-forward state mutation (per head, in order) --
    6. Compute KL_pass from (w_softmax, w_KAN) and update KL_ema

    7. Apply the autonomous local learning rule
       if status[h] == SoftmaxActive: NLMS toward exp(s)        (§5.5.2)
       else:                          NLMS toward sharpened(w)  (§5.5.3)

    8. Run mechanism #1 sweep, cascading to fixed point
       (high-clip + low-lift + product-preserving redistribute, §5.2)
       Update clip_rate_ema and consecutive_high_sweeps counters.

    9. Apply GM = 1 gauge renormalisation
       c_j ← c_j / G_h     (§5.3)

   -- Post-mutation policy checks --
    10. If status[h] == SoftmaxActive
        and KL_ema < ε_KL for ≥ N_confirm consecutive passes:
           perform handover (§6.5)

  -- Per-layer post-head checks (after all heads processed) --
  11. If ActiveHeads < H_max
      and clip_rate_ema > τ_squaring
      for ≥ K_squaring consecutive sweeps:
         perform head squaring (§5.4.4)

  -- Concat and project --
  12. Concatenate per-head attended outputs across all H_max heads.
      Inactive heads (h ≥ ActiveHeads) contribute zero (mask).

  13. Apply the layer's standard linear output projection W_O.
```

### 7.1 Why This Order

- **Step 4 commits the forward output before any state mutation** (steps 6–11). Inference never sees partially-mutated state.
- **Step 7 (learning) precedes step 8 (mechanism #1)** so mechanism #1 cleans up after the additive update's perturbation of window products.
- **Step 8 (mechanism #1) precedes step 9 (gauge)** so the rebalance operates on the pre-renormalised state where window products are meaningfully preserved; the gauge then resets the canonical scale.
- **Step 10 (handover check) is per-head and runs after step 9** so the check uses the canonicalised state.
- **Step 11 (squaring) is per-layer and runs after all heads' bookkeeping** so `clip_rate_ema` reflects the full layer's pressure.
- **Step 12 (concat) is unchanged from standard multi-head attention** — the spec requires no modification to the concat or output-projection plumbing.

### 7.2 Cost of the Post-Forward Steps

Steps 6–11 add work proportional to `H · L²` (KL computation, NLMS updates) plus `H · N · cascade_iters` (mechanism #1) plus `H · N` (gauge). The first term is asymptotically equal to the existing forward score cost; the latter two are negligible relative to the attention's `O(L² · d)` cost. Net per-pass overhead during Phase M is approximately **2× standard softmax attention**; during Phase D it drops to approximately **1.2× standard softmax attention** (no `w_softmax` computation needed in step 3, no mimicry target evaluation in step 7).

A configuration flag may disable Phase M's `w_softmax` computation in step 3 once a head has handed over (skip the redundant softmax). Default: enabled (skip).

---

## 8. State and Checkpointing

### 8.1 Per-Layer State Schema

Every attention layer using this spec carries the following state, which must round-trip through any save/load operation:

| Field | Type / shape | Purpose |
|---|---|---|
| `coefficients` | `float[H_max, N]` | B-spline coefficients per head; one row per (possibly inactive) head. |
| `ActiveHeads` | `int` | Current active head count, `2 ≤ ActiveHeads ≤ H_max`. |
| `status` | `enum[H_max]` | Per-head phase flag: `SoftmaxActive` or `KANActive`. |
| `KL_ema` | `float[H_max]` | Per-head convergence EMA (§6.2). |
| `consecutive_low_passes` | `int[H_max]` | Per-head counter for handover criterion (§6.3). |
| `clip_rate_ema` | `float` | Per-layer rebalance activity EMA (§5.2.7). |
| `consecutive_high_sweeps` | `int` | Per-layer counter for squaring criterion (§5.4.3). |
| `clip_count_total` | `int[H_max]` | Per-head diagnostic counter (§5.2.7). |
| `cascade_cap_hits` | `int` | Per-layer diagnostic counter: number of mechanism-#1 sweeps that hit `max_iter` without reaching fixed point (§5.2.6). Sustained nonzero values indicate misconfiguration. |
| `rng_state` | `opaque` (per implementation) | Per-layer seeded RNG used for head-split perturbations (§5.4.4) and any other randomness. |
| `grid_spec` | `(low: float, high: float, knots: int, order: int)` | Immutable B-spline grid configuration. Saved for cross-version compatibility. |

### 8.2 Immutable vs Mutable

- **Immutable** (set at layer construction, never modified): `grid_spec`, `H_max`, the basis function lookup tables (derived from `grid_spec`).
- **Mutable** (evolves at every inference step): `coefficients`, `KL_ema`, `clip_rate_ema`, all counters, `rng_state`.
- **Mutable but rare**: `status`, `ActiveHeads`. These change only on handover events and squaring events respectively.

### 8.3 Checkpoint Format

The serialisation format is the host library's existing checkpoint format extended with the fields listed in §8.1. Two binary layouts are supported:

- **Standard mode:** save all of §8.1. Round-trips Phase D state, post-takeover splines, in-flight head squaring, etc. Use for production checkpoints.
- **Retrofit-source mode:** save only the host library's existing standard layer state (Q/K/V projections, output projection). This is what an externally-trained vanilla softmax model produces. On load into a KAN attention layer, the §8.1 fields are initialised per §8.4.

The two modes are distinguished by a header marker. Loaders auto-detect and dispatch.

### 8.4 Cold-Start Initialisation (Retrofit Path)

When loading a retrofit-source checkpoint into a KAN attention layer:

```
ActiveHeads             := 2
status[h]               := SoftmaxActive             for all h
coefficients[h, :]      := psi_exp_fit_on_grid(grid_spec)  for all h ∈ 0 .. H_max−1
                            -- ψ-space fit per §5.1: ψ(s) ≈ exp(s/2),
                            -- so φ = ψ² ≈ exp(s) over the grid range
KL_ema[h]               := +∞                         for all h
consecutive_low_passes  := 0   for all h
clip_rate_ema           := 0
consecutive_high_sweeps := 0
clip_count_total[h]     := 0
cascade_cap_hits        := 0
rng_state               := derived from a deterministic seed function of the
                            layer's identity (e.g. layer index + grid_spec hash)
                            so retrofit loads from the same source produce
                            identical evolution under identical input streams
```

The first forward pass after a retrofit load produces attention outputs **bit-identical to standard softmax attention** (because all heads are `SoftmaxActive` and `coefficients` are an `exp` fit, but the forward path uses softmax in either case). The deployment can therefore validate correctness on the very first request, before any KAN evolution has occurred.

### 8.5 Save/Load Determinism

A model saved with KAN attention state and reloaded must produce **byte-identical evolution** under any subsequent input sequence (§9). This requires:

- All counters and EMAs are saved at full precision.
- `rng_state` is saved opaquely such that the RNG produces the same sequence after reload.
- Floating-point operations on reload are performed in the same order as before save (a property of the host library's general FP determinism guarantees).

### 8.6 Forward Compatibility

`grid_spec` is saved alongside the coefficients to allow cross-version interoperability. A future spec version that changes basis order or knot count must define an upgrade path: re-fit coefficients on the new grid such that the spline output is preserved over the overlap region. v1 does not define this path; it is reserved for future revisions.

### 8.7 Memory Footprint

Per attention layer:

- `coefficients`: `H_max · N` floats. With defaults `H_max ≈ 256`, `N = 64`: `16 384` floats = 64 KB.
- All counters and EMAs combined: `O(H_max)` ≈ a few KB.
- Negligible relative to Q/K/V projection weights, which are typically `O(d²)` per layer (e.g. for `d = 512`, `d² = 262 144` floats per projection × 4 projections = 4 MB).

---

## 9. Determinism and Thread Safety

### 9.1 Determinism Property

Given:

- A saved per-layer state `S` (per §8.1).
- An identical sequence of input batches `(X_1, X_2, …)`.

Two independent runs of an inference deployment using this spec must produce **byte-identical**:

- Forward outputs at every batch.
- Per-layer state evolution at every batch.
- Handover events and squaring events at the same batch indices.

This property is essential for: regression testing, debugging, A/B reproducibility, regulatory audit trails, and the integrity of the spec's invariant tests (§13).

### 9.2 Sources of Determinism

The spec achieves bitwise determinism by:

- **No wall-clock reads.** No timing function, no `time()`, `clock()`, or platform-specific timer is consulted in any rule.
- **No thread-ID leakage.** The rebalance, learning rule, and gauge operations do not branch on thread identity.
- **No environmental randomness.** No `/dev/urandom`, no platform-RNG calls. All randomness flows from the per-layer seeded `rng_state`.
- **Fixed iteration order in mechanism #1's cascade** (§5.2.6): row-major over coefficient indices.
- **Fixed score-update order in NLMS** (§5.5): row-major over the `L × L` attention score matrix, head-major across heads.
- **Floating-point operations issued in the same sequence on every run.** The host library's existing FP determinism guarantees (e.g. consistent reduction order in vectorised operations) are inherited.

### 9.3 Thread Safety Strategy

The host library runs inference forward passes in parallel across threads in a batch. The KAN attention layer integrates with this in one of two supported strategies (implementation choice; defaults marked):

#### 9.3.1 Strategy A — Batch-Boundary Locking (default)

- Forward passes for samples in a batch run **lock-free** in parallel. Each thread reads the layer's state but does not write.
- The post-forward bookkeeping (steps 6–11 of §7) is **deferred**. Each thread accumulates, in thread-local buffers:
  - the per-coefficient additive NLMS deltas `Δc_j` (a single `H_max × N` float buffer per thread),
  - the per-row KL contributions for `KL_pass`,
  - any per-head clip/lift event counts.
- At batch end, a single per-layer write lock is acquired and a deterministic merge runs:
  1. Sum the thread-local `Δc_j` buffers across threads in **sample-index order** within the batch (addition is associative; the fixed order makes the FP result deterministic).
  2. Apply the summed delta to the layer's coefficients in a single sweep: `c_j ← c_j + Σ_threads Δc_j`.
  3. Run **one** mechanism-#1 sweep (cascading to fixed point) for the entire batch, not one per sample.
  4. Run **one** GM=1 gauge step for the entire batch.
  5. Update `KL_ema` from the summed `KL_pass` contributions.
  6. Run handover check and squaring check once.
- Lock acquisitions: O(1) per batch per layer.

**Dynamics note.** This batch-amortised application produces dynamics that differ in detail from a hypothetical fully-serial inference (where mechanism #1 + gauge would run after every sample's update). Both behaviours are stable, deterministic, and respect the same conservation laws; the batch-amortised version is the canonical v1 dynamics, and the serial-inference dynamics are not used by any part of the spec.

#### 9.3.2 Strategy B — Per-Thread Shadow Copies

- Each thread maintains a per-thread shadow copy of the layer state.
- Forward passes use the shadow copy.
- Shadow copies are merged at synchronisation points by averaging or last-writer-wins.

More complex; only worth implementing if profiling shows Strategy A's per-batch lock is a measured bottleneck. v1 ships only Strategy A.

### 9.4 Determinism Across Hardware

This spec does not guarantee bitwise determinism **across heterogeneous hardware** (different CPU instruction sets, different SIMD widths, GPU vs CPU). The guarantee is only for runs on the same hardware, same compiler, same library build.

For cross-hardware reproducibility, a stricter mode (round all FP operations to a canonical precision, force a specific FP reduction order) would be required. v1 does not include this; it is noted as a possible v2 extension.

### 9.5 Verification

A deterministic-evolution test is mandatory in the spec's test suite (§13.2):

```
Given: a saved state S, an input stream X.
Run 1: load S, replay X, save the resulting state S'_1 and outputs Y_1.
Run 2: identical to run 1 in a separate process, producing S'_2 and Y_2.
Assert: S'_1 == S'_2 (bit-identical) and Y_1 == Y_2 (bit-identical).
```

A failure indicates a determinism leak somewhere in the implementation; the test must be treated as load-bearing infrastructure, not as a soft check.

---

## 10. Invariants and Guarantees (Consolidated)

This section consolidates every invariant introduced in §5–§9 into a single canonical list. Implementations must satisfy all of these. The test suite (§12) provides one or more checks for each.

### 10.1 Mechanism #1 Invariants (per head, after each pipeline step)

- **(M1-1) Per-window product preservation.** For every coefficient `c_i` and its window `W(i)`, `Π_{j ∈ W(i)} |c_j|` is unchanged across a single mechanism-#1 sweep, up to floating-point tolerance.
- **(M1-2) Sign preservation.** No mechanism-#1 operation flips the sign of any coefficient.
- **(M1-3) Bounded coefficients.** After cascade termination, every coefficient satisfies `g/2 ≤ |c_i| ≤ 3·g` where `g` is its window's geometric mean.
- **(M1-4) Scale equivariance.** For any `α > 0`, mechanism #1 applied to `(α · c_j)` produces `α · (mechanism #1 applied to (c_j))`.

### 10.2 GM=1 Gauge Invariants (per head, after pipeline step 9)

- **(GM-1) Gauge fixed.** `|G − 1| < ε_gauge`, where `ε_gauge` is at machine-precision tolerance for the head's coefficient count.
- **(GM-2) Forward output preservation.** The attention weights computed before and after gauge renormalisation are mathematically equal; the per-weight FP error from the rescale is bounded by `O(N · ε_machine)`.
- **(GM-3) Sign preservation.** Gauge renormalisation preserves coefficient signs.

### 10.3 Head-Squaring Invariants (per layer, across the run)

- **(HS-1) Bounded.** `2 ≤ ActiveHeads ≤ H_max` at all times.
- **(HS-2) Power-of-2 progression.** `ActiveHeads ∈ {2, 4, 16, 256, …}` at all times, including when clamped at `H_max` (which is itself constrained to be a power of 2; §5.4.2).
- **(HS-3) Inactive masking.** Heads with index `≥ ActiveHeads` contribute zero to the layer's forward output.
- **(HS-4) Deterministic squaring.** Squaring transitions are deterministic given the per-layer RNG state.
- **(HS-5) Monotonicity.** `ActiveHeads` is monotonic non-decreasing across a deployment run.
- **(HS-6) Child inheritance.** A child head spawned at squaring inherits its parent's `status`, `KL_ema`, `consecutive_low_passes`, and `clip_count_total`; only the spline coefficients are perturbed.

### 10.4 Local-Learning Invariants (per coefficient, per update)

- **(LL-1) Locality.** Each NLMS update touches at most `k+1` coefficients per observed score.
- **(LL-2) Determinism.** Each update is deterministic given `(s, target, denom, η, current coefficients)`.
- **(LL-3) Sign preservation in expectation.** Mechanism #1 immediately corrects any pathological single-step sign flip via low-lift.
- **(LL-4) No autograd.** No gradient state, optimiser state, or autograd graph is constructed.

### 10.5 Lifecycle Invariants (per head, per layer, across the run)

- **(LC-1) Phase monotonicity.** `status` transitions only `SoftmaxActive → KANActive`, never the reverse.
- **(LC-2) Per-head independence.** No layer-level synchronisation gates per-head transitions.
- **(LC-3) Atomic handover.** A head's `status` flips between two consecutive forward passes; no intermediate state is observable.
- **(LC-4) Retrofit identity.** At first forward pass after a retrofit load, the layer's output equals standard softmax attention's output to the precision of the `exp` grid fit.

### 10.6 Determinism Invariants (per deployment run)

- **(DT-1) State-determined evolution.** Given identical saved state and identical input sequences, two runs produce bit-identical outputs and bit-identical post-state.
- **(DT-2) No environmental randomness.** All randomness in the layer derives from the saved per-layer RNG.
- **(DT-3) Save/load round-trip.** A model checkpointed mid-run and reloaded continues evolution as if no save/load had occurred.
- **(DT-4) Batch-amortised application.** Under Strategy A (§9.3.1), per-batch state mutation applies the sum of thread-local NLMS deltas (in fixed sample-index order) followed by exactly one mechanism-#1 sweep and one gauge step per batch. This is the canonical v1 dynamics; serial-per-sample dynamics are not part of the spec.

### 10.7 Compositional Invariants (across mechanisms)

- **(C-1) Output gauge invariance.** The forward attention output is invariant under uniform positive rescaling of any head's coefficients (the property GM=1 exploits).
- **(C-2) Productivity-squaring coupling.** The aggressive low-lift threshold (`g/2`) is a precondition for the meaning of the squaring trigger; without it, squaring fires on wasted-budget noise rather than genuine capacity demand.
- **(C-3) Sharpening-budget coupling.** The Phase D self-distillation rule cannot collapse to one-hot because mechanism #1 + GM=1 together pin the multiplicative budget; sharpening can only redistribute mass within the budget.

---

## 11. Parameters and Defaults

All parameters are per-layer-overridable. Defaults are tuned for typical transformer attention shapes (`d ∈ [128, 1024]`, `L ∈ [128, 4096]`).

### 11.1 B-Spline Configuration

| Parameter | Symbol | Default | Notes |
|---|---|---|---|
| Basis order | `k` | 3 | Cubic. Window size = `2k+1 = 7`. |
| Knot count | `N` | 64 | Resolution of the spline; trade off vs memory. |
| Grid range | `[s_low, s_high]` | `[−8, +8]` nats | Covers typical `Q·Kᵀ/√d` range. Re-tune per layer if scores known to live elsewhere. |
| Non-negativity | — | `φ = ψ²` parametrisation | Ensures `φ(s) ≥ 0` strictly. |

### 11.2 Mechanism #1 Thresholds

| Parameter | Default | Notes |
|---|---|---|
| High-clip ratio | 3 | `|c_i| > 3·g` triggers high clip. Outlier detection. |
| Low-lift ratio | 2 (i.e. `g/2`) | `|c_i| < g/2` triggers low lift. Productivity-tuned. |
| Cascade iteration cap | 16 | Safety against pathological loops. Typical convergence: 1–3. |

### 11.3 Auto-Mode Thresholds

| Parameter | Default | Notes |
|---|---|---|
| Calm boundary | `τ_calm = 0.02` | `clip_rate < 2 %` → Mode A. |
| Stressed boundary | `τ_stressed = 0.10` | `clip_rate > 10 %` → Mode C. |
| Mode hysteresis | hysteresis between `τ_calm` and `τ_stressed` | Avoids rapid mode flapping near boundary. |

### 11.4 Local-Learning Parameters

| Parameter | Symbol | Default | Notes |
|---|---|---|---|
| NLMS step | `η` | `1.0` | NLMS auto-scales via basis-norm denominator. Effective per-step magnitude is auto-tuned. |
| NLMS denominator guard | `δ` | `1e-6` | Prevents divide-by-zero when basis activations vanish. |
| Sharpening power (Phase D) | `α` | `1.1` | Gentle per-step push. Mechanism #1 enforces budget. |

### 11.5 Convergence and Handover

| Parameter | Default | Notes |
|---|---|---|
| KL convergence threshold | `ε_KL = 0.01 nats` | Tight per-pass distribution match. |
| KL EMA decay | `λ = 0.01` | Effective horizon ≈ 100 passes. |
| Sustained passes for handover | `N_confirm = 100` | Avoids handover on lucky single-batch noise. |
| Optional task-loss tolerance | `ε_task = max(0.005, 0.01·L_task)` | Used only if labelled eval stream available. |
| Optional task-loss eval cadence | `K_eval = 100 batches` | Same. |

### 11.6 Head-Squaring Parameters

| Parameter | Default | Notes |
|---|---|---|
| Squaring trigger ratio | `τ_squaring = 0.10` | `clip_rate_ema > 10 %`. |
| Sustained sweeps for squaring | `K_squaring = 64` | Avoids noise-driven squaring. |
| Initial active heads | 2 | Specification mandate. |
| Maximum active heads | `H_max = largest power of 2 ≤ ⌊embedding_dim / 2⌋` | Architectural ceiling: ≥ 2 channels per head, power of 2 to keep squaring integer. |
| Head-split perturbation σ | `σ = 0.01` | Log-normal multiplicative perturbation on cloned coefficients. |

### 11.7 Determinism / Thread Safety

| Parameter | Default | Notes |
|---|---|---|
| Threading strategy | A (batch-boundary lock) | See §9.3.1. Strategy B reserved for v2. |
| Per-layer RNG seed | derived from layer identity + `grid_spec` hash | Ensures retrofit reproducibility across deployments. |

### 11.8 Configuration Flags

| Flag | Default | Effect |
|---|---|---|
| `disable_kan_at_inference` | `false` | If `true`, skip all post-forward state mutation; behaves exactly as standard softmax attention. Use for emergency rollback. |
| `force_mode` | `Auto` | Override the per-layer Auto rule; can pin the layer to Mode A, B, C, or D for ablation studies. |
| `pin_active_heads` | none | If set to an integer `≥ ActiveHeads`, disable squaring while still permitting evolution within the pinned head count. |
| `freeze_after_takeover` | `false` | If `true`, disable Phase D evolution once a head has handed over (Phase D becomes a no-op for the local rule; mechanism #1 and gauge still run). For benchmarking the static-KAN baseline. |
| `skip_redundant_softmax` | `true` | Skip `w_softmax` computation in step 3 of the pipeline once a head has handed over. Reduces post-takeover overhead. |

### 11.9 Tuning Notes

- **`α` (sharpening power)** is the primary knob for "how aggressively does the KAN diverge from softmax." `α = 1.0` makes Phase D a no-op; `α = 1.1` is the spec default; `α = 1.5` produces aggressive divergence and may saturate mechanism #1.
- **`τ_squaring`** trades head-growth speed against the meaningfulness of each squaring event. Lower values grow heads faster (capacity comes online sooner); higher values demand stronger evidence before adding capacity.
- **`N_confirm`** is the primary knob for "how cautious is handover." `N_confirm = 100` is the default; `N_confirm = 1000` is ultra-cautious; `N_confirm = 10` is aggressive (recommended only for testing).
- **`H_max`** can be reduced below `embedding_dim / 2` to limit the worst-case memory cost of a layer that aggressively squares; at the cost of forcing the failure mode (§5.4.8) to engage sooner under capacity pressure.

---

## 12. Testing Strategy

The test suite is structured in three tiers. Tier 1 (Invariant Tests) verifies that the spec's mathematical guarantees hold. Tier 2 (Functional Tests) verifies that the design dynamics behave as intended. Tier 3 (Integration Tests) verifies end-to-end behaviour on realistic models. **Every invariant in §10 must have at least one corresponding tier-1 test.**

### 12.1 Tier 1 — Invariant Tests

| ID | Verifies | Description |
|---|---|---|
| **IT-M1** | M1-1 | After any single mechanism-#1 sweep on a randomised coefficient set, every window's `Π|c_j|` is preserved within `1e-6` relative tolerance. |
| **IT-Sign** | M1-2, GM-3 | Run mechanism #1 + gauge on randomised coefficients (including negatives); assert no sign flips. |
| **IT-Bound** | M1-3 | After cascade termination, every coefficient lies in `[g/2 − ε, 3·g + ε]` for its window's `g`. |
| **IT-Scale** | M1-4 | Apply mechanism #1 to `(c_j)` and to `(α · c_j)` for `α ∈ {0.1, 1, 10, 100}`; assert results differ by exactly factor `α`. |
| **IT-Gauge** | GM-1 | After step 9 of the pipeline, per head, `|G − 1| < 1e-6`. |
| **IT-GaugeOutput** | GM-2 | The forward attention weights computed with raw coefficients vs renormalised coefficients differ by less than FP rounding noise. |
| **IT-HSBound** | HS-1, HS-2 | Across a synthetic input stream forcing many squaring events, `ActiveHeads` always lies in the allowed power-of-2 progression and never exceeds `H_max`. |
| **IT-HSMask** | HS-3 | With `ActiveHeads = 4` and `H_max = 16`, the layer's forward output equals the layer's forward output computed only over heads 0–3. |
| **IT-HSMonotonic** | HS-5 | Across any input stream, `ActiveHeads` never decreases. |
| **IT-LLLocal** | LL-1 | An NLMS update on score `s` modifies only the `k+1` coefficients with `B_j(s) ≠ 0`. |
| **IT-LCMonotonic** | LC-1 | Across any input stream, `status` for any head transitions only `SoftmaxActive → KANActive`, never reverse. |
| **IT-Retrofit** | LC-4 | Load a vanilla softmax checkpoint; first forward output equals standard softmax attention's output to within `1e-4` (precision of the `exp` grid fit). |
| **IT-Det** | DT-1, DT-3 | Run the determinism verification protocol of §9.5; assert byte-identical state and outputs across two independent runs. |

### 12.2 Tier 2 — Functional Tests

| ID | Verifies | Description |
|---|---|---|
| **FT-Mimic** | Phase M dynamics | On a stationary score distribution, `KL_ema` falls below `ε_KL` within a bounded number of inferences (e.g. `5 · N_confirm`). |
| **FT-Handover** | §6.3 | Once `KL_ema` is below threshold for `N_confirm` passes, the head's `status` flips between two consecutive passes (atomicity). |
| **FT-Phase D** | §5.5.4 | After takeover, on a stream where one specific score region carries genuine signal, `φ` develops a peak at that region within a bounded number of inferences. |
| **FT-NoCollapse** | §5.5.4, C-3 | After extended Phase D running, no head collapses to a one-hot attention distribution; entropy of the attention output remains above a small floor. |
| **FT-Squaring** | §5.4.3 | A synthetic input stream engineered to force high `clip_rate` triggers `ActiveHeads` growth within `K_squaring` consecutive sweeps. |
| **FT-SquaringSelfLimit** | §5.4.5 | After two squaring events, `clip_rate_ema` falls below `τ_squaring` and `ActiveHeads` freezes. |
| **FT-ModeAbsent** | Mode D exclusion | Run the layer with all default settings; assert `(HighClipShare, LowLiftShare) = (Proportional, InverseProportional)` (Mode D) is never selected by Auto. |
| **FT-ModeABDifferent** | §5.2.5 | Pin Mode A vs Mode C via `force_mode`; on identical input streams, assert coefficient trajectories diverge measurably (sanity check that the mode flag is load-bearing). |

### 12.3 Tier 3 — Integration Tests

| ID | Verifies | Description |
|---|---|---|
| **INT-VanillaParity** | End-to-end retrofit | Train a small softmax transformer on a benchmark (e.g. CIFAR-10 patch-classifier or a MNIST-scale text task); retrofit; run inference; assert task metrics after Phase-M convergence are within statistical noise of the original softmax model. |
| **INT-PhaseDImproves** | Information surfacing | Same model as INT-VanillaParity, run for an extended Phase-D period on the test set; assert task metrics either improve or are preserved (no regression). |
| **INT-Checkpoint** | DT-3 | Mid-run, save a checkpoint, kill the process, reload, continue. Compare the resulting state and outputs after another N batches against a non-interrupted run. Assert byte-identical. |
| **INT-MultiLayer** | Per-layer independence | A multi-layer transformer must show different layers in different phases at the same wall-clock time, and `ActiveHeads` distributions vary per layer based on layer-specific data complexity. |
| **INT-DisableSwitch** | `disable_kan_at_inference` | Setting the kill switch makes the layer behaviourally identical to standard softmax attention. |

### 12.4 Test Infrastructure Requirements

- **Reproducible RNG.** All tests must use seeded RNGs; failures must be reproducible from the seed alone.
- **FP tolerances documented.** Each test specifies its allowed FP tolerance; tolerances differ between strict bit-identity checks (DT-1) and approximate equality checks (INT-VanillaParity).
- **Synthetic input generators.** Tier 1 and Tier 2 tests use constructed inputs (not real data) to isolate the property under test.
- **Performance regression tests** (separate suite, not gating): verify the per-pass overhead targets in §7.2 are met within tolerance.

### 12.5 What Is Not Tested at the Unit Level

- **Soundness of Phase D divergence on real-world data distributions.** This requires INT-tier benchmarks against held-out evaluation sets and is fundamentally empirical, not a property the spec can assert by construction.
- **Performance characteristics on specific hardware.** Performance tests are platform-specific and are not part of the property-test suite.

---

## 13. Roadmap

The spec ships in three tiers of increasing scope. v1 is the focus of the initial implementation; v2 and v3 are reserved for future revisions and may evolve as v1 deployment experience accumulates.

### 13.1 v1 — Softmax-Replacement KAN (this spec)

Scope:

- KAN normaliser replacing softmax only.
- Q/K/V and output projections remain standard linear layers.
- Mimicry → takeover → divergence per-head lifecycle.
- Mechanism #1 with all four modes plus Auto.
- GM=1 gauge fixing.
- Head squaring with self-limiting dynamics.
- Determinism Strategy A (batch-boundary lock).
- Retrofit path for any pre-trained softmax transformer.

Deliverable: one new attention layer class, one new network-builder method, full test suite per §12.

### 13.2 v2 — KAN-ified Q/K/V Projections

Scope:

- Extend the deferred-mimicry-then-takeover lifecycle to Q, K, V projections.
- Each projection gains a per-head KAN edge that learns to mimic its trained linear projection at inference, then takes over.
- Post-takeover divergence rule for projections: **attention-coupled** — reinforce projections that contribute to sharpening (measured by entropy of the post-softmax attention output). This couples the projection KANs and the normaliser KAN toward the shared "surface hidden information" goal.
- Same mechanism #1, gauge, head squaring infrastructure inherited from v1.

Deliverable: extended attention layer class supporting full-KAN at inference. Strategy B (per-thread shadow copies) implementation if profiling demands it.

### 13.3 v3 — Cross-Layer Coupling (Speculative)

Scope:

- Layers that have handed over broadcast a coarse "attention sharpness" signal to adjacent layers.
- Adjacent layers' Auto-mode and squaring decisions become functions of (own state, neighbour states).
- Goal: coordinated capacity allocation across the stack rather than per-layer independent dynamics.

Deliverable: TBD pending v1 + v2 deployment experience.

### 13.4 Out of Scope (No Planned Version)

- **Training-time KAN.** The spec's "vanilla training, KAN at inference" stance is permanent. No version will introduce KAN state at training time.
- **Cross-hardware bitwise determinism.** The same-hardware property of §9.4 is sufficient for almost all use cases. A canonical-precision mode is not planned.
- **Head merging.** The one-way head-squaring property of §5.4.7 is permanent. To reduce head count, load a checkpoint from before the squaring event.

---

## 14. Integration Notes (Library-Abstract)

This section describes how the spec integrates with a host attention library, abstracted across implementation languages.

### 14.1 Touch Surface

The spec is a **single new attention normaliser layer**, plus one new network-builder convenience method that wraps the existing attention construction with the new normaliser substituted for the existing softmax operator.

| Component | Status |
|---|---|
| Q/K/V projections | **Reused unchanged** from the host library's existing attention. |
| Score computation `Q·Kᵀ/√d` | **Reused unchanged**. |
| Row-normalisation step | **Replaced** by the new normaliser layer. |
| Multi-head concat | **Reused unchanged**, with the new layer's mask of inactive heads. |
| Output projection `W_O` | **Reused unchanged**. |
| Forward / backward training paths | **Reused unchanged** (training is vanilla softmax). |
| Save / load infrastructure | **Extended** by the per-layer state schema of §8.1, with the retrofit-source compatibility of §8.3. |

The total new code surface is on the order of **one source file** in the host library, encapsulating the new layer class and its supporting state-management helpers.

### 14.2 Construction API (Abstract)

A new builder method should mirror the host library's existing attention construction signature, differing only in the choice of normaliser:

```
AddKANSelfAttention(
    Heads: int = 2,                    -- initial active head count
    HeadCeiling: int = embedding_dim/2,  -- H_max
    GridLow: float = -8.0,
    GridHigh: float = 8.0,
    GridKnots: int = 64,
    BasisOrder: int = 3,
    HighClipShare: enum = Auto,
    LowLiftShare: enum = Auto,
    SharpeningPower: float = 1.1,
    KLThreshold: float = 0.01,
    KLConfirmPasses: int = 100,
    SquaringClipRate: float = 0.10,
    SquaringSweeps: int = 64,
    DisableKANAtInference: bool = false,
    ForceMode: enum = Auto,
    PinActiveHeads: int = 0,           -- 0 means use squaring
    FreezeAfterTakeover: bool = false
)
```

The host library's existing `AddSelfAttention` remains unchanged. Both methods can coexist; users opt in to the KAN normaliser by calling the new builder.

### 14.3 Inference-Mode Flag

The host library typically distinguishes training from inference via a `network.inference = true` (or similar) flag set before serving. The KAN attention layer must consult this flag and:

- During training (`inference = false`): forward pass uses softmax exactly; all KAN bookkeeping (steps 6–11 of §7) is **skipped**. Backward pass flows through softmax normally. The layer is functionally identical to the host library's standard attention.
- During inference (`inference = true`): the full §7 pipeline is engaged.

Switching the flag mid-deployment is supported but should be rare; flipping back to `inference = false` during a deployed run is undefined behaviour in this spec.

**Distinction from `disable_kan_at_inference`.** The two flags address different concerns and are not interchangeable:

- `inference` is the **host library's training-mode flag**. It decides whether the layer is in training (softmax forward, vanilla backward, no KAN bookkeeping) or inference (full §7 pipeline). It is set once before serving and should not be flipped during a deployed run.
- `disable_kan_at_inference` is the **spec's emergency rollback flag** (§11.8). It runs *within* inference mode and disables the post-forward state mutation (steps 6–11 of §7), causing the layer to behave exactly as standard softmax attention while remaining "in inference" from the host library's perspective. It can be flipped at any time during a deployed run; flipping it on freezes the KAN state at its current value, flipping it off resumes evolution from that frozen state.

Operators rolling back KAN behaviour mid-deployment should always reach for `disable_kan_at_inference`, never for the `inference` flag.

### 14.4 Device Placement

This spec targets CPU implementation and assumes the host library's existing CPU code paths. GPU implementation is **not** in v1's scope. The spec's data structures (per-head coefficient arrays, counters, RNG state) are CPU-resident and small enough that this is not a memory-pressure concern even for large models.

A future GPU port would require revisiting §9 (determinism) and §5.5 (NLMS update ordering); the per-coefficient locality of the local rules is friendly to parallelisation but the deterministic ordering guarantees would need to be re-established under GPU execution model.

### 14.5 Logging and Telemetry

Recommended (not mandatory) per-layer telemetry the implementation should expose:

- `KL_ema[head]` — current convergence metric per head.
- `status[head]` — current phase per head.
- `clip_rate_ema` — current per-layer clip pressure.
- `ActiveHeads` — current head count.
- `clip_count_total[head]` — diagnostic since last reset.
- Phase transitions (handover, squaring) as discrete events with timestamps.

These are essential for diagnosing deployments and for the "diagnostic value" property of §5.4.9. The host library's existing logging hooks should be reused.

### 14.6 Failure Modes and Operator Guidance

Documented behaviours an operator may observe in production:

| Symptom | Likely cause | Operator response |
|---|---|---|
| `KL_ema` plateaus high, never converges | Score range exceeds the spline grid `[s_low, s_high]` | Adjust grid range per layer, reload from retrofit-source checkpoint |
| `ActiveHeads` reaches `H_max` and `clip_rate_ema` stays high | Layer under-provisioned for data complexity | Architectural change required (more channels), or accept degraded regime |
| `KL_ema` oscillates around threshold | `N_confirm` too small | Increase `N_confirm`; or use `force_mode = A` to stabilise |
| Post-takeover task metrics regress | Phase D divergence too aggressive for task | Reduce `α` toward 1.0; or set `freeze_after_takeover = true` |
| Need to roll back KAN behaviour | Any of the above | Set `disable_kan_at_inference = true` for the affected layer; falls back to standard softmax attention |

---

*End of specification.*












